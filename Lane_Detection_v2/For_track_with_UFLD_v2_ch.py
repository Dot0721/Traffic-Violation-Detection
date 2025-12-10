import torch, os, sys
import time
import shutil
from pathlib import Path
import subprocess
import cv2 as cv
import re
from cv2 import dnn_superres
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict, deque
import scipy.special, tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils.dist_utils import dist_print
from utils.common import merge_config, get_model
from utils.lane import pred2coords
from data.dataset import LaneTestDataset
import torchvision.transforms as transforms
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from itertools import combinations

sr_dnn = dnn_superres.DnnSuperResImpl_create()
sr_dnn.readModel("weights/EDSR_x2.pb")
sr_dnn.setModel("edsr", 2)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
# --- 路徑 ---
PENDING_EXECUTION_DIR   = Path("./pending_execution")
OUTPUT_ROOT    = Path("./output")     # 暫存輸出根目錄）
VIOLATION_ROOT = Path("./violations") # 有違規才搬到這（依日期/檔名分層）
PROCESSED_ROOT = Path("./processed")  # 無違規則搬到這
DONE_MARK      = ".done"              # 處理過標記檔

# --- 設定 ---
VIDEO_EXTS        = {".mp4", ".mov", ".avi", ".mkv"}
POLL_INTERVAL_SEC = 2    # 每幾秒掃一次資料夾
STABLE_CHECKS     = 3
STABLE_SLEEP_SEC  = 1

# --- 影像參數 ---
START_SEC = 6
END_SEC   = 15
VIOLATION_MAP = {
    "signal":  {"code": 42, "desc": "汽車駕駛人，不依規定使用燈光者。"},
    "crossing":{"code": 48-1-2, "desc": "汽車駕駛人，跨越禁止變換或禁止超車之標線。"}
}

def is_file_ready(p):
    last_size = -1
    stable = 0
    
    for _ in range(STABLE_CHECKS):
        sz = p.stat().st_size
        if sz == last_size:
            stable += 1
        else:
            stable = 0
        last_size = sz
        time.sleep(STABLE_SLEEP_SEC)
        
    return stable >= (STABLE_CHECKS - 1)

def already_done(video_path):
    return video_path.with_suffix(video_path.suffix + f".{DONE_MARK}").exists()

def mark_done(video_path):
    done_file = video_path.with_suffix(video_path.suffix + f".{DONE_MARK}")
    done_file.touch()

def scan_new_video():
    vids = []
    for p in PENDING_EXECUTION_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not already_done(p) and is_file_ready(p):
            vids.append(p)

    return sorted(vids, key = lambda x:x.stat().st_mtime)

def check_dirs():
    PENDING_EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIOLATION_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

def write_txt(violations, out_dir, file_name):
    if not violations:
        return None

    lines = []

    for v in violations:
        lines.append(f"Violation: {v.get('code')}")
        lines.append(f"license_plate: {v.get('license_plate')}")
        lines.append(f"details: {v.get('desc')}")

    txt_path = out_dir / file_name
    with open(txt_path, "w", encoding = "utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

def process_video(video_path: Path):

    out_dir = OUTPUT_ROOT / video_path.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR Cannot open video: {video_path}")
        return (False, out_dir)
    
    cap.set(cv.CAP_PROP_POS_MSEC, START_SEC * 1000)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_width= int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    scale = min(1920/frame_width, 1080/frame_height)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out_video_path = out_dir / f"{video_path.stem}_out.mp4"
    out = cv.VideoWriter(str(out_video_path), fourcc, fps, (int(scale * frame_width), int(scale * frame_height)))

    monitor = YOLOv8(fps)
    violations = []
    recorded_violations = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.resize(frame, (int(scale * frame_width), int(scale * frame_height)))
            frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            cur = frame_num / fps   

            vehicle_index = monitor.process_frame(cur, frame, int(scale * frame_width), int(scale * frame_height), fps)
            lane_frame = monitor.lanedete(frame, net, img_transforms, cfg, cls_num_per_lane, int(scale * frame_height), int(scale * frame_width), vehicle_index, fps)
        
            y_position = 20
            line_spacing = 75
            
            font_size_large = 55
            header_drawn = False

            try:
                font_large = ImageFont.truetype(monitor.font_path, font_size_large)
            except IOError:
                print(f"警告：找不到字型檔 '{monitor.font_path}'。使用預設字型。")
                font_large = ImageFont.load_default()
            
            messages_to_draw = []

            for v_type, data in monitor.violation.items():
                if 'vid' in data and data['vid']:
                    for vid in data['vid']:

                        if vid in monitor.tracked_vehicles:
                            vehicle_index.remove(vid)
                            lane_frame = monitor.draw_info(lane_frame, vid, True)
                            license_plate = monitor.tracked_vehicles[vid]['license_plate']
                            
                            if v_type == 'signal':
                                message = f"{license_plate}: 變換車道沒有使用方向燈"
                            elif v_type == 'crossing':
                                message = f"{license_plate}: 不依標線指示，跨越實線!"

                            violation_key = (vid, v_type)

                            if violation_key not in recorded_violations:
                                violations.append({
                                    "license_plate": license_plate,
                                    "code": VIOLATION_MAP[v_type]['code'],
                                    "desc": VIOLATION_MAP[v_type]['desc']
                                })
                                recorded_violations.add(violation_key)

                            messages_to_draw.append((message, (255, 255, 255)))

            if messages_to_draw:
                messages_to_draw.insert(0, ("=== 違規車輛 ===", (255, 255, 255)))

                img_rgb = cv.cvtColor(lane_frame, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                bg_padding = 15
                bg_x, bg_y = 10, 10
                bg_height = len(messages_to_draw) * line_spacing + bg_padding * 2
                bg_width = 900 

                overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0)) #創建透明圖層(RGBA)
                draw_overlay = ImageDraw.Draw(overlay)
                
                draw_overlay.rectangle(
                    (bg_x, bg_y, bg_x + bg_width, bg_y + bg_height),
                    fill=(50, 50, 50, 200)#fill = (R, G, B, A) Alpha值0-255，200大約是80%不透明度
                )
                pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)

                draw = ImageDraw.Draw(pil_img)
                text_x = bg_x + bg_padding
                current_y = bg_y + bg_padding + 5

                for text, color in messages_to_draw:
                    draw.text((text_x, current_y), text, font=font_large, fill=color)
                    current_y += line_spacing

                lane_frame = cv.cvtColor(np.array(pil_img), cv.COLOR_RGBA2BGR)

            for vid in vehicle_index:
                lane_frame = monitor.draw_info(lane_frame, vid, False)
            out.write(lane_frame)
            #cv.imshow('Traffic Monitor', lane_frame)
            if cur >= END_SEC:
                print("Video end")
                break

            if cv.waitKey(int(1000/fps)) & 0xFF == 'q':
                print("Program interrupted.")
                break


    except Exception as e:
        print(f"Exception while processing {video_path.name}: {e}")

    finally:
        out.release()
        cap.release()
        cv.destroyAllWindows()

    check_violation = len(violations) > 0
    if check_violation:
        write_txt(violations, out_dir, file_name = "violations.txt")
        final_dir = VIOLATION_ROOT / video_path.stem
    else:
        final_dir = PROCESSED_ROOT / video_path.stem

    if final_dir.exists():
            shutil.rmtree(final_dir)
    shutil.move(str(out_dir), str(final_dir))
    
    return (check_violation, final_dir)


def lanemodel():
    args, cfg = merge_config()
    cfg.batch_size = 32
    #print('Setting batch_size to 1 for demo generation')

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval().cuda()

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return net, img_transforms, cls_num_per_lane, cfg
    
class YOLOv8:
    def __init__(self, fps):
        self.det_track = YOLO(r"weights/all_best.pt")
        self.det_part  = YOLO(r"weights/all_best.pt")
        self.font_path = "./font/msjh.ttc"
        self.name2id = {v:k for k, v in self.det_track.names.items()}
        self.veh_ids  = [self.name2id[x] for x in ('bus', 'car', 'cementTruck', 'heavyTruck', 'lightTruck', 'tanker') if x in self.name2id]
        self.plate_ids = [self.name2id[x] for x in ['licencePlate'] if x in self.name2id]
        #--------
        self.light_model = YOLO(r"weights/light_best.pt")
        self.ransac_model = RANSACRegressor()
        self.tracked_vehicles = defaultdict(lambda: {
            'id': None,
            'bbox': None,
            'cx' : None,
            'license_plate': None,
            'license_coord_ratio': None,
            'left_signal': 'OFF',
            'right_signal': 'OFF',
            'signal_region': {'left': None, 'right': None},
            'signal_brightness': {'left': 0, 'right': 0},
            'signal_history': deque(maxlen = int(fps * 5)),
            'signal_duration': 0,
            'lane_changing' : 'stable',
            'touch_line_code': None,
            'lane_touching_frames' : 0,
            'direction_change' : None
        })
        self.check_timethreshold = int(0.25 * fps)
        self.lane = {
            'L1': {'pts': None, 'type': "Unknown"},
            'R1': {'pts': None, 'type': "Unknown"},
        }
        self.violation = {
            'signal': {'vid': []},
            'cross': {'vid': []}
        }
    
    def process_lane_image(self, img_path, img_transform, crop_size):
        img = img_path
        w,h = img.shape[:2]   
        blurred = cv.GaussianBlur(img, (5, 5), 0)
        lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv.merge((l_eq, a, b))
        enhanced = cv.cvtColor(lab_eq, cv.COLOR_LAB2BGR)
        enhanced=cv.GaussianBlur(enhanced, (5, 5), 0)
        process = cv.cvtColor(enhanced, cv.COLOR_BGR2GRAY)
        low_threshold = 40
        high_threshold = 120
        edges = cv.Canny(process, low_threshold, high_threshold)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        edges_close = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(edges_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(edges_close, contours, -1, 255, thickness=cv.FILLED)
        non_edge_mask = (edges_close == 0)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv[..., 2][non_edge_mask] = hsv[..., 2][non_edge_mask] * 0.7
        mask = edges_close > 0
        hsv[..., 1][mask] = np.clip(hsv[..., 1][mask] * 1.5, 0, 255)
        hsv[...,2][mask] = np.clip(hsv[...,2][mask] * 1.2, 0, 255)
        processed_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        img_pil = Image.fromarray(processed_image)
        img_tensor = img_transform(img_pil)
        img_tensor = img_tensor[:, -crop_size:, :]
        return img_tensor.unsqueeze(0)

    def lane_filter(self, points, dist_thresh=80, threshold_deg=25):
        filtered = [points[0]]
        for i in range(1, len(points) - 1):
            prev = filtered[-1]
            cur = points[i]
            nxt = points[i + 1]
            dist = np.linalg.norm(np.array(cur) - np.array(prev))
            if dist > dist_thresh:
                continue
            v1 = np.array([cur[0] - prev[0], cur[1] - prev[1]])
            v2 = np.array([nxt[0] - cur[0], nxt[1] - cur[1]])
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
            if angle > threshold_deg:
                continue
            filtered.append(cur)
        return filtered
    
    def check_violation(self,best_code, direction, vid, fps):# !best_code會用到只是現在測試東西才先註解掉之後記得改回來
        lane_type = "Dotted_Yellow_Line"#self.lane[best_code]['type']  #先測試用而已之後記得改回來
        if lane_type == "Solid_Yellow_Line" or lane_type == "Solid_White_Line":
            if vid not in self.violation['cross']['vid']:
                self.violation['cross']['vid'].append(vid)

        if direction == "left":
            signal_state = self.tracked_vehicles[vid]['left_signal']
        else:
            signal_state = self.tracked_vehicles[vid]['right_signal']

        if self.tracked_vehicles[vid]['signal_duration'] >= int(0.6 * fps) and signal_state == "FLASHING":
            self.tracked_vehicles[vid]['signal_duration'] -= 1
        elif signal_state != "FLASHING":
            self.tracked_vehicles[vid]['signal_duration'] += 1
            if self.tracked_vehicles[vid]['signal_duration'] >= self.check_timethreshold:
                if vid not in self.violation['signal']['vid']:
                    self.violation['signal']['vid'].append(vid)
        else:
            self.tracked_vehicles[vid]['signal_duration'] = 0
            self.check_timethreshold = int(0.9 * fps)

    def check_lane_change(self, image, line_dict, fps, vehicle_index, draw_frame):
        def x_on_polyline_at_y(polyline, y):
            pl = np.asarray(polyline, dtype=np.float32).reshape(-1,2)
            best = None
            best_dy = 1e9
            for i in range(len(pl)-1):
                (x1, y1),(x2, y2) = pl[i], pl[i+1]

                if y1 == y2:
                    dy = abs(y - y1)
                    if dy < best_dy:
                        best_dy = dy
                        best = (x1+x2)/2.0
                    continue
                lo, hi = (y1, y2) if y1<=y2 else (y2,y1)
                if lo <= y <= hi:
                    t = (y - y1) / (y2 - y1)
                    x = x1 + t*(x2 - x1)
                    return float(x)

                dy = min(abs(y - y1), abs(y - y2))
                if dy < best_dy:
                    best_dy = dy
                    best = x1 if abs(y-y1) < abs(y-y2) else x2
            return float(best) if best is not None else None

        def make_line_mask(shape, line_pts, thickness):
            m = np.zeros(shape[:2], dtype=np.uint8)
            if line_pts is None or len(line_pts) < 2: 
                return m
            pts = np.array(line_pts, np.int32).reshape(-1,1,2)
            cv.polylines(m, [pts], isClosed=False, color=255, thickness=thickness)
            return m
        
        H, W = image.shape[:2]
        min_frames = int(fps * 0.25)
        line_masks = {code: make_line_mask(image.shape, (lane or {}).get('pts'), thickness=10)
                for code, lane in line_dict.items()}
        for idx in vehicle_index:
            x, y, w, h = self.tracked_vehicles[idx]['bbox']
            bottom_y = y + h
            if not (0 <= bottom_y < H): 
                continue

            bx1 = int(x + w*0.30); bx2 = int(x + w*0.75)
            bx1 = max(0, min(W-1, bx1))
            bx2 = max(0, min(W,   bx2))
            if bx2 < bx1: bx1, bx2 = bx2, bx1
            y1 = max(0, bottom_y-2); y2 = min(H, bottom_y+2)
            if bx2 - bx1 < 2 or y2 - y1 < 1: 
                continue
            seg_area = (y2 - y1) * (bx2 - bx1)

            best_code, best_ratio = None, 0.0
            for code, m in line_masks.items():
                nz = int(np.count_nonzero(m[y1:y2, bx1:bx2]))
                ratio = nz / seg_area
                if ratio > best_ratio:
                    best_ratio, best_code = ratio, code

            touching = (best_ratio > 0.04)

            if touching and best_code is not None and line_dict.get(best_code, {}).get('pts') is not None:
                veh = self.tracked_vehicles[idx]
                cx = veh.get('cx') #這裡的cx已經是全圖的座標了
                if cx is None:
                    cx = x + w // 2

                x_line = x_on_polyline_at_y(line_dict[best_code]['pts'], bottom_y)
                if x_line is not None:
                    side = 'left' if cx < x_line else 'right'
                    direction_now = 'right' if side == 'left' else 'left'
                else:
                    direction_now = None
            else:
                direction_now = None
            state = self.tracked_vehicles[idx].get('lane_changing', 'stable')
            self.tracked_vehicles[idx].setdefault('lane_touching_frames', 0)

            if state == 'stable':
                if touching:
                    self.tracked_vehicles[idx]['lane_touching_frames'] += 1
                    if self.tracked_vehicles[idx]['lane_touching_frames'] >= min_frames and direction_now:
                        self.tracked_vehicles[idx]['direction_change'] = direction_now
                        self.tracked_vehicles[idx]['touch_line_code'] = best_code
                        self.tracked_vehicles[idx]['lane_changing'] = 'changing'
                else:
                    self.tracked_vehicles[idx]['lane_touching_frames'] = 0
            elif state == 'changing':
                not_touching_any = True
                best_code = self.tracked_vehicles[idx]['touch_line_code']
                mask = line_masks.get(best_code, None)

                best_code0 = self.tracked_vehicles[idx].get('touch_line_code')
                direction0 = self.tracked_vehicles[idx].get('direction_change')

                if mask is None:
                    not_touching_any = True
                else:
                    if np.count_nonzero(mask[y1:y2, x:x+w]) > 0.08:
                        not_touching_any = False
                        self.check_violation(best_code0, direction0, idx, fps)
                        break

                veh = self.tracked_vehicles[idx]
                cx = veh.get('cx') #這裡的cx已經是全圖的座標了
                if cx is None:
                    cx = x + w // 2

                crossed = False
                if best_code0 and line_dict.get(best_code0, {}).get('pts') is not None:
                    x_line0 = x_on_polyline_at_y(line_dict[best_code0]['pts'], bottom_y)
                    if x_line0 is not None:
                        if direction0 == 'right':
                            crossed = (cx > x_line0)
                        elif direction0 == 'left':
                            crossed = (cx < x_line0)

                if not_touching_any and crossed:
                    self.tracked_vehicles[idx]['lane_changing'] = 'finished'

            elif state == 'finished':
                self.tracked_vehicles[idx]['lane_touching_frames'] = 0
                self.tracked_vehicles[idx]['lane_changing'] = "stable"
                self.tracked_vehicles[idx]['direction_change'] = None
                self.tracked_vehicles[idx]['touch_line_code'] = None
                if idx in self.violation['signal']['vid']:
                    self.violation['signal']['vid'].remove(idx)
                if idx in self.violation['cross']['vid']:
                    self.violation['cross']['vid'].remove(idx)
    
    def extract_lane_region(self, image, line, vehicle_index):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array(line, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(mask, [pts], isClosed=False, color=255, thickness=15)
        for idx in vehicle_index:
            x, y, w, h = self.tracked_vehicles[idx]['bbox']
            cv.rectangle(mask, (x, y), (x + w, y + h), color=0, thickness=-1)
        extracted = cv.bitwise_and(image, image, mask=mask)
        return extracted, mask
    
    def crack_like_reject(self, image_bgr, lane_poly):
        def make_band_masks(shape, poly, center_th=10, gap=6, side_th=12):
            h, w = shape[:2]
            base = np.zeros((h, w), np.uint8)
            pts = np.asarray(poly, np.int32).reshape(-1,1,2)
            cv.polylines(base, [pts], False, 255, center_th)

            k_gap  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (gap, gap))
            k_side = cv.getStructuringElement(cv.MORPH_ELLIPSE, (side_th, side_th))
            ring   = cv.dilate(base, k_gap)
            side   = cv.dilate(base, k_side)
            side   = cv.subtract(side, ring)# 兩側帶 先擴一圈再扣掉中心帶
            return base, side
        
        def polyline_length(poly):
            if poly is None or len(poly) < 2: 
                return 0.0
            p = np.asarray(poly, dtype=np.float32).copy()
            p = p[np.argsort(p[:,1])]
            p_start = p[0]
            p_end = p[-1]
            return float(np.linalg.norm(p_start - p_end))
        
        if lane_poly is None or len(lane_poly) < 2:
            return True

        center_mask, side_mask = make_band_masks(image_bgr.shape, lane_poly, center_th=10, gap=6, side_th=18)

        M_center = max(1, int(np.count_nonzero(center_mask)))
        M_side   = max(1, int(np.count_nonzero(side_mask)))

        hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
        H,S,V = hsv[...,0], hsv[...,1], hsv[...,2]

        black = ((V < 50) & (S < 120)).astype(np.uint8)*255

        white = cv.inRange(hsv, np.array([0, 0, 150]),  np.array([179, 60, 255]))
        yellow= cv.inRange(hsv, np.array([20,60,120]), np.array([40,255,255]))

        black_ratio = np.count_nonzero(cv.bitwise_and(black, black, mask=center_mask)) / M_center
        paint_ratio = (
            np.count_nonzero(cv.bitwise_and(white, white, mask=center_mask)) + np.count_nonzero(cv.bitwise_and(yellow,yellow,mask=center_mask))
        ) / M_center

        Vc = cv.mean(V, mask=center_mask)[0]
        Vs = cv.mean(V, mask=side_mask)[0]
        contrast = Vc - Vs

        area = M_center
        length = polyline_length(lane_poly)
        thinness = area / max(1.0, length)

        # 門檻按影片微調
        if black_ratio > 0.35:          return True
        if paint_ratio < 0.05:          return True
        if contrast < -15:              return True
        if thinness < 4.0:              return True

        return False
    
    def enhance_edges_color(self, frame):
        enhanced = frame.copy()
        enhanced = cv.GaussianBlur(enhanced, (5, 5), 0)
        process = cv.cvtColor(enhanced, cv.COLOR_BGR2GRAY)
        low_threshold = 40
        high_threshold = 120
        edges = cv.Canny(process, low_threshold, high_threshold)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        edges_close = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(edges_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(edges_close, contours, -1, 255, thickness=cv.FILLED)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = edges_close > 0
        hsv[..., 1][mask] = np.clip(hsv[..., 1][mask] * 1.5, 0, 255)
        hsv[...,2][mask] = np.clip(hsv[...,2][mask] * 1.2, 0, 255)
        enhanced_frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return enhanced_frame
    
    def color_threshold_check(self, enhance_frame, lane, vehicle_index):
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([179, 60, 255])

        lower_yellow = np.array([20, 60, 120])
        upper_yellow = np.array([40, 255, 255])

        lower_red1 = np.array([  0,  70,  50], dtype=np.uint8)
        upper_red1 = np.array([ 10, 255, 255], dtype=np.uint8)

        lower_red2 = np.array([170,  70,  50], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

        extracted, lane_mask = self.extract_lane_region(enhance_frame, lane, vehicle_index)
        hsv = cv.cvtColor(extracted, cv.COLOR_BGR2HSV)

        r_mask1 = cv.inRange(hsv, lower_red1, upper_red1)
        r_mask2 = cv.inRange(hsv, lower_red2, upper_red2)

        white_ratio, yellow_ratio, red_ratio = 0, 0, 0

        white_mask = cv.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
        red_mask = cv.bitwise_or(r_mask1, r_mask2)

        valid_pixels = np.sum(np.any(extracted != [0, 0, 0], axis=-1))
        if valid_pixels > 0:
            red_ratio = np.sum(red_mask > 0) / valid_pixels
            yellow_ratio = np.sum(yellow_mask > 0) / valid_pixels
            white_ratio = np.sum(white_mask > 0) / valid_pixels
            if red_ratio < 0.1 and yellow_ratio < 0.01 and white_ratio < 0.01:
                return white_ratio, yellow_ratio, red_ratio, lane_mask, yellow_mask, white_mask, False
            else:
                return white_ratio, yellow_ratio, red_ratio, lane_mask, yellow_mask, white_mask, True
        else:
            return white_ratio, yellow_ratio, red_ratio, lane_mask, yellow_mask, white_mask, False
        
    def lanedete(self,frame, net, img_transforms, cfg, cls_num_per_lane, frame_h, frame_w, vehicle_index, fps):
        def dotted_or_solid(roi_color_lane_mask, roi_lane_mask, min_len):
            if roi_color_lane_mask is None: 
                return 'Unknown', 0.0

            kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3,7))
            kernel_open  = cv.getStructuringElement(cv.MORPH_RECT, (3,5))
            m = cv.morphologyEx(roi_color_lane_mask, cv.MORPH_CLOSE, kernel_close, iterations=1)
            m = cv.morphologyEx(m, cv.MORPH_OPEN,  kernel_open,  iterations=1)

            ys, xs = np.where(roi_lane_mask > 0)
            if len(ys) < min_len:
                return 'Unknown', 0.0

            proj = (m>0).sum(axis = 1).astype(np.float32)
            proj = (proj > 0).astype(np.uint8)

            edges = np.diff(np.pad(proj, (1,1))) != 0
            idx = np.flatnonzero(edges)
            on_runs = (idx[1::2] - idx[::2]) if len(idx) >= 2 else np.array([])
            segments = len(on_runs)
            duty = proj.mean()

            neg_ratio = 1.0 - (np.count_nonzero(m) / max(np.count_nonzero(roi_lane_mask),1))

            if duty > 0.85 and segments <= 2:
                return 'Solid', duty
            if (segments >= 3 and duty < 0.80) or (neg_ratio >= 0.30 and segments >= 2) or (duty < 0.4 and segments <= 2):
                return 'Dotted', duty
            return 'Unknown', duty
        
        def check_y_position(poly, frame_height):
            if poly is None or len(poly) < 2: 
                return 0.0
            p = np.asarray(poly, dtype=np.float32).copy()
            p = p[np.argsort(p[:,1])]
            p_end = p[0]
            if p_end[1] <= frame_height * 0.35:
                return False
            else:
                return True
            
        def polyline_length(poly):
            if poly is None or len(poly) < 2: 
                return 0.0
            p = np.asarray(poly, dtype=np.float32).copy()
            p = p[np.argsort(p[:,1])]
            p_start = p[0]
            p_end = p[-1]
            return float(np.linalg.norm(p_start - p_end))
        
        image_tensor = self.process_lane_image(frame, img_transforms, crop_size= cfg.train_height)
        image_tensor = image_tensor.cuda()
        MIN_ABS_LEN = 120          # 最小絕對長度像素門檻
        REL_DROP_THRESH = 0.6      # 當前長度需≥前一幀的60%
        ABS_DROP_PX = 150          # 與前一幀的絕對長度差超過此值則跳出
        with torch.no_grad():
            pred = net(image_tensor)
        draw_line_frame = np.copy(frame)

        img_rgb = cv.cvtColor(draw_line_frame, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font_size_large = 50
        
        try:
            font_large = ImageFont.truetype(self.font_path, font_size_large)
        except IOError:
            print(f"警告：找不到字型檔 '{self.font_path}'。使用預設字型。")
            font_large = ImageFont.load_default()# !!!!!!!!!!11/14改到這邊

        L2, L1, R1, R2 = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=frame_w, original_image_height=frame_h)
        coords = {"L1":L1, "R1": R1}
        enhance_frame = self.enhance_edges_color(frame)
        for code, lane in coords.items():
            prev_lane = self.lane.get(code, {}).get('pts', None)

            if len(lane) > 0:
                lane = np.array(lane[0], dtype = np.int32)

                if len(lane) > 7:
                    lane = self.lane_filter(lane)
                    lane = np.array(lane, dtype = np.int32)

                    if len(lane) > 7:
                        X = lane[:, 0].reshape(-1, 1)
                        y = lane[:, 1]
                        model = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor())
                        model.fit(X, y)

                        x_fit = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
                        y_fit = model.predict(x_fit)

                        fitted_curve = np.hstack((x_fit, y_fit.reshape(-1, 1)))
                        lane = np.array(fitted_curve, dtype = np.int32)
                        lane = self.lane_filter(lane)
                        lane = np.array(lane, dtype = np.int32)
                        cur_lane = lane
                        
                        cur_cmp, prev_cmp = cur_lane, prev_lane

                        cur_len  = polyline_length(cur_cmp)
                        prev_len = polyline_length(prev_cmp) if prev_lane is not None else 0.0
                        reject = self.crack_like_reject(frame, lane)
                        y_axis_ok = check_y_position(cur_cmp, frame_h)
                        use_prev = False
                        if cur_len < MIN_ABS_LEN:
                            use_prev = prev_lane is not None
                        elif prev_lane is not None:
                            rel_ok  = (cur_len >= prev_len * REL_DROP_THRESH)
                            abs_ok  = ((prev_len - cur_len) <= ABS_DROP_PX)
                            y_ok = check_y_position(cur_cmp, frame_h)
                            if not (rel_ok and abs_ok and y_ok):
                                use_prev = True
                        _, _, _, _, _, _, color_check = self.color_threshold_check(enhance_frame, lane, vehicle_index)
                        if use_prev:
                            lane = prev_lane
                        elif prev_lane is None and (color_check == False or reject == True or y_axis_ok == False):
                            continue
                        else:
                            self.lane[code]['pts'] = lane
                    else:
                        if prev_lane is not None:
                            lane = prev_lane
                        else:
                            continue
                else:
                    if prev_lane is not None:
                        lane = prev_lane
                    else:
                        continue
            else:
                if prev_lane is not None:
                    lane = prev_lane
                else:
                    continue
            reject = self.crack_like_reject(frame, lane)
            white_ratio, yellow_ratio, red_ratio , lane_mask, yellow_mask, white_mask, color_check = self.color_threshold_check(enhance_frame, lane, vehicle_index)

            type_dict = {"Solid": "實線", "Dotted": "虛線"}
            
            if color_check and reject == False:
                print(f"{code}: yellow:{yellow_ratio}    white:{white_ratio} red:{red_ratio}")
                p = np.asarray(lane, dtype=np.float32).copy()
                p = p[np.argsort(p[:,1])]
                x, y = map(int, p[-1])
                lane_type = None
                duty = 0
                lane_points = [tuple(pt) for pt in lane.reshape(-1, 2)]

                if yellow_ratio > 0.09 or (yellow_ratio > white_ratio and yellow_ratio > red_ratio and yellow_ratio > 0.06):
                    lane_type, duty = dotted_or_solid(yellow_mask, lane_mask, MIN_ABS_LEN)
                    text_to_draw = ""
                    if lane_type == "Unknown":
                        kernel_horizon = cv.getStructuringElement(cv.MORPH_RECT, (4, 12))
                        kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (12, 4))
                        yellow_dilated_horizontal = cv.dilate(yellow_mask, kernel_horizon, iterations=1)
                        yellow_dilated_both = cv.dilate(yellow_dilated_horizontal, kernel_vertical, iterations=1)
                        contours, _ = cv.findContours(yellow_dilated_both, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        if len(contours) >= 3:
                            text_to_draw = "黃色虛線"
                            self.lane[code]['type'] = "Dotted_Yellow_Line"
                        else: 
                            text_to_draw = "黃色實線"
                            self.lane[code]['type'] = "Solid_Yellow_Line"
                    else:
                        translated_type = type_dict.get(lane_type, lane_type)
                        text_to_draw = f"{translated_type}黃線"
                        self.lane[code]['type'] = f"{lane_type}_Yellow_Line"
                    
                    color_rgb = (255, 255, 0)
                    draw.text((x + 50, y - 300), text_to_draw, font=font_large, fill=color_rgb, stroke_width=2, stroke_fill=(0, 0, 0))
                    draw.line(lane_points, fill=color_rgb, width=5)

                elif white_ratio > 0.09 or (white_ratio > yellow_ratio and white_ratio > red_ratio and white_ratio > 0.06):
                    lane_type, duty = dotted_or_solid(yellow_mask, lane_mask, MIN_ABS_LEN)
                    text_to_draw = ""
                    if lane_type == "Unknown":
                        kernel_horizon = cv.getStructuringElement(cv.MORPH_RECT, (4, 12))
                        kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (12, 4))
                        white_dilated_horizontal = cv.dilate(white_mask, kernel_horizon, iterations=1)
                        white_dilated_both = cv.dilate(white_dilated_horizontal, kernel_vertical, iterations=1)
                        contours, _ = cv.findContours(white_dilated_both, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        if len(contours) >= 3:
                            text_to_draw = "白色虛線"
                            self.lane[code]['type'] = "Dotted_White_Line"
                        else:
                            text_to_draw = "白色實線"
                            self.lane[code]['type'] = "Solid_White_Line"
                    else: 
                        translated_type = type_dict.get(lane_type, lane_type)
                        text_to_draw = f"白色{translated_type}"
                        self.lane[code]['type'] = f"{lane_type}_White_Line"
                    
                    color_rgb = (255, 255, 255)
                    draw.text((x + 50, y - 300), text_to_draw, font=font_large, fill=color_rgb, stroke_width=2, stroke_fill=(0, 0, 0))
                    draw.line(lane_points, fill=color_rgb, width=5)

                elif red_ratio > 0.1 :
                    text_to_draw = "紅色線"
                    self.lane[code]['type'] = "Red_Line"
                    
                    color_rgb = (255, 0, 0)
                    draw.text((x + 50, y - 300), text_to_draw, font=font_large, fill=color_rgb, stroke_width=2, stroke_fill=(0, 0, 0))
                    draw.line(lane_points, fill=color_rgb, width=5)
                
                else:
                    x, y = map(int, lane[0])
                    kernel_horizon = cv.getStructuringElement(cv.MORPH_RECT, (4, 12))
                    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (12, 4))
                    white_dilated_horizontal = cv.dilate(white_mask, kernel_horizon, iterations=1)
                    white_dilated_both = cv.dilate(white_dilated_horizontal, kernel_vertical, iterations=1)
                    contours, _ = cv.findContours(white_dilated_both, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    
                    text_to_draw = ""
                    if len(contours) >= 3:
                        text_to_draw = "白色虛線"
                        self.lane[code]['type'] = "Dotted_White_Line"
                    else:
                        text_to_draw = "白色實線"
                        self.lane[code]['type'] = "Solid_White_Line"
                    
                    color_rgb = (255, 255, 255)
                    draw.text((x + 50, y - 300), text_to_draw, font=font_large, fill=color_rgb, stroke_width=2, stroke_fill=(0, 0, 0))
                    draw.line(lane_points, fill=color_rgb, width=5)
            
            else: # 這是最外層 `if color_check and reject == False:` 的 else
                p = np.asarray(self.lane[code]['pts'], dtype=np.float32).copy()
                p = p[np.argsort(p[:,1])]
                x, y = map(int, p[-1])
                
                lane_points = [tuple(pt) for pt in lane.reshape(-1, 2)]
                text_to_draw = ""
                color_rgb = (255, 255, 255)

                if self.lane[code]['type'] == "Dotted_White_Line" or self.lane[code]['type'] == "Solid_White_Line":
                    text_to_draw = "白色虛線" if self.lane[code]['type'] == "Dotted_White_Line" else "白色實線"
                    color_rgb = (255, 255, 255)
                elif self.lane[code]['type'] == "Dotted_Yellow_Line" or self.lane[code]['type'] == "Solid_Yellow_Line":
                    text_to_draw = "黃色虛線" if self.lane[code]['type'] == "Dotted_Yellow_Line" else "黃色實線"
                    color_rgb = (255, 255, 0)
                elif self.lane[code]['type'] == "Red_Line":
                    text_to_draw = "紅色線"
                    color_rgb = (255, 0, 0)
                
                if text_to_draw:
                    draw.text((x + 50, y - 300), text_to_draw, font=font_large, fill=color_rgb, stroke_width=2, stroke_fill=(0, 0, 0))
                    draw.line(lane_points, fill=color_rgb, width=5)

        self.check_lane_change(enhance_frame, self.lane, fps, vehicle_index, draw_line_frame)
        
        for idx in vehicle_index:
            bbox = self.tracked_vehicles[idx]['bbox']
            x, y, w, h = bbox
            if idx == 1:#記得改回來
                if self.tracked_vehicles[idx]['lane_changing'] == 'stable':
                    final_text = "車道不變"
                    
                elif self.tracked_vehicles[idx]['lane_changing'] == 'finished':
                    final_text = "變換結束"

                else:
                    final_text = "變換車道中"

                    if self.tracked_vehicles[idx]['direction_change'] is not None:
                        if self.tracked_vehicles[idx]['direction_change'] == 'left':
                            direction_text = "往左"
                        else:
                            direction_text = "往右"
                    final_text = f"{direction_text}變換車道中"
                draw.text((x, y - 70), final_text, font=font_large, fill = (255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

        draw_line_frame = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
        return draw_line_frame

    
    def extract_vehicle_regions(self, cur_time, frame, results):
        vehicle_regions = {}
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                vehicle_regions[track_id] = {
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'image': frame[y1:y2, x1:x2]
                }

                self.tracked_vehicles[track_id]['id'] = track_id
                self.tracked_vehicles[track_id]['bbox'] = (x1, y1, x2 - x1, y2 - y1)
        
        return vehicle_regions

    def moving_average(self, data, window):
        if len(data) < 2:
            return np.array(data, dtype=float)
        w = int(window)
        if len(data) < w:
            out = np.array(data, dtype=float)
            for i in range(1, len(out)):
                out[i] = 0.5*out[i] + 0.5*out[i-1]
            return out
        kernel = np.ones(w, dtype=float) / w
        return np.convolve(np.array(data), kernel, mode='same')

    def detect_slope_pattern(self, gray_history, fps, bin_size, min_gap_bins, slope_threshold, stable_len_frames):
        def is_high(s1, s2, min_gap_bins):
            gap = abs(s1 - s2)
            bin_jump_ok = abs(int(s1 // bin_size) - int(s2 // bin_size)) >= min_gap_bins
            amp_ok = gap >= max(0.6 * bin_size, 1.2 * mad)
            return bin_jump_ok and amp_ok
        if len(gray_history) < 6:
            return False, 'UNKNOWN', {"N": len(gray_history), "stable_segments": [], "transitions_found": 0, "reason": "not_enough_samples", "bin_size": bin_size}
        window = max(3, fps // 20)
        if window % 2 == 0:
            window += 1
        smoothed_gray = np.asarray(gray_history, dtype=float)
        #smoothed_gray = np.asarray(self.moving_average(gray_history, window), dtype=float)
        median = float(np.median(smoothed_gray))
        mad = float(np.median(np.abs(smoothed_gray - median))) + 1e-6
        dyn_bin = max(8.0, min(16.0, 2.5 * mad))
        if bin_size is None:
            bin_size = dyn_bin
        dyn_slope = max(1.5, 1.2 * mad)
        slope_thr = max(slope_threshold, dyn_slope)
        vals = np.clip(smoothed_gray, 0, 255)
        bins = (vals // bin_size).astype(int)
        diff = np.diff(smoothed_gray)

        segs = []
        start = 0
        cur = bins[0]
        for i in range(1, len(bins)):
            if bins[i] != cur:
                segs.append({"start": start, "end": i-1, "bin": int(cur), "length": i-start})
                start = i
                cur = bins[i]
        segs.append({"start": start, "end": len(bins)-1, "bin": int(cur), "length": len(bins)-start})

        if stable_len_frames is None:
            stable_len_frames = max(8, int(0.23 * fps))
        
        stable = []
        for s in segs:
            if s["length"] >= stable_len_frames:
                st = s["start"]
                ed = s["end"]
                mean_gray = float(np.mean(smoothed_gray[st:ed+1]))
                s2 = dict(s)
                s2["mean_gray"] = mean_gray
                stable.append(s2)

        if len(stable) < 2:
            return False, 'UNKNOWN', {"N": len(gray_history), "stable_segments": [], "transitions_found": 0,  "reason": "not_enough_stable_segments", "bin_size": bin_size}
        
        cand = stable[-5:]
        transitions = [] 
        is_flashing = False
        last_phase = 'UNKNOWN'

        for i in range(len(cand) - 1):
            first_seg, sec_seg = cand[i], cand[i+1]
            if first_seg["bin"] == sec_seg["bin"]:
                continue
            if is_high(sec_seg["mean_gray"], first_seg["mean_gray"], min_gap_bins) or is_high(first_seg["mean_gray"], sec_seg["mean_gray"], min_gap_bins):
                k = max(1, sec_seg["start"] - 1)
                local = diff[max(1, k-2): min(len(diff)-1, k+2)]
                if local.size and np.max(np.abs(local)) >= slope_thr:
                    transitions.append((i, first_seg, sec_seg))

        if len(transitions) == 1 and len(cand) >= 3: #應該依照len(trasition)的次數來取出cand的人比較is_high在判斷is_flashing 如果每次都只取最後三個cand來看會不準確
            # m0, m1, m2 = cand[-3]["mean_gray"], cand[-2]["mean_gray"], cand[-1]["mean_gray"]
            # if (is_high(m1, m0, min_gap_bins) and is_high(m1, m2, min_gap_bins)) or \
            #     (is_high(m0, m1, min_gap_bins) and is_high(m2, m1, min_gap_bins)):
            is_flashing = True
        elif len(transitions) >= 2 and len(cand) >=4:
            is_flashing = True
        
        if len(stable) >= 2:
            ref = stable[-2]["mean_gray"]
        else:
            ref = median
        last_mean = stable[-1]["mean_gray"]
        last_phase = 'HIGH' if (last_mean - ref) >= (min_gap_bins * bin_size / 2.0) else 'LOW'

        details = {
            "N": len(gray_history),
            "stable_segments": [(s["start"], s["end"], s["bin"], round(s["mean_gray"],1)) for s in stable],
            "transitions_found": len(transitions),
            "reason": "No exception",
            "bin_size": bin_size
        }
        return is_flashing, last_phase, details

    def detect_periodic_pattern(self, gray_history, fps):
        if len(gray_history) < fps // 2:
            return False, 0, 0, 0, 0

        diffs = []
        smoothed_gray = self.moving_average(gray_history, 5)
        diffs = np.diff(smoothed_gray)
        if len(diffs) < fps // 4:
            return False, 0, 0, 0, 0

        peaks = []
        valleys = []
        diff_threshold = 3
        
        for i in range(1, len(diffs) - 1):
            if diffs[i] > diffs[i-1] and diffs[i] > diffs[i+1] and diffs[i] > diff_threshold:
                peaks.append(i)
            elif diffs[i] < diffs[i-1] and diffs[i] < diffs[i+1] and diffs[i] < -diff_threshold:
                valleys.append(i)
        
        total_extremes = len(peaks) + len(valleys)

        available_time = len(gray_history) / fps
        expected_cycles = available_time * 1.8 
        expected_extremes = expected_cycles * 2

        min_extremes = max(2, int(expected_extremes * 0.4))
        max_extremes = int(expected_extremes * 2.5)

        gray_range = max(gray_history) - min(gray_history)
        amplitude_threshold = 12

        frequency_check = min_extremes <= total_extremes <= max_extremes
        amplitude_check = gray_range > amplitude_threshold
        return frequency_check and amplitude_check, gray_range, total_extremes, max_extremes, min_extremes

    def analyze_signal_region(self, region, side, vehicle_id, cur_time, fps):
        signal_region = cv.resize(region, (90, 90))
        gray = cv.cvtColor(signal_region, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(signal_region, cv.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 210])
        upper_white = np.array([179, 45, 255])
        lower_yellow = np.array([15, 60, 100])
        upper_yellow = np.array([40, 255, 255])
        white_mask = cv.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
        combined_mask = cv.bitwise_or(white_mask, yellow_mask)
        gray_mean = 0

        gray_mean = np.mean(gray)
        brightness_threshold = 160
        color_threshold = 0.005
        brightness = 0
        combined_ratio = cv.countNonZero(combined_mask) / (combined_mask.shape[0] * combined_mask.shape[1])
        hsv[...,2] = np.clip(hsv[...,2] * 1.2, 0, 255)
        brightness = float(cv.mean(hsv[:, :, 2], mask = combined_mask)[0])

        brightness_raw = brightness
        signal_history = self.tracked_vehicles[vehicle_id]['signal_history']

        brightness_used = brightness_raw

        if self.tracked_vehicles[vehicle_id]['signal_brightness'][side] != 0:
            prev_brightness = self.tracked_vehicles[vehicle_id]['signal_brightness'][side]
        else:
            prev_brightness = brightness_used

        b_details, details = {
            "N": 0,
            "stable_segments": [],
            "transitions_found": 0,
            "reason": "None",
            "bin_size": 0
        },{
            "N": 0,
            "stable_segments": [],
            "transitions_found": 0,
            "reason": "None",
            "bin_size": 0
        }

        gray_is_periodic = False
        g_is_periodic_changing = False
        b_is_periodic_changing = False
        gray_amplitude = 0
        total_extremes = 0
        max_extremes = 0
        min_extremes = 0

        if len(signal_history) > fps // 2:
            gray_history = [entry[4] for entry in signal_history if entry[0] == side]
            gray_is_periodic, gray_amplitude, total_extremes, max_extremes, min_extremes = self.detect_periodic_pattern(gray_history, fps)
            g_is_periodic_changing,  last_phase, details = self.detect_slope_pattern(gray_history, fps, bin_size = None, min_gap_bins = 2, slope_threshold = 3, stable_len_frames=None)

            brightness_history = [entry[3] for entry in signal_history if entry[0] == side]
            b_is_periodic_changing,  b_last_phase, b_details = self.detect_slope_pattern(brightness_history, fps, bin_size = None,  min_gap_bins = 2, slope_threshold = 2,stable_len_frames = None)

        is_bright = brightness_used > brightness_threshold and combined_ratio > color_threshold
        cur_state = 0
        brightness_diff = 0
        if prev_brightness != 0:
            brightness_diff = brightness_used - prev_brightness

        brightness_condition = (brightness_diff > 20 and is_bright) 

        periodic_condition = brightness_used > 210 and gray_is_periodic 
        
        if brightness_condition or periodic_condition:
            cur_state = 1
        
        self.tracked_vehicles[vehicle_id]['signal_region'][side] = signal_region
        self.tracked_vehicles[vehicle_id]['signal_brightness'][side] = brightness_used
        self.tracked_vehicles[vehicle_id]['signal_history'].append((side, cur_state, cur_time, brightness_used, gray_mean))
        print(f"[{vehicle_id} - {side}] gray_amplitude:{gray_amplitude:.1f} total_extremes: {total_extremes:.1f}, max_extremes: {max_extremes:.1f}, min_extremes: {min_extremes:.1f}")
        print(f"[{vehicle_id} - {side}] is_change:{g_is_periodic_changing} stable_seg:{details['stable_segments']} transitions_found:{details['transitions_found']:.1f} reason:{details['reason']} bin_size:{details['bin_size']}" )
        print(f"[{vehicle_id} - {side}] is_change:{b_is_periodic_changing} stable_seg:{b_details['stable_segments']} transitions_found:{b_details['transitions_found']:.1f} reason:{b_details['reason']} bin_size:{b_details['bin_size']}" )
        if g_is_periodic_changing or b_is_periodic_changing:
            return "2OFF"#"FLASHING"
        # elif periodic_condition:
        #      return "FLASHING"
        else:
            return "2OFF"
        
    def recognize_plate(self, plate_image):
        cropped_img_np = np.array(plate_image)
        cropped_img_np = sr_dnn.upsample(cropped_img_np)
        result = ocr.ocr(cropped_img_np, cls=False)
        if result and result[0]:
            data = result[0][0][1]
            text, confidence = data
            print(f"Detected text: '{text}', Confidence: {confidence:.2f}")
            return text
        
    def process_frame(self, cur_time, frame, frame_width, frame_height, fps):
        def iou(box1, idx1, box2, idx2):
            xy_max = np.minimum(box1[2:], box2[2:])
            xy_min = np.maximum(box1[:2], box2[:2])

            intersection = np.clip(xy_max - xy_min, a_min = 0, a_max = np.inf)
            inter_area = intersection[0] * intersection [1]

            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area

            ratio = inter_area / union_area
            if ratio > 0.1:
                return idx1 if box1_area < box2_area else idx2
            
        def _crop_with_pad(img, bbox_xywh, pad_ratio):
            x, y, w, h = bbox_xywh
            H, W = img.shape[:2]
            px, py = int(w * pad_ratio), int(h * pad_ratio)
            x1 = max(0, x - px); y1 = max(0, y - py)
            x2 = min(W, x + w + px); y2 = min(H, y + h + py)
            if x2 <= x1 or y2 <= y1:
                return None, None
            return img[y1:y2, x1:x2], (x1, y1), (x1, y1, x2-x1, y2-y1)
        
        def choose_best_light_pair(cand_lights, best_pair, car_x, car_w, car_center):
            best_score = float('-inf')
            cx_default = car_x + car_w / 2.0
            expected = 0.6 * max(car_w, 1e-6)
            if car_center is not None:
                car_cx = car_center
            else:
                car_cx = cx_default

            for p in combinations(cand_lights, 2):
                (idx1, (x1, y1)), (idx2, (x2, y2)) = p
                if x1 > x2:
                    x1, y1, idx1, x2, y2, idx2 = x2, y2, idx2, x1, y1, idx1

                horiz_dist = abs(x2 - x1)
                vert_diff = abs(y2 - y1)
                mid_x = (x1 + x2) / 2

                dist_score = -abs(horiz_dist - expected)
                vert_score = -vert_diff
                center_score = -abs(mid_x - car_cx)

                s = dist_score + 0.5*vert_score + 0.3*center_score
                if s > best_score:
                    best_score = s
                    best_pair = (idx1, idx2)
            return best_pair

        r1 = self.det_track.track(
            frame, imgsz=960, conf=0.45, classes=self.veh_ids, persist=True
        )[0]

        crops, offsets, vids, vehicle_regions = [], [], [], {}
        for box in r1.boxes:
            if box.id is None:
                continue
            vid = int(box.id.item()) if hasattr(box.id, "item") else int(box.id)
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            x, y, w, h = x1, y1, x2-x1, y2-y1
            crop, (ox,oy), (x, y, w, h) = _crop_with_pad(frame, (x,y,w,h), -0.05)
            crops.append(crop); offsets.append((ox,oy)); vids.append(vid)
            vehicle_regions[vid] = {"bbox": (x,y,w,h), "image": crop}
            self.tracked_vehicles[vid]['id'] = vid
            self.tracked_vehicles[vid]['bbox'] = (x, y, w, h)

        if crops:
            B = min(16, len(crops))
            plate_res_list = self.det_part(
                crops, imgsz=768, conf=0.25, classes = self.plate_ids,
                batch=B, verbose=False
            )

            light_res_list = self.light_model(
                crops, conf=0.45, iou = 0.45, max_det = 3
            )
            light_data = {}
            for vid, plate_part, light_part, (ox, oy) in zip(vids, plate_res_list, light_res_list, offsets): #ox oy才是裁切之後的左上座標 
                best_plate = {}
                light_data = {}
                light_number = 0
                car_x, car_y, car_w, car_h = vehicle_regions[vid]['bbox']
                left_state = self.tracked_vehicles[vid]['left_signal']
                right_state = self.tracked_vehicles[vid]['right_signal']
                for pb in getattr(plate_part, "boxes", []):
                    x1, y1, x2, y2 = pb.xyxy[0].int().tolist()
                    gx1, gy1, gx2, gy2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue
                    conf = float(pb.conf)
                    if not best_plate or conf > best_plate["conf"]:
                        plate_in_car_cx = int((x1 + x2) / 2)
                        plate_in_car_cy = int((y1 + y2) / 2)
                        plate_x_ratio = float(plate_in_car_cx / car_w)
                        plate_y_ratio = float(plate_in_car_cy / car_h)
                        best_plate = {"gbbox": (gx1, gy1, gx2, gy2), "conf": conf,
                                       "plate_x_ratio": plate_x_ratio, "plate_y_ratio": plate_y_ratio}
                
                for lb in getattr(light_part, "boxes", []):
                    x1, y1, x2, y2 = lb.xyxy[0].int().tolist()
                    gx1, gy1, gx2, gy2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue

                    light_cy = (gy1 + gy2) / 2
                    light_cx = (gx1 + gx2) / 2
                    if (car_y + car_h*0.2) < light_cy < (car_y + car_h*0.8):
                        light_data[light_number] = {
                            'light_bbox' : (gx1, gy1, gx2, gy2),
                            'light_cx': light_cx,
                            'light_cy': light_cy,
                            'image': frame[gy1:gy2, gx1:gx2]
                        }
                        light_number += 1

#---------------車牌
                plate_text = None
                car_cx = None
                car_cy = None
                if best_plate:
                    gx1, gy1, gx2, gy2 = best_plate['gbbox']
                    plate_img = frame[gy1:gy2, gx1:gx2]
                    if self.tracked_vehicles[vid]['license_plate'] is None and vid == 1:#記得改回來
                        plate_text = self.recognize_plate(plate_img)
                    
                    car_cx = int((gx1 + gx2) / 2)
                    car_cy = int((gy1 + gy2) / 2)

                    plate_x_ratio = best_plate['plate_x_ratio']
                    plate_y_ratio = best_plate['plate_y_ratio']
                    self.tracked_vehicles[vid]['license_coord_ratio'] = (plate_x_ratio, plate_y_ratio)
                elif self.tracked_vehicles[vid]['license_coord_ratio'] is not None:
                    (plate_x_ratio, plate_y_ratio)  = self.tracked_vehicles[vid]['license_coord_ratio']
                    car_cx = int(ox + car_w * plate_x_ratio)
                    car_cy = int(oy + car_h * plate_y_ratio)
                if plate_text:
                    pattern = r'^\d{4}[A-Za-z]\d$'
                    if re.match(pattern, plate_text):
                        self.tracked_vehicles[vid]['license_plate'] = plate_text
                    else:
                        print("不符合格式")

#---------------方向燈過濾
                all_lights = []
                all_lights = [ (idx, (v['light_bbox'])) for idx, v in light_data.items() ]  
                to_drop = set()
                for (idx1, b1), (idx2, b2) in combinations(all_lights, 2):
                    if idx1 in to_drop or idx2 in to_drop:
                        continue
                    kick = iou(b1, idx1, b2, idx2)
                    if kick is not None:
                        to_drop.add(kick)
                if car_cy is not None:
                    for idx, light_info in light_data.items():
                        if idx in to_drop:
                            continue
                        light_cy = light_info['light_cy']
                        if light_cy > car_cy:
                            to_drop.add(idx)

                for k in to_drop:
                    light_data.pop(k, None)
                    light_number -= 1
#---------------方向燈

                if len(light_data) == 0:
                    left_light_missing_region = self.tracked_vehicles[vid]['signal_region']['left']
                    if left_light_missing_region is not None:
                        left_state = self.analyze_signal_region(left_light_missing_region, 'left', vid, cur_time, fps)
                        self.tracked_vehicles[vid]['left_signal'] = left_state
                    
                    right_light_missing_region = self.tracked_vehicles[vid]['signal_region']['right']
                    if right_light_missing_region is not None:
                        right_state = self.analyze_signal_region(right_light_missing_region, 'right', vid, cur_time, fps)
                        self.tracked_vehicles[vid]['right_signal'] = right_state
                    continue
                elif len(light_data) == 1:
                    frame_center = frame_width / 2
                    car_loc_center = car_x + car_w / 2
                    car_area = car_w * car_h
                    frame_area = frame_width * frame_height
                    car_ratio = car_area / frame_area
                    missing = None
                    if car_cx is not None and car_cy is not None:
                        for _, light_info in light_data.items():
                            x1, y1, x2, y2 = light_info['light_bbox']
                            light_cx = (x1 + x2) * 0.5
                            side = 'left' if light_cx < car_cx else 'right'
                            missing = 'right' if side == 'left' else 'left'
                            light_missing_region = self.tracked_vehicles[vid]['signal_region'][missing]                      
                            light_region = light_info['image']

                            state = self.analyze_signal_region(light_region, side, vid, cur_time, fps)
                            self.tracked_vehicles[vid][f'{side}_signal'] = state

                            if light_missing_region is not None:
                                state = self.analyze_signal_region(light_missing_region, missing, vid, cur_time, fps)
                                self.tracked_vehicles[vid][f'{missing}_signal'] = state

                            self.tracked_vehicles[vid]['cx'] = int(np.clip(
                                car_cx,
                                max(0, car_x),
                                min(frame_width - 1, car_x + car_w - 1)
                            ))

                    elif car_ratio > 0.25: #如果車子的比例太大又只有一個車燈，可以判斷成離你很近又只能看到單邊的車燈
                        for _, light_info in light_data.items():
                            x1, y1, x2, y2 = light_info['light_bbox']
                            side = 'left' if car_loc_center  > frame_center else 'right'
                            missing = 'right' if side == 'left' else 'left'    
                            light_missing_region = self.tracked_vehicles[vid]['signal_region'][missing]                           
                            light_region = light_info['image']

                            state = self.analyze_signal_region(light_region, side, vid, cur_time, fps)
                            self.tracked_vehicles[vid][f'{side}_signal'] = state

                            if light_missing_region is not None:
                                state = self.analyze_signal_region(light_missing_region, missing, vid, cur_time, fps)
                                self.tracked_vehicles[vid][f'{missing}_signal'] = state

                            if side == "left":
                                self.tracked_vehicles[vid]['cx'] = int(np.clip(
                                    int((x1 + x2) // 2 + (car_w // 5)),
                                    max(0, car_x),
                                    min(frame_width - 1, car_x + car_w - 1)
                                ))
                            else:
                                self.tracked_vehicles[vid]['cx'] = int(np.clip(
                                    int((x1 + x2) // 2 - (car_w // 5)),
                                    max(0, car_x),
                                    min(frame_width - 1, car_x + car_w - 1)
                                ))
                    else:#如果比例不大，代表只是模型偵測不到另一個車燈，且還沒有車牌可以判斷。
                        pass
                elif len(light_data) == 2:
                    mid_light_x = sum([data['light_cx'] for data in light_data.values()]) / 2
                    for _, light_info in light_data.items():
                        x1, y1, x2, y2 = light_info['light_bbox']
                        light_center_x = light_info['light_cx']
                        side = 'left' if mid_light_x >= light_center_x else 'right'
                        light_region = light_info['image']

                        state = self.analyze_signal_region(light_region, side, vid, cur_time, fps)
                        self.tracked_vehicles[vid][f'{side}_signal'] = state

                    if car_cx is not None and car_cy is not None:
                        self.tracked_vehicles[vid]['cx'] = int(np.clip(
                            car_cx,
                            max(0, car_x),
                            min(frame_width - 1, car_x + car_w - 1)
                        ))
                    else:
                        self.tracked_vehicles[vid]['cx'] = int(mid_light_x)
                else:
                    cand_lights = []
                    cand_lights = [ (idx, (v['light_cx'], v['light_cy'])) for idx, v in light_data.items() ]
                    best_pair = None
                    if car_cx is not None:
                        best_pair = choose_best_light_pair(cand_lights,best_pair, car_x, car_w, car_cx)
                    else:
                        best_pair = choose_best_light_pair(cand_lights,best_pair, car_x, car_w, None)

                    if best_pair is not None:
                        left_light_index, right_light_index = best_pair
                        x1, y1, x2, y2 = light_data[left_light_index]['light_bbox']
                        left_light_region = light_data[left_light_index]['image']
                        left_state = self.analyze_signal_region(left_light_region, 'left', vid, cur_time, fps)
                        self.tracked_vehicles[vid]['left_signal'] = left_state
                        left_cx = (x1 + x2) // 2

                        x1, y1, x2, y2 = light_data[right_light_index]['light_bbox']
                        right_light_region = light_data[right_light_index]['image']
                        right_state = self.analyze_signal_region(right_light_region, 'right', vid, cur_time, fps)
                        self.tracked_vehicles[vid]['right_signal'] = right_state
                        right_cx = (x1 + x2) // 2

                        if car_cx is not None and car_cy is not None:
                            self.tracked_vehicles[vid]['cx'] = int(np.clip(
                                car_cx,
                                max(0, car_x),
                                min(frame_width - 1, car_x + car_w - 1)
                            ))
                        else:
                            self.tracked_vehicles[vid]['cx'] = min(0, max((left_cx + right_cx) // 2, frame_width - 1))

        return vids
        
    def draw_info(self, frame, vehicle_id, violation):
        bbox = self.tracked_vehicles[vehicle_id]['bbox']
        x, y, w, h = bbox

        color_map = {"OFF": (0, 0, 255), "2OFF": (0, 0, 255),"FLASHING": (0, 255, 0), "ON": (255, 255, 0)}    
        chinese_map = {"OFF": "關", "2OFF": "關", "ON": "開", "FLASHING": "閃爍"}
        font_size_large = 50

        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font_large = ImageFont.truetype(self.font_path, font_size_large)
        except IOError:
            print(f"警告：找不到字型檔 '{self.font_path}'。使用預設字型。")
            font_large = ImageFont.load_default()

        if vehicle_id == 1:
            box_color_bgr = (0, 0, 255) if violation else (255, 0, 0)
            box_color_rgb = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])
            draw.rectangle([(x, y), (x + w, y + h)], outline=box_color_rgb, width=2)

            if self.tracked_vehicles[vehicle_id]['cx'] is not None:
                cx = self.tracked_vehicles[vehicle_id]['cx']
                cy = y + h
                radius = 4
                draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=(0, 255, 0))

            if self.tracked_vehicles[vehicle_id]['license_plate']:
                plate_text = self.tracked_vehicles[vehicle_id]['license_plate']
                draw.text((x, y - 120), f"車牌: {plate_text}", font=font_large, fill = (255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

            left_status_en = self.tracked_vehicles[vehicle_id]['left_signal']
            left_status_ch = chinese_map.get(left_status_en, left_status_en)
            left_color_bgr = color_map.get(left_status_en, (255, 255, 255))
            left_color_rgb = (left_color_bgr[2], left_color_bgr[1], left_color_bgr[0])
            draw.text((x - 190, y + 15), f"左燈: {left_status_ch}", font=font_large, fill=left_color_rgb, stroke_width=2, stroke_fill=(0, 0, 0))
            
            right_status_en = self.tracked_vehicles[vehicle_id]['right_signal']
            right_status_ch = chinese_map.get(right_status_en, right_status_en)
            right_color_bgr = color_map.get(right_status_en, (255, 255, 255))
            right_color_rgb = (right_color_bgr[2], right_color_bgr[1], right_color_bgr[0])
            draw.text((x + w + 10, y + 15), f"右燈: {right_status_ch}", font=font_large, fill=right_color_rgb, stroke_width=2, stroke_fill=(0, 0, 0))
            frame = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

        return frame
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    net, img_transforms, cls_num_per_lane, cfg = lanemodel() #要執行前記得先確定utils.common裡面的args

    check_dirs()
    while True:
        try:
            candidate_video = scan_new_video()
            for cvideo in candidate_video:

                has_violation, final_dir = process_video(cvideo)
                mark_done(cvideo)
                print(f"[INFO] Finished processing. Output at: {final_dir}")

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl + c :Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] main loop exception: {e}")
            break
        time.sleep(POLL_INTERVAL_SEC)
