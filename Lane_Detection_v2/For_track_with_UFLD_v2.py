import torch, os, sys
import subprocess
import cv2 as cv
from cv2 import dnn_superres
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict, deque
import scipy.special, tqdm
from PIL import Image
from utils.dist_utils import dist_print
from utils.common import merge_config, get_model
from utils.lane import pred2coords
from data.dataset import LaneTestDataset
import torchvision.transforms as transforms
from sklearn.linear_model import RANSACRegressor

# car_model = YOLO(r"car_best.pt")
# light_model = YOLO(r"light_best.pt")
# licence_model = YOLO(r"licencePlate_best.pt")
# licence_model = YOLO(r"licence_best.pt")
sr_dnn = dnn_superres.DnnSuperResImpl_create()
sr_dnn.readModel("weights/EDSR_x2.pb")
sr_dnn.setModel("edsr", 2)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
class YOLOv8:
    def __init__(self, buffer_size=50):
        self.car_model = YOLO(r"weights/car_best.pt")
        self.light_model = YOLO(r"weights/light_best.pt")
        self.licence_model = YOLO(r"weights/licencePlate_best.pt")
        self.ransac_model = RANSACRegressor()
        self.tracked_vehicles = defaultdict(lambda: {
            'id': None,
            'bbox': None,
            'license_plate': None,
            'left_signal': 'OFF',
            'right_signal': 'OFF',
            'signal_history': deque(maxlen=buffer_size),
            'last_seen': 0,
        })
    #UFLD_V2 - lanemodel
    def lanemodel(self):
        args, cfg = merge_config()
        cfg.batch_size = 1
        print('Setting batch_size to 1 for demo generation')

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
    
    def process_lane_image(self, img_path, img_transform, crop_size):

        img = img_path
        blurred = cv.GaussianBlur(img, (5, 5), 0)

        lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv.merge((l_eq, a, b))
        enhanced = cv.cvtColor(lab_eq, cv.COLOR_LAB2BGR)
        enhanced=cv.GaussianBlur(enhanced, (5, 5), 0)

        img_pil = Image.fromarray(enhanced)

        img_tensor = img_transform(img_pil)
        img_tensor = img_tensor[:, -crop_size:, :]
        return img_tensor.unsqueeze(0)

    def lane_filter(self, points, dist_thresh=80, threshold_deg=30):
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
        filtered.append(points[-1])
        return filtered
    def lanedete(self, frame, net, img_transforms, cfg, cls_num_per_lane, img_h, img_w):
        image_tensor = self.process_lane_image(frame, img_transforms, crop_size= cfg.train_height)
        image_tensor = image_tensor.cuda()

        with torch.no_grad():
            pred = net(image_tensor)

        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w, original_image_height=img_h)

        for lane in coords:
            if len(lane) > 2:
                lane = self.lane_filter(lane)
                lane = np.array(lane, dtype=np.int32)
                cv.polylines(frame, [lane], isClosed=False, color=(0, 255, 0), thickness=5)

        return frame

    def detect_and_track(self, frame):
        results = self.car_model.track(frame, persist=True, conf = 0.4)
        return results
    
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
                
                #更新資訊
                self.tracked_vehicles[track_id]['id'] = track_id
                self.tracked_vehicles[track_id]['bbox'] = (x1, y1, x2 - x1, y2 - y1)
                self.tracked_vehicles[track_id]['last_seen'] = cur_time
        
        return vehicle_regions
    
    def split_image(self, carimg, rows=3, cols=3):
        """將圖片分割成多個區域"""
        img_height, img_width,_ = carimg.shape
        
        # 計算每個區域的大小
        width_per_part = img_width // cols
        height_per_part = img_height // rows
        parts = []
        for i in range(rows):
            for j in range(cols):
                left = j * width_per_part
                upper = i * height_per_part
                right = (j + 1) * width_per_part
                lower = (i + 1) * height_per_part
                
                # 裁剪並添加到 parts
                part = carimg[upper:lower, left:right]
                parts.append((part, (left, upper, right, lower)))  # 傳回裁剪後的圖片和位置座標
        return parts
     
    def recognize_plate(self,i, plate_image):
        cropped_img_np = np.array(plate_image)
        cropped_img_np = sr_dnn.upsample(cropped_img_np)
        #Image.fromarray(cropped_img_np).save(f"debug_{i}.png")
        # Perform OCR on the cropped region
        result = ocr.ocr(cropped_img_np, cls=False)
        if result and result[0]:  # Check if result is not empty
            data = result[0][0][1]  # Extract recognized text and its confidence
            text, confidence = data
            print(f"Region {i + 1}: Detected text: '{text}', Confidence: {confidence:.2f}")
            return text
        
    def detect_license_plate(self, vehicle_image):
        img_parts = self.split_image(vehicle_image)

        for i, (part, (left, upper, right, lower)) in enumerate(img_parts):
            backup_part = np.array(part)
            lab = cv.cvtColor(backup_part, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced_lab = cv.merge([l, a, b])
            part = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)
            result = self.licence_model(vehicle_image)

            if result[0].boxes is None or len(result[0].boxes) == 0:
                continue

            best_plate = result[0].boxes[0]
            x1, y1, x2, y2 = map(int, best_plate.xyxy[0])
            plate_region = vehicle_image[y1:y2, x1:x2]

            plate_text = self.recognize_plate(i, plate_region)
            
            return plate_text, (left + x1, upper + y1, x2-x1, y2-y1)
            
        return None, None
    
    def analyze_flashing_pattern(self, signal_history, side, previous_light_state):
        side_history = [(state, timestamp) for s, state, timestamp in signal_history if s == side]
        print(f"{side}: {side_history}")
        recent_states = [state for state, _ in side_history[-12:]]
        #if len(side_history) < 10:
        #    return "1OFF" 
        transitions = 0
        for i in range(1, len(side_history)):
            if side_history[i][0] != side_history[i-1][0]:
                transitions += 1
        if transitions >= 5:
            return "FLASHING"
        if sum(recent_states) < 3:
            return "2OFF"
        elif sum(recent_states) > 9:
            # if previous_light_state == "FLASHING":
            #     return "FLASHING"
            # else:
            #     return "ON"
            return "ON"
        else:
            return "FLASHING"
    
    def analyze_signal_region(self, region, side, vehicle_id, cur_time, previous_light):
        hsv = cv.cvtColor(region, cv.COLOR_BGR2HSV)
        brightness_threshold = 100
        color_threshold = 0.005
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([179, 30, 255])
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        white_mask = cv.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
        combined_mask = cv.bitwise_or(white_mask, yellow_mask)
        combined_ratio = cv.countNonZero(combined_mask) / (region.shape[0] * region.shape[1])
        brightness = 0
        if cv.countNonZero(combined_mask) > 0:
            brightness = cv.mean(hsv[:, :, 2], mask=combined_mask)[0]
        #1:亮 0:暗
        cur_state = 1 if combined_ratio > color_threshold and brightness > brightness_threshold else 0

        signal_history = self.tracked_vehicles[vehicle_id]['signal_history']
        print(f"ID: {vehicle_id}:")
        signal_history.append((side, cur_state, cur_time))
        return self.analyze_flashing_pattern(signal_history, side , previous_light)
     
    def process_frame(self, cur_time, frame, frame_width, frame_height):
        results = self.detect_and_track(frame)

        vehicle_regions = self.extract_vehicle_regions(cur_time, frame, results)
        
        vehicle_data = {}
        processed_frame = frame.copy()
        light_data = {}
        for vehicle_id, region in vehicle_regions.items():
            #車牌
            plate_text, plate_bbox = self.detect_license_plate(region['image'])
            if plate_text:
                self.tracked_vehicles[vehicle_id]['license_plate'] = plate_text
            
            #方向燈(detect light)
            car_x, car_y, car_h, car_w = region['bbox']
            center_x = car_w / 2
            results = self.light_model(region['image'], conf = 0.45)
            boxes = results[0].boxes
            light_number = 0
            left_state = self.tracked_vehicles[vehicle_id]['left_signal']
            right_state = self.tracked_vehicles[vehicle_id]['right_signal']
            if len(boxes) == 2:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                    class_id = int(box.cls[0])
                    object_name = results[0].names[class_id]
                    if object_name == "carLight":
                        cx = (x1 + x2) / 2
                        light_data[light_number] = {
                            'light_bbox' : (x1, y1, x2, y2),
                            'light_cx': cx,
                            'image': region['image'][y1:y2, x1:x2]
                        }
                        light_number += 1
                mid_light_x = sum([data['light_cx'] for data in light_data.values()]) / 2
                for _, light_info in light_data.items():
                    light_x1, light_y1, light_x2, light_y2 = light_info['light_bbox']
                    light_center_x = (light_x1 + light_x2) / 2
                    side = 'left' if mid_light_x >= light_center_x else 'right'
                    light_region = light_info['image']
                    if side == "left":
                        previous_light_left = self.tracked_vehicles[vehicle_id]['left_signal']
                        left_state = self.analyze_signal_region(light_region, 'left', vehicle_id, cur_time, previous_light_left)
                        self.tracked_vehicles[vehicle_id]['left_signal'] = left_state
                    else:
                        previous_light_right = self.tracked_vehicles[vehicle_id]['left_signal']
                        right_state = self.analyze_signal_region(light_region, 'right', vehicle_id, cur_time, previous_light_right)
                        self.tracked_vehicles[vehicle_id]['right_signal'] = right_state
            else:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                    class_id = int(box.cls[0])
                    object_name = results[0].names[class_id]
                    light_center_y = (y1 + y2) / 2
                    if object_name == "carLight" and ( car_h*0.2 < light_center_y < car_h*0.8 ):
                        light_center = (x1 + x2) / 2
                        side = 'left' if center_x  > light_center else 'right'
                        light_region =region['image'][y1:y2, x1:x2]
                        if side == "left":
                            previous_light_left = self.tracked_vehicles[vehicle_id]['left_signal']
                            left_state = self.analyze_signal_region(light_region, 'left', vehicle_id, cur_time, previous_light_left)
                            self.tracked_vehicles[vehicle_id]['left_signal'] = left_state
                        else:
                            previous_light_right = self.tracked_vehicles[vehicle_id]['left_signal']
                            right_state = self.analyze_signal_region(light_region, 'right', vehicle_id, cur_time, previous_light_right)
                            self.tracked_vehicles[vehicle_id]['right_signal'] = right_state
            
            #車輛資訊
            vehicle_data[vehicle_id] = {
                'bbox': region['bbox'],
                'license_plate': self.tracked_vehicles[vehicle_id]['license_plate'],
                'left_signal': left_state,
                'right_signal': right_state
            }
            self.draw_info(processed_frame, vehicle_id, vehicle_data[vehicle_id])
        return processed_frame, vehicle_data
        
    def draw_info(self, frame, vehicle_id, vehicle_info):
        bbox = vehicle_info['bbox']
        x, y, w, h = bbox
        # 繪製車輛邊界框
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 顯示車輛 ID
        cv.putText(frame, f"ID: {vehicle_id}", (x, y - 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # 顯示車牌號碼
        if vehicle_info['license_plate']:
            cv.putText(frame, f"Plate: {vehicle_info['license_plate']}", (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 顯示方向燈狀態
        color_map = {"OFF": (0, 0, 255), "1OFF": (0, 0, 255), "2OFF": (0, 0, 255),"FLASHING": (0, 255, 0), "ON": (255, 255, 0)}
        # 左方向燈標記
        left_color = color_map[vehicle_info['left_signal']]
        cv.putText(frame, f"L: {vehicle_info['left_signal']}", (x - 40, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
        # 右方向燈標記
        right_color = color_map[vehicle_info['right_signal']]
        cv.putText(frame, f"R: {vehicle_info['right_signal']}", (x + w - 20, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    monitor = YOLOv8()
    net, img_transforms, cls_num_per_lane, cfg=monitor.lanemodel()
    cap = cv.VideoCapture('videoplayback_3.mp4')#videoplayback_2 car_test motorcycle_data

    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_width= int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter("output2.mp4", fourcc, fps, (frame_width, frame_height)) 
    while True:
        ret, frame = cap.read()
        testframe=frame
        if not ret:
            break
        
        frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        cur = frame_num / fps

        processed_frame, vehicle_data = monitor.process_frame(cur, frame, frame_width, frame_height)
        lane_frame=monitor.lanedete(processed_frame, net, img_transforms, cfg, cls_num_per_lane, frame_height, frame_width)
        out.write(lane_frame)
        cv.imshow('Traffic Monitor', lane_frame)

        if cv.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv.destroyAllWindows()