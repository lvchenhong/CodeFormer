import sys
import os
import cv2
import torch
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python enhance_sr.py <input_path>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = input_path.replace('.png', '_sr.png').replace('.jpg', '_sr.jpg').replace('.jpeg', '_sr.jpeg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from facelib.detection.retinaface.retinaface import RetinaFace

detector = RetinaFace(network_name='resnet50', half=True if device.type == 'cuda' else False)

img = cv2.imread(input_path)
if img is None:
    print(f"Error: Cannot read image {input_path}")
    sys.exit(1)

h, w = img.shape[:2]

faces = detector.detect_faces(img)

face_mask = np.zeros((h, w), dtype=np.float32)

if len(faces) > 0:
    faces = faces[faces[:, 4] > 0.9]
    
    for face in faces:
        x1, y1, x2, y2 = face[:4].astype(int)
        x1 = max(0, x1 - 20)
        y1 = max(0, y1 - 30)
        x2 = min(w, x2 + 20)
        y2 = min(h, y2 + 20)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        roi_h, roi_w = y2 - y1, x2 - x1
        if roi_h < 20 or roi_w < 20:
            continue
        
        center_x = roi_w // 2
        center_y = roi_h // 2
        radius_x = roi_w // 2 + 15
        radius_y = roi_h // 2 + 15
        
        xx, yy = np.meshgrid(np.arange(roi_w), np.arange(roi_h))
        dx = (xx - center_x) / radius_x
        dy = (yy - center_y) / radius_y
        dist = np.sqrt(dx * dx + dy * dy)
        
        roi_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        roi_mask[dist < 0.8] = 1.0
        transition = (dist >= 0.8) & (dist < 1.1)
        roi_mask[transition] = 1.0 - (dist[transition] - 0.8) / 0.3
        
        face_mask[y1:y2, x1:x2] = np.maximum(face_mask[y1:y2, x1:x2], roi_mask)

from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

model_path = 'weights/RealESRGAN_x4plus.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file not found {model_path}")
    sys.exit(1)

state_dict = torch.load(model_path, map_location=device)
if 'params_ema' in state_dict:
    state_dict = state_dict['params_ema']
elif 'params' in state_dict:
    state_dict = state_dict['params']

model.load_state_dict(state_dict, strict=True)
model.eval()
model = model.to(device)
model = model.half()

bg_img = img.copy()
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
bg_img = bg_img.astype(np.float32) / 255.0
bg_img = torch.from_numpy(np.transpose(bg_img, (2, 0, 1))).float()
bg_img = bg_img.unsqueeze(0).to(device)

tile = 256
tile_pad = 10
scale = 4

orig_h, orig_w = bg_img.shape[2:]
out_h, out_w = orig_h * scale, orig_w * scale

bg_output = torch.zeros(1, 3, out_h, out_w).to(device)

tile_cnt = 0
for y in range(0, orig_h, tile - tile_pad * 2):
    for x in range(0, orig_w, tile - tile_pad * 2):
        tile_y = y
        tile_x = x
        tile_h = min(tile, orig_h - tile_y)
        tile_w = min(tile, orig_w - tile_x)
        
        tile_img = bg_img[:, :, tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
        
        pre_pad = 10
        tile_img = torch.nn.functional.pad(tile_img, (pre_pad, pre_pad, pre_pad, pre_pad), 'reflect')
        tile_img = tile_img.half()
        
        with torch.no_grad():
            tile_output = model(tile_img)
        
        tile_output = tile_output[:, :, pre_pad * scale:(pre_pad + tile_h) * scale, pre_pad * scale:(pre_pad + tile_w) * scale]
        
        bg_output[:, :, tile_y * scale:(tile_y + tile_h) * scale, tile_x * scale:(tile_x + tile_w) * scale] = tile_output
        
        tile_cnt += 1

bg_output = bg_output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
bg_output = np.transpose(bg_output[[2, 1, 0], :, :], (1, 2, 0))
bg_output = (bg_output * 255.0).round().astype(np.uint8)

face_img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

face_mask_upscaled = cv2.resize(face_mask, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
face_mask_upscaled = np.expand_dims(face_mask_upscaled, axis=2)
face_mask_upscaled = np.repeat(face_mask_upscaled, 3, axis=2)

feather_radius = 20
kernel_size = feather_radius * 2 + 1
face_mask_upscaled = cv2.GaussianBlur(face_mask_upscaled, (kernel_size, kernel_size), feather_radius / 3)

final_img = face_img.astype(np.float32) * face_mask_upscaled + bg_output.astype(np.float32) * (1.0 - face_mask_upscaled)
final_img = final_img.astype(np.uint8)

gamma = 1.05
final_img = np.power(final_img / 255.0, gamma) * 255.0

final_img = cv2.convertScaleAbs(final_img, alpha=1.02, beta=0)

saturation = 1.03
hsv = cv2.cvtColor(final_img, cv2.COLOR_BGR2HSV)
hsv[:, :, 1] = hsv[:, :, 1] * saturation
final_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

final_img = np.clip(final_img, 0, 255).astype(np.uint8)

cv2.imwrite(output_path, final_img)

print(f"DONE: {output_path}")
