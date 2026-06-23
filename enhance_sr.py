import sys
import os
import cv2
import torch
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python enhance_sr.py <input_path> <mask_path>")
    sys.exit(1)

input_path = sys.argv[1]
mask_path = sys.argv[2]
output_path = input_path.replace('.png', '_sr.png').replace('.jpg', '_sr.jpg').replace('.jpeg', '_sr.jpeg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

img = cv2.imread(input_path)
if img is None:
    print(f"Error: Cannot read image {input_path}")
    sys.exit(1)

h, w = img.shape[:2]

face_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if face_mask is None:
    face_mask = np.zeros((h, w), dtype=np.float32)
else:
    face_mask = cv2.resize(face_mask, (w, h))
    face_mask = face_mask.astype(np.float32) / 255.0

mask_3ch = np.expand_dims(face_mask, axis=2)
mask_3ch = np.repeat(mask_3ch, 3, axis=2)

bg_img = img.astype(np.float32) * (1.0 - mask_3ch)
bg_img = bg_img.astype(np.uint8)

bg_tensor = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
bg_tensor = bg_tensor.astype(np.float32) / 255.0
bg_tensor = torch.from_numpy(np.transpose(bg_tensor, (2, 0, 1))).float()
bg_tensor = bg_tensor.unsqueeze(0).to(device)

tile = 512
tile_pad = 10
scale = 4

orig_h, orig_w = bg_tensor.shape[2:]
out_h, out_w = orig_h * scale, orig_w * scale

bg_output = torch.zeros(1, 3, out_h, out_w).to(device)

for y in range(0, orig_h, tile - tile_pad * 2):
    for x in range(0, orig_w, tile - tile_pad * 2):
        tile_y = y
        tile_x = x
        tile_h = min(tile, orig_h - tile_y)
        tile_w = min(tile, orig_w - tile_x)
        
        tile_img = bg_tensor[:, :, tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
        
        pre_pad = 10
        tile_img = torch.nn.functional.pad(tile_img, (pre_pad, pre_pad, pre_pad, pre_pad), 'reflect')
        tile_img = tile_img.half()
        
        with torch.no_grad():
            tile_output = model(tile_img)
        
        tile_output = tile_output[:, :, pre_pad * scale:(pre_pad + tile_h) * scale, pre_pad * scale:(pre_pad + tile_w) * scale]
        
        bg_output[:, :, tile_y * scale:(tile_y + tile_h) * scale, tile_x * scale:(tile_x + tile_w) * scale] = tile_output

bg_output = bg_output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
bg_output = np.transpose(bg_output[[2, 1, 0], :, :], (1, 2, 0))
bg_output = (bg_output * 255.0).round().astype(np.uint8)

face_img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

face_mask_upscaled = cv2.resize(face_mask, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
face_mask_upscaled = np.expand_dims(face_mask_upscaled, axis=2)
face_mask_upscaled = np.repeat(face_mask_upscaled, 3, axis=2)

feather = 24
face_mask_upscaled = cv2.GaussianBlur(face_mask_upscaled, (feather*2+1, feather*2+1), feather/3)

face_mean = np.mean(face_img * face_mask_upscaled)
bg_mean = np.mean(bg_output * (1 - face_mask_upscaled))

if face_mean > 10 and bg_mean > 10:
    ratio = face_mean / (bg_mean + 1e-6)
    bg_output = bg_output * ratio
    bg_output = np.clip(bg_output, 0, 255)

bg_output[:, :, 2] = np.clip(bg_output[:, :, 2] * 1.02, 0, 255)
bg_output[:, :, 0] = np.clip(bg_output[:, :, 0] * 0.98, 0, 255)

final_img = face_img.astype(np.float32) * face_mask_upscaled + bg_output.astype(np.float32) * (1.0 - face_mask_upscaled)
final_img = np.clip(final_img, 0, 255).astype(np.uint8)

cv2.imwrite(output_path, final_img)

print(f"DONE: {output_path}")
