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

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
img = img.unsqueeze(0).to(device)

tile = 512
tile_pad = 10
scale = 4

h, w = img.shape[2:]
out_h, out_w = h * scale, w * scale

output = torch.zeros(1, 3, out_h, out_w).to(device)

tile_cnt = 0
for y in range(0, h, tile - tile_pad * 2):
    for x in range(0, w, tile - tile_pad * 2):
        tile_y = y
        tile_x = x
        tile_h = min(tile, h - tile_y)
        tile_w = min(tile, w - tile_x)
        
        tile_img = img[:, :, tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
        
        pre_pad = 10
        tile_img = torch.nn.functional.pad(tile_img, (pre_pad, pre_pad, pre_pad, pre_pad), 'reflect')
        tile_img = tile_img.half()
        
        with torch.no_grad():
            tile_output = model(tile_img)
        
        tile_output = tile_output[:, :, pre_pad * scale:(pre_pad + tile_h) * scale, pre_pad * scale:(pre_pad + tile_w) * scale]
        
        output[:, :, tile_y * scale:(tile_y + tile_h) * scale, tile_x * scale:(tile_x + tile_w) * scale] = tile_output
        
        tile_cnt += 1

output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round().astype(np.uint8)

cv2.imwrite(output_path, output)

print(f"DONE: {output_path}")
