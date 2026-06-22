import os
import subprocess
import shutil
import glob
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def dslr_print_quality_v3(img):
    """
    V3商业证件照终版（打印级）
    模拟：Canon / Sony 人像镜头
    """

    img = img.astype(np.float32)

    img = (img - 128) * 1.03 + 128

    blur = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = cv2.addWeighted(img, 1.06, blur, -0.06, 0)

    img = np.clip(img, 0, 255)
    img = (img - 127) * 1.04 + 127

    b, g, r = cv2.split(img)
    r = r * 1.01
    g = g * 1.00
    b = b * 0.99
    img = cv2.merge([b, g, r])

    return np.clip(img, 0, 255).astype(np.uint8)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(("jpg", "jpeg", "png")):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件（JPG/PNG）")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename)[1]
    safe_filename = f"upload_{timestamp}{ext}"
    
    for f in INPUT_DIR.rglob("*.[jpJP][pnPN]*[gG]"):
        f.unlink()
    
    file_path = INPUT_DIR / safe_filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"success": True, "message": "上传成功"}


@app.post("/process")
async def process_image(
    w: float = 0.35,
    use_blend: bool = True,
    original_ratio: float = 0.88,
    ai_ratio: float = 0.12,
    use_esrgan: bool = False
):
    blend_ratio = (original_ratio, ai_ratio)
    input_files = list(INPUT_DIR.glob("*.[jpJP][pnPN]*[gG]"))
    if not input_files:
        raise HTTPException(status_code=400, detail="请先上传图片")
    
    original_file = input_files[0]
    
    for f in OUTPUT_DIR.rglob("*"):
        if f.is_file():
            f.unlink()
    
    python_path = os.path.join(os.environ.get("CONDA_PREFIX", ""), "python.exe")
    if not os.path.exists(python_path):
        python_path = "C:\\ProgramData\\miniconda3\\envs\\idphoto\\python.exe"
    
    result = subprocess.run(
        [python_path, "inference_codeformer.py", "-w", str(w), "-i", "inputs", "-o", "outputs"],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        encoding="utf-8",
        errors="replace"
    )
    
    if result.returncode != 0:
        error_detail = result.stderr[:500] if len(result.stderr) > 500 else result.stderr
        raise HTTPException(status_code=500, detail=f"处理失败: {error_detail}")
    
    final_results_dir = OUTPUT_DIR / "final_results"
    if not final_results_dir.exists():
        raise HTTPException(status_code=500, detail="未生成处理结果目录")
    
    png_files = list(final_results_dir.glob("*.png"))
    jpg_files = list(final_results_dir.glob("*.jpg"))
    all_files = png_files + jpg_files
    
    if not all_files:
        raise HTTPException(status_code=500, detail="未生成处理结果文件")
    
    latest_file = max(all_files, key=lambda f: f.stat().st_mtime)
    
    if use_blend:
        original_img = cv2.imread(str(original_file))
        ai_img = cv2.imread(str(latest_file))
        
        if original_img is None or ai_img is None:
            raise HTTPException(status_code=500, detail="图片读取失败")
        
        ai_img = cv2.resize(ai_img, (original_img.shape[1], original_img.shape[0]))
        
        final_img = cv2.addWeighted(original_img, blend_ratio[0], ai_img, blend_ratio[1], 0)
        final_img = dslr_print_quality_v3(final_img)
        
        result_filename = f"blended_{os.path.basename(str(latest_file))}"
        blended_path = final_results_dir / result_filename
        cv2.imwrite(str(blended_path), final_img)
    else:
        result_filename = os.path.basename(str(latest_file))
    
    return {
        "success": True,
        "result_url": f"/outputs/{result_filename}",
        "w": w,
        "use_blend": use_blend,
        "blend_ratio": blend_ratio,
        "mode": "v3_print_quality",
        "output_type": "commercial_id_photo",
        "print_ready": True,
        "ai_ratio": ai_ratio
    }


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    final_results_dir = OUTPUT_DIR / "final_results"
    file_path = final_results_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(file_path))


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
