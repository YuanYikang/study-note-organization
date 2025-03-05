import os
from pathlib import Path
import re
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from PIL import Image
import torch
import pandas as pd
import csv

# 配置参数
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "GroundingDINO/groundingdino_swint_ogc.pth"
DEVICE = "cpu"
# INPUT_DIR = "street_images"  # 输入图片目录

BASE_PATH = r"E:/TencentSV/mnt/nas/huangyj/"  # 基础路径
# CSV_PATH = "TX_xy_path.csv"  # CSV文件路径
CSV_PATH = "new_xy.csv"

OUTPUT_DIR = "result01"   # 裁剪后的广告牌保存目录

TEXT_PROMPT = "billboard . signboard . advertisement . signage ."
BOX_TRESHOLD = 0.15
TEXT_TRESHOLD = 0.25

'''
def parse_coordinates(filename):
    """从文件名解析经纬度"""
    pattern = r"(\d+)_(-?\d+\.\d+),(-?\d+\.\d+)_([A-Z])\.png"
    match = re.search(pattern, filename)
    if match:
        print(filename, match.group(2), match.group(3))
        image_id, lng, lat, street_id = match.groups()
        return int(image_id), float(lng), float(lat), street_id
    return None, None, None, None
'''


def process_images():
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(CSV_PATH)

    # 加载模型
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    
    # 遍历CSV中的每一行
    for index, row in df.iterrows():
        try:
            # 构建完整的图片路径
            img_path = os.path.join(BASE_PATH, row['new_path'])
            img_path = Path(img_path)
            lon = row['lon_wgs']
            lat = row['lat_wgs']
            
            if not os.path.exists(img_path):
                print(f"图片不存在: {img_path}")
                continue
                
            # 加载并处理图片
            image_source, image = load_image(str(img_path))
            
            # 预测广告牌位置
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD,
                    device=DEVICE
                )
            
            if len(boxes) == 0:
                print(f"未检测到广告牌")
                continue

            # 保存标注后的完整图片
            img_name = img_path.stem
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(str(Path(OUTPUT_DIR) / f"annotated_{img_name}.jpg"), annotated_frame)
            
            # 裁剪并保存每个检测到的广告牌
            image_pil = Image.open(img_path)
            width, height = image_pil.size
            print(f"原始图片尺寸: {width}x{height}")

            for i, (box, score) in enumerate(zip(boxes, logits)):
                # 转换坐标从相对值(0-1)到像素值
                # cx, cy, x_w, y_h = box.tolist()
                # 计算像素坐标
                '''
                x1_px = max(0, int(x1 * width))
                x2_px = min(width, int(x2 * width))
                y1_px = max(0, int(y1 * height))
                y2_px = min(height, int(y2 * height))
                '''

                cx, cy, x_w, y_h = [int(coord * dim) for coord, dim in 
                                    zip(box.tolist(), [width, height]*2)]
                x1 = cx - x_w // 2
                y1 = cy - y_h // 2
                x2 = cx + x_w // 2
                y2 = cy + y_h // 2
                x1_px = max(0, x1)
                x2_px = min(width, x2)
                y1_px = max(0, y1)
                y2_px = min(height, y2)

                print(f"框 {i}: 原始坐标 ({cx:.3f}, {cy:.3f}, {x_w:.3f}, {y_h:.3f})")
                print(f"框 {i}: 像素坐标 ({x1_px}, {y1_px}, {x2_px}, {y2_px})")
                
                try:
                    # 裁剪图像
                    crop_img = image_pil.crop((x1_px, y1_px, x2_px, y2_px))

                    # 如果是RGBA模式，转换为RGB
                    if crop_img.mode == 'RGBA':
                        crop_img = crop_img.convert('RGB')

                    # 使用经纬度信息命名
                    crop_name = f"ad_{lon:.6f}_{lat:.6f}_{i}_conf{float(score):.2f}.jpg"
                    crop_path = str(Path(OUTPUT_DIR) / crop_name)
                    crop_img.save(crop_path)
                    print(f"成功保存裁剪图片: {crop_name}")
                    
                
                # 验证裁剪结果
                    if os.path.exists(crop_path):
                        saved_img = Image.open(crop_path)
                        print(f"裁剪尺寸: {saved_img.size}")
                except Exception as e:
                    print(f"裁剪失败: {str(e)}")                    
                
            print(f"处理完成: {img_path.name}, 检测到{len(boxes)}个广告牌")
            
        except Exception as e:
            print(f"处理{img_path.name}时出错: {str(e)}")
            continue

if __name__ == "__main__":
    with torch.no_grad():
        process_images()