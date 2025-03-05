# 安装CLIP工具链
!pip install git+https://github.com/openai/CLIP.git

import clip
import torch
from PIL import Image
import os
import pandas as pd
from pathlib import Path
import re

# 加载预训练模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("CLIP-main/ViT-L-14-336px.pt", device=device)

# 定义风格标签（可扩展）
# style_labels = ["简约现代", "复古传统", "赛博朋克", "自然生态", "卡通动漫"]

style_labels_ = [

    "High Contrast",     # 高对比度
    "Medium Contrast",   # 中对比度
    "Low Contrast"       # 低对比度
    

]

style_labels = [
    "Modern Minimalist",     # 简约现代
    "Vintage Traditional",   # 复古传统
    "Mainstream Commercial", # 大众市场
    "Cyberpunk",             # 赛博朋克
    "Dopamine Style",        # 多巴胺风格
]
style_labels_03 = [
    "Mass Commercial Style, Eye-catching mass-market advertising with bold promotional offers, vibrant colors, and straightforward messaging.", # 大众市场广告，大胆的促销优惠，鲜艳的色彩，直接的信息传递
    "Luxury Minimalist Style, Sophisticated minimalist design for luxury brands featuring monochromatic palette, premium textures, and ample negative space.", # 高雅的广告，注重奢侈品、品质和独特性
    "Retro Traditional Style, Nostalgic traditional aesthetics with vintage color schemes, classical patterns, and artisanal craftsmanship details."
]

style_labels_01 = [
    "Modern Minimalist",     # 简约现代
    "Vintage Traditional",   # 复古传统 
    "Cyberpunk",            # 赛博朋克
    # "Natural Ecological",    # 自然生态
    # "Cartoon Anime"         # 卡通动漫
]
# 色调风格标签
# style_labels = ["清新淡雅", "明快活泼", "高冷优雅", "低饱和度", "高饱和度"]
style_labels_02 = [
    "Fresh Elegant",        # 清新淡雅
    "Bright Lively",        # 明快活泼
    "Cool Sophisticated",   # 高冷优雅
    "Low Saturation",       # 低饱和度
    "High Saturation"       # 高饱和度
]

def extract_info_from_filename(filename):
    # 从文件名提取信息
    # pattern = r'ad_(\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+)_conf(\d+\.\d+)_([A-Z])\.jpg'
    pattern = r'ad_(\d+\.\d+)_(\d+\.\d+)_(\d+)_conf(\d+\.\d+)\.jpg'
    match = re.match(pattern, filename)
    if match:
        lng, lat, billboard_id, conf = match.groups()
        return float(lng), float(lat), int(billboard_id), float(conf)
    return None

# 处理裁剪后的广告牌图片
image = preprocess(Image.open("street_images/1.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(style_labels).to(device)

# 计算相似度
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 计算相似度
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(3)

# 输出风格概率
print("Label probs:", probs)
# 输出风格标签
print("Top styles:")
for value, index in zip(values, indices):
    print(f"{style_labels[index]:>25s}: {100 * value.item():.2f}%")
# 示例输出：{'简约现代':0.6, '复古传统':0.3, ...}

def analyze_billboard_styles(image_folder, output_csv):
    results = []
    
    # 遍历文件夹中的所有图片
    for img_file in os.listdir(image_folder):
        if img_file.startswith('ad_') and img_file.endswith('.jpg'):
            # 提取文件名信息
            info = extract_info_from_filename(img_file)
            if info:
                lng, lat, billboard_id, conf = info
                
                # 加载并处理图片
                image_path = os.path.join(image_folder, img_file)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                
                # 计算风格特征
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    
                    # 获取top1风格
                    value, index = similarity[0].topk(1)
                    top_style = style_labels[index]
                    style_score = value.item()

                    # 风格偏离度
                    deviation = 1 - value.item()
                
                # 保存结果
                results.append({
                    
                    'longitude': lng,
                    'latitude': lat,
                    'billboard_id': billboard_id,
                    'confidence': conf,
                    'style': top_style,
                    'style_score': style_score,
                    'deviation': deviation
                })
                
                print(f"处理完成: {img_file}")
    # 保存为CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"结果已保存至: {output_csv}")

image_folder = "result01"  # 广告牌图片文件夹
output_csv = "contrast.csv"   # 输出文件名

analyze_billboard_styles(image_folder, output_csv)