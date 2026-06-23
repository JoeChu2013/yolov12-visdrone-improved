import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def process_image():
    # Since I cannot download the image directly, I will assume the user 
    # saves it as 'ghostnet.png' in the current directory.
    img_path = "ghostnet.png" 
    out_path = "ghostnet_annotated.png"
    
    if not os.path.exists(img_path):
        print(f"请先将您提供的图片保存到 {os.path.abspath(img_path)}，然后再运行此脚本。")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片，请确保格式正确。")
        return
        
    h, w = img.shape[:2]

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load font
    font_size = int(h * 0.04) 
    if font_size < 12: font_size = 12
    
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc", # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf", # SimHei
        "C:/Windows/Fonts/simsun.ttc" # SimSun
    ]
    
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, font_size)
            break
            
    if font is None:
        font = ImageFont.load_default()
        print("警告: 未找到中文字体，可能无法正常显示中文。")
    
    # Bounding boxes to cover original (a) and (b)
    # Based on the image, (a) is roughly at 30-40% height, (b) is at 85-95% height
    rect_a_x1 = int(w * 0.40)
    rect_a_x2 = int(w * 0.60)
    rect_a_y1 = int(h * 0.32)
    rect_a_y2 = int(h * 0.42)
    
    rect_b_x1 = int(w * 0.40)
    rect_b_x2 = int(w * 0.60)
    rect_b_y1 = int(h * 0.88)
    rect_b_y2 = int(h * 0.98)
    
    # Draw white rectangles
    draw.rectangle([rect_a_x1, rect_a_y1, rect_a_x2, rect_a_y2], fill=(255, 255, 255))
    draw.rectangle([rect_b_x1, rect_b_y1, rect_b_x2, rect_b_y2], fill=(255, 255, 255))
    
    text_a = "(a) 传统卷积生成特征图方式"
    text_b = "(b) GhostNet模块的特征生成机制"
    
    # Calculate text position
    try:
        bbox_a = draw.textbbox((0, 0), text_a, font=font)
        text_w_a = bbox_a[2] - bbox_a[0]
        text_h_a = bbox_a[3] - bbox_a[1]
        
        bbox_b = draw.textbbox((0, 0), text_b, font=font)
        text_w_b = bbox_b[2] - bbox_b[0]
        text_h_b = bbox_b[3] - bbox_b[1]
    except AttributeError:
        # Older PIL fallback
        text_w_a, text_h_a = draw.textsize(text_a, font=font)
        text_w_b, text_h_b = draw.textsize(text_b, font=font)
    
    # Center text
    x_a = (w - text_w_a) // 2
    y_a = rect_a_y1 + (rect_a_y2 - rect_a_y1 - text_h_a) // 2
    
    x_b = (w - text_w_b) // 2
    y_b = rect_b_y1 + (rect_b_y2 - rect_b_y1 - text_h_b) // 2
    
    draw.text((x_a, y_a), text_a, font=font, fill=(0, 0, 0))
    draw.text((x_b, y_b), text_b, font=font, fill=(0, 0, 0))
    
    img_out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img_out)
    print(f"处理完成！图片已保存为: {out_path}")

if __name__ == "__main__":
    process_image()
