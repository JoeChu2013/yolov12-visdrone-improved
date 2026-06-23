import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

def put_chinese_text(img, text, position, font_path, font_size, text_color=(0, 0, 0)):
    # Convert cv2 image to PIL image
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=text_color)
    # Convert PIL image back to cv2 image
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return cv2_im_processed

def process_image(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load image {img_path}")
        return

    # Image dimensions
    H, W = img.shape[:2]
    
    # We will use fixed relative positions since the layout of this figure is standard.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find black text (pixels < 100)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate to connect text components
    kernel = np.ones((5, 15), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Font settings
    font_path = "C:/Windows/Fonts/simsun.ttc"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/simsun.ttf"
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out noise
        if w < 10 or h < 8 or h > 100:
            continue
            
        cx, cy = x + w/2, y + h/2
        
        is_text = False
        text_to_draw = ""
        font_size = 20
        
        if cy > H * 0.35 and cy < H * 0.45 and cx > W * 0.2 and cx < W * 0.8:
            is_text = True
            text_to_draw = "(a) MS COCO (左) 与 VisDrone (右) 实例尺度对比"
            font_size = max(18, int(h * 0.8))
        elif cy > H * 0.92:
            is_text = True
            text_to_draw = "(b) VisDrone 数据集目标特征"
            font_size = max(18, int(h * 0.8))
        elif cx < W * 0.2:
            is_text = True
            if cy < H * 0.7:
                text_to_draw = "人"
            elif cy < H * 0.82:
                text_to_draw = "摩托车"
            else:
                text_to_draw = "汽车"
            font_size = max(16, int(h * 0.9))
        elif cx > W * 0.8:
            is_text = True
            if cy < H * 0.7:
                text_to_draw = "行人"
            elif cy < H * 0.82:
                text_to_draw = "摩托车"
            else:
                text_to_draw = "三轮车"
            font_size = max(16, int(h * 0.9))
            
        if is_text:
            # Mask out the original text with white
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 255), -1)
            
            # Calculate text width approx to center it
            approx_char_width = font_size
            text_w = len(text_to_draw) * approx_char_width
            # If it's a title, center it based on original box center
            draw_x = int(cx - text_w/2)
            # If it's a side label, center it horizontally in the box
            if cx < W * 0.2 or cx > W * 0.8:
                draw_x = int(x + w/2 - len(text_to_draw)*font_size/2)
                
            img = put_chinese_text(img, text_to_draw, (draw_x, y-2), font_path, font_size)

    cv2.imwrite(out_path, img)
    print(f"Processed image saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='translated_diagram.png', help='Path to output image')
    args = parser.parse_args()
    process_image(args.input, args.output)
