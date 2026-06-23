from PIL import Image, ImageDraw, ImageFont
import os

def annotate_image(image_path, output_path):
    try:
        # Open the image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try to load a Chinese font (SimHei or similar on Windows)
        try:
            # You might need to adjust the path based on your Windows fonts
            font = ImageFont.truetype("simhei.ttf", 24)
        except IOError:
            try:
                font = ImageFont.truetype("msyh.ttc", 24) # Microsoft YaHei
            except IOError:
                # Fallback to default if no Chinese font is found (will likely not render Chinese correctly)
                print("Warning: Chinese font not found. Text might not render correctly.")
                font = ImageFont.load_default()

        # Image dimensions to help with positioning
        width, height = img.size
        
        # Define text and approximate positions based on the provided image
        # These coordinates are estimates based on standard (a) and (b) placements in such figures.
        # We'll try to find the (a) and (b) text or just place it at estimated relative positions.
        
        text_a = "传统卷积生成特征图方式"
        text_b = "GhostNet模块的特征生成机制"
        
        # Assuming (a) is around x: center, y: 35% of height
        # Assuming (b) is around x: center, y: 90% of height
        # We will add the text to the right of the center.
        
        # Let's just place them manually based on visual inspection of the prompt image
        # Center is roughly width/2
        
        # Position for (a)
        # In the original image, (a) is centered below the top diagram.
        # Let's place the text right next to where (a) likely is.
        x_a = width / 2 + 30 
        y_a = height * 0.35
        
        # Position for (b)
        # In the original image, (b) is centered at the bottom.
        x_b = width / 2 + 30
        y_b = height * 0.95
        
        # We can also just draw white rectangles over the original (a) and (b) and rewrite the whole string
        # "(a) 传统卷积生成特征图方式" to make it look cleaner.
        
        # Draw text
        draw.text((x_a, y_a), text_a, fill="black", font=font)
        draw.text((x_b, y_b), text_b, fill="black", font=font)
        
        # Save the result
        img.save(output_path)
        print(f"Successfully saved annotated image to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Assuming the uploaded image is named 'ghostnet_diagram.png' or similar.
    # Since I don't have direct access to the uploaded file's local path, 
    # I will create a script that you can run if you save the image locally.
    pass
