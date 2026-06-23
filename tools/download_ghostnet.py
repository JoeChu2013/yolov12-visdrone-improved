import cv2
import numpy as np
import urllib.request
import os

# Download the image from the URL provided in the prompt
url = "https://picx.zhimg.com/80/v2-949e2908f99e3ec014b2d5a37172080a_1440w.webp?source=1def8aca"
img_path = "ghostnet.png"

try:
    print(f"Downloading image from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with open(img_path, 'wb') as out_file:
            out_file.write(response.read())
    print("Download successful.")
except Exception as e:
    print(f"Failed to download image: {e}")
