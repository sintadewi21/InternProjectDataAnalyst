from PIL import Image
import sys
import numpy as np

def remove_background(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    datas = img.getdata()

    newData = []
    # Threshold for "white" or "checkerboard light/dark grey"
    # Checkerboard usually alternates white and light grey.
    # We will assume the logo is dark (blue/purple) and background is light.
    
    for item in datas:
        # Check if pixel is white-ish or light grey (checkerboard)
        # RGB > 200, 200, 200 is a safe bet for white/light grey background if logo is dark.
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            newData.append((255, 255, 255, 0)) # Make Transparent
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(output_path, "PNG")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        pass
    else:
        remove_background(sys.argv[1], sys.argv[2])
