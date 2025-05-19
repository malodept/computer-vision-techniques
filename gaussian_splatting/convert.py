from PIL import Image
import os

input_dir = r"D:\malo\Documents\cours_tsp\cv\gaussian_splatting\photos\Photos-001"
output_dir = r"D:\malo\Documents\cours_tsp\cv\gaussian_splatting\photos\Photos-001-png"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        img = Image.open(os.path.join(input_dir, filename)).convert("RGB")
        base = os.path.splitext(filename)[0]
        img.save(os.path.join(output_dir, base + ".png"))
