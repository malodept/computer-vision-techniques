import pycolmap
from pathlib import Path

sparse_path = Path(r"D:\malo\Documents\cours_tsp\cv\gaussian_splatting\colmap_project\sparse\0")
reconstruction = pycolmap.Reconstruction(sparse_path)

print("\nImages référencées dans images.bin :")
for image in reconstruction.images.values():
    print("→", image.name)
