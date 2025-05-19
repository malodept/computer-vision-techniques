import os
import struct
import numpy as np
from plyfile import PlyData, PlyElement
import math

def read_points3D_bin(bin_path):
    points3D = {}
    with open(bin_path, "rb") as f:
        while True:
            binary_data = f.read(872)
            if len(binary_data) != 872:
                break  # On arrête proprement à la fin du fichier
            unpacked = struct.unpack("Q3d3dHd100Q", binary_data)
            track_id = unpacked[0]
            xyz = unpacked[1:4]
            rgb = unpacked[4:7]
            error = unpacked[7]
            points3D[track_id] = (xyz, rgb, error)
    return points3D

def export_ply(points3D, output_path):
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3D)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for (x, y, z), (r, g, b), error in points3D.values():
            if any(math.isnan(v) for v in (x, y, z, r, g, b)):
                continue  # Ignore les points invalides
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

# Exécution
input_path = "D:/malo/Documents/cours_tsp/cv/gaussian_splatting/colmap_project/sparse/0/points3D.bin"
output_path = "D:/malo/Documents/cours_tsp/cv/gaussian_splatting/exports/splat.ply"
points3D = read_points3D_bin(input_path)
export_ply(points3D, output_path)
print("PLY exported to", output_path)
