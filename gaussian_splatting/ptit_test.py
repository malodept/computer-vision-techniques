import struct
import math

bin_path = r"D:\malo\Documents\cours_tsp\cv\gaussian_splatting\colmap_project\sparse\0\points3D.bin"

# Nouvelle structure : 888 bytes = Q 3d 3d H 3B d 100Q
point_struct = struct.Struct("Q3d3dH3Bd100Q")

with open(bin_path, "rb") as f:
    i = 0
    while True:
        data = f.read(point_struct.size)
        if len(data) < point_struct.size:
            break

        unpacked = point_struct.unpack(data)
        x, y, z = unpacked[1:4]
        r, g, b = unpacked[10:13]

        if any(map(math.isnan, (x, y, z))):
            continue  # skip NaN
        if not all(0 <= c <= 255 for c in (r, g, b)):
            r, g, b = 128, 128, 128  # fallback color

        print(f"Point {i+1}: x={x:.2f}, y={y:.2f}, z={z:.2f} | r={r} g={g} b={b}")
        i += 1
        if i >= 10:
            break
