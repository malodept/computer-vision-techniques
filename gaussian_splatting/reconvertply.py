import struct
import math

def read_points3D_bin(path):
    points3D = []
    with open(path, "rb") as f:
        while True:
            data = f.read(43 * 8 + 2)
            if len(data) < 43 * 8 + 2:
                break
            try:
                unpacked = struct.unpack("Q3d3dHd100Q", data)
                x, y, z = unpacked[1:4]
                r, g, b = unpacked[7:10]
                if all(map(lambda v: not math.isnan(v) and 0 <= v <= 255, (r, g, b))):
                    points3D.append((x, y, z, int(r), int(g), int(b)))
            except:
                continue
    return points3D

def write_ply(points, output_path):
    with open(output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt in points:
            f.write("{} {} {} {} {} {}\n".format(*pt))

# === À personnaliser si besoin ===
input_path = r"D:\malo\Documents\cours_tsp\cv\gaussian_splatting\colmap_project\sparse\0\points3D.bin"
output_path = r"D:\malo\Documents\cours_tsp\cv\gaussian_splatting\exports\splat_fixed.ply"

points = read_points3D_bin(input_path)
print(f"{len(points)} points valides trouvés.")
write_ply(points, output_path)
print("Export terminé : ", output_path)
