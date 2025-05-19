from read_write_model import read_points3D_binary

input_path = "D:/malo/Documents/cours_tsp/cv/gaussian_splatting/colmap_project/sparse/0/points3D.bin"
output_path = "D:/malo/Documents/cours_tsp/cv/gaussian_splatting/exports/colmap_points.ply"

points3D = read_points3D_binary(input_path)

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

    for point in points3D.values():
        x, y, z = point.xyz
        r, g, b = point.rgb
        f.write(f"{x} {y} {z} {r} {g} {b}\n")
