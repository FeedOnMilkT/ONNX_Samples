import os
import shutil
from scipy.io import loadmat

# 你的目录路径
val_img_dir = "/root/autodl-tmp/imagenet/val"  # 你解压后存放图像的目录
val_gt_file = "/root/autodl-tmp/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
meta_mat_file = "/root/autodl-tmp/ILSVRC2012_devkit_t12/data/meta.mat"
output_dir = "/root/autodl-tmp/imagenet/val_sorted"

# 加载 ground truth 标签（每行是第 i 张图的类索引）
with open(val_gt_file) as f:
    labels = [int(line.strip()) for line in f]

# 加载 meta.mat 获取类别 ID → WordNet ID 映射
meta = loadmat(meta_mat_file)
synsets = meta["synsets"]
wnid_list = [str(synsets[i][0][1][0]) for i in range(1000)]

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 重排图像
for i, label in enumerate(labels):
    wnid = wnid_list[label - 1]  # 标签从1开始
    target_dir = os.path.join(output_dir, wnid)
    os.makedirs(target_dir, exist_ok=True)

    filename = f"ILSVRC2012_val_{i+1:08d}.JPEG"
    src = os.path.join(val_img_dir, filename)
    dst = os.path.join(target_dir, filename)

    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"⚠️ Warning: {src} not found!")

print(f"✅ Done! Reorganized validation set is in: {output_dir}")
