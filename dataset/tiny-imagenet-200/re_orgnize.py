import os
import shutil

val_dir = '/root/tiny-imagenet-200/val'
img_dir = os.path.join(val_dir, 'images')
ann_file = os.path.join(val_dir, 'val_annotations.txt')

# 创建类别文件夹
with open(ann_file, 'r') as f:
    for line in f:
        img, cls = line.split('\t')[:2]
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        shutil.move(os.path.join(img_dir, img), os.path.join(cls_dir, img))

# 可选：删除空的 images 文件夹
shutil.rmtree(img_dir)