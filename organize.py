
import os
import shutil

VAL_DIR = "/cs/cs153/projects/olivia-elsa/cv_final_project/ImageNet_val_final/ILSVRC2012_img_val"
GROUND_TRUTH_FILE = "/cs/cs153/projects/olivia-elsa/cv_final_project/ILSVRC2015_clsloc_validation_ground_truth.txt"
MAP_FILE = "/cs/cs153/projects/olivia-elsa/cv_final_project/map_clsloc.txt"
OUTPUT_DIR = "/cs/cs153/projects/olivia-elsa/cv_final_project/ImageNet_final_val_sorted"

class_to_wnid = {}

with open(MAP_FILE, "r") as f:
    for line in f:
        wnid, class_id, _ = line.strip().split()
        class_id = int(class_id)
        class_to_wnid[class_id] = wnid

gt_labels = []

with open(GROUND_TRUTH_FILE, "r") as f:
    for line in f:
        gt_labels.append(int(line.strip()))

os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, class_id in enumerate(gt_labels, start=1):

    img_name = f"ILSVRC2012_val_{idx:08d}.JPEG"
    src_path = os.path.join(VAL_DIR, img_name)

    if not os.path.exists(src_path):
        continue

    wnid = class_to_wnid[class_id]

    wnid_dir = os.path.join(OUTPUT_DIR, wnid)
    os.makedirs(wnid_dir, exist_ok=True)

    dst_path = os.path.join(wnid_dir, img_name)
    shutil.move(src_path, dst_path)

    if idx % 5000 == 0:
        print(f"Processed {idx} images")

