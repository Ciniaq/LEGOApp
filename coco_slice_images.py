from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="coco_dataset.json",
    image_dir="dataset/images/train",
    output_coco_annotation_file_name="mini_coco_dataset.json",
    ignore_negative_samples=True,
    output_dir="mini_dataset/images/train",
    slice_height=416,
    slice_width=416,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.9,
    verbose=True
)

print(f"Sliced COCO dataset saved")
