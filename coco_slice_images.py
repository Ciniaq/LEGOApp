from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="coco_dataset.json",
    image_dir="dataset/images/all",
    output_coco_annotation_file_name="mini_coco_dataset.json",
    ignore_negative_samples=True,
    output_dir="medium_dataset/images/train",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.9,
    verbose=True
)

print(f"Sliced COCO dataset saved")
