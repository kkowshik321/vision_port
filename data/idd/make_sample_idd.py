import json
from pathlib import Path

from PIL import Image, ImageDraw


# 1. Project directories
root = Path(__file__).resolve().parent

image_dir = root / "images"
ann_dir = root / "annotations"

image_dir.mkdir(parents=True, exist_ok=True)
ann_dir.mkdir(parents=True, exist_ok=True)


# 2. Categories
categories = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 3, "name": "pedestrian", "supercategory": "person"},
    {"id": 4, "name": "bus", "supercategory": "vehicle"},
    {"id": 5, "name": "truck", "supercategory": "vehicle"}
]


# 3. Image information
images = [
    {"id": 1, "file_name": "traffic_001.jpg", "width": 1280, "height": 720},
    {"id": 2, "file_name": "traffic_002.jpg", "width": 1280, "height": 720},
    {"id": 3, "file_name": "traffic_003.jpg", "width": 1280, "height": 720}
]


# 4. Object annotations
annotations = [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [100, 180, 260, 150], "area": 39000, "iscrowd": 0},

    {"id": 2, "image_id": 1, "category_id": 3,
     "bbox": [500, 200, 60, 160], "area": 9600, "iscrowd": 0},

    {"id": 3, "image_id": 1, "category_id": 2,
     "bbox": [700, 300, 120, 140], "area": 16800, "iscrowd": 0},

    {"id": 4, "image_id": 2, "category_id": 4,
     "bbox": [200, 150, 400, 300], "area": 120000, "iscrowd": 0},

    {"id": 5, "image_id": 2, "category_id": 5,
     "bbox": [800, 250, 300, 200], "area": 60000, "iscrowd": 0},

    {"id": 6, "image_id": 3, "category_id": 1,
     "bbox": [100, 180, 260, 150], "area": 39000, "iscrowd": 0},

    {"id": 7, "image_id": 3, "category_id": 2,
     "bbox": [700, 300, 120, 140], "area": 16800, "iscrowd": 0},

    {"id": 8, "image_id": 3, "category_id": 4,
     "bbox": [200, 150, 400, 300], "area": 120000, "iscrowd": 0}
]


# 5. Category colors
fill = {
    1: (120, 120, 200),
    2: (120, 200, 120),
    3: (200, 160, 100),
    4: (200, 120, 120),
    5: (150, 150, 150)
}


# 6. Generate sample traffic images
for info in images:

    canvas = Image.new(
        "RGB",
        (info["width"], info["height"]),
        (225, 225, 225)
    )

    draw = ImageDraw.Draw(canvas)

    for annotation in annotations:

        if annotation["image_id"] == info["id"]:

            x, y, w, h = annotation["bbox"]

            draw.rectangle(
                [x, y, x + w, y + h],
                fill=fill[annotation["category_id"]],
                outline=(0, 0, 0),
                width=3
            )

    canvas.save(image_dir / info["file_name"])

    print("Created:", info["file_name"])


# 7. Save COCO annotations
dataset = {
    "images": images,
    "categories": categories,
    "annotations": annotations
}

with open(
    ann_dir / "instances.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(dataset, f, indent=2)


# 8. Completion message
print("\n================================")
print("VISIONPORT DATASET CREATED")
print("================================")

print("Images:", len(images))
print("Categories:", len(categories))
print("Annotations:", len(annotations))

print("\nImage directory:", image_dir)
print("Annotation file:", ann_dir / "instances.json")

print("\nReminder: This is synthetic teaching data,")
print("not the official IDD dataset.")