
"""Create the synthetic practice dataset for VisionPort Session 6.

Run from the project root:
    python make_audit_dataset.py

Creates:
    data/audit/images/
    data/audit/annotations/instances.json

Does not modify the Session 5 data in data/idd/.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# Reproducible random data
random.seed(7)
rng = np.random.default_rng(7)

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "data" / "audit" / "images"
ANNOTATION_DIR = ROOT / "data" / "audit" / "annotations"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# Image dimensions
WIDTH = 640
HEIGHT = 360

# Six traffic categories
categories = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "motorcycle"},
    {"id": 3, "name": "pedestrian"},
    {"id": 4, "name": "bus"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "autorickshaw"},
]

# Rectangle colours for synthetic objects
colours = {
    1: (60, 60, 200),
    2: (60, 200, 60),
    3: (200, 150, 60),
    4: (200, 60, 60),
    5: (120, 120, 120),
    6: (230, 200, 40),
}

# Images that will contain deliberate problems
DARK = {10, 11, 12}
BLURRY = {20, 21, 22, 23}
CORRUPT = {30, 31}
MISSING = {40, 41}
SIZE_MISMATCH = {50}
NO_OBJECTS = {90, 91, 92, 93, 94}

# Only five images contain autorickshaws
autorickshaw_images = set(
    random.sample(range(1, 90), 5)
)

images = []
annotations = []
annotation_id = 1

# Weighted selection of common classes
common_classes = (
    [1] * 45
    + [2] * 25
    + [3] * 15
    + [4] * 6
    + [5] * 9
)


# -------------------------------------------------
# Generate 100 synthetic images
# -------------------------------------------------

for image_id in range(1, 101):

    filename = f"img_{image_id:03d}.jpg"

    images.append({
        "id": image_id,
        "file_name": filename,
        "width": WIDTH,
        "height": HEIGHT
    })

    # Textured background
    pixels = np.clip(
        rng.normal(150, 25, (HEIGHT, WIDTH, 3)),
        0,
        255
    )

    # Empty-scene images have no objects
    number_of_objects = (
        0 if image_id in NO_OBJECTS
        else random.randint(2, 6)
    )

    selected_classes = [
        random.choice(common_classes)
        for _ in range(number_of_objects)
    ]

    if image_id in autorickshaw_images and selected_classes:
        selected_classes[0] = 6

    # Generate object rectangles and annotations
    for category_id in selected_classes:

        box_width = random.randint(40, 140)
        box_height = random.randint(40, 120)

        x = random.randint(0, WIDTH - box_width)
        y = random.randint(0, HEIGHT - box_height)

        base_colour = np.array(
            colours[category_id],
            dtype=float
        )

        pixels[
            y:y + box_height,
            x:x + box_width
        ] = np.clip(
            base_colour
            + rng.normal(0, 18, (box_height, box_width, 3)),
            0,
            255
        )

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, box_width, box_height],
            "area": box_width * box_height,
            "iscrowd": 0
        })

        annotation_id += 1

    picture = Image.fromarray(
        pixels.astype("uint8")
    )

    # Deliberately dark images
    if image_id in DARK:
        picture = Image.fromarray(
            (np.asarray(picture) * 0.2).astype("uint8")
        )

    # Deliberately blurry images
    if image_id in BLURRY:
        picture = picture.filter(
            ImageFilter.GaussianBlur(radius=6)
        )

    # Image dimensions differ from JSON
    if image_id in SIZE_MISMATCH:
        picture = picture.resize(
            (WIDTH // 2, HEIGHT // 2)
        )

    # Missing images: JSON record exists, file is not saved
    if image_id in MISSING:
        continue

    # Corrupt images: invalid JPEG data
    if image_id in CORRUPT:
        (IMAGE_DIR / filename).write_bytes(
            b"this is not a jpeg" * 20
        )
        continue

    picture.save(
        IMAGE_DIR / filename,
        quality=92
    )


# -------------------------------------------------
# Add deliberate annotation problems
# -------------------------------------------------

def add_annotation(image_id, category_id, bbox, area=None):
    global annotation_id

    annotations.append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": (
            area if area is not None
            else max(bbox[2], 0) * max(bbox[3], 0)
        ),
        "iscrowd": 0
    })

    annotation_id += 1


# Zero width
add_annotation(60, 1, [100, 100, 0, 50])

# Zero height
add_annotation(61, 1, [200, 100, 60, 0])

# Box outside right and bottom edges
add_annotation(62, 1, [600, 300, 120, 100])

# Box starts outside left edge
add_annotation(63, 2, [-20, 50, 80, 80])

# Box outside bottom edge
add_annotation(64, 3, [300, 330, 50, 60])

# Tiny boxes
add_annotation(65, 3, [100, 100, 4, 5])
add_annotation(66, 2, [300, 200, 6, 6])

# Duplicate an existing box on image 67
first_67 = next(
    a for a in annotations
    if a["image_id"] == 67
)

add_annotation(
    67,
    first_67["category_id"],
    list(first_67["bbox"])
)

# Unknown category
add_annotation(68, 99, [50, 50, 80, 60])

# Unknown image
add_annotation(999, 1, [50, 50, 80, 60])

# Incorrect area
add_annotation(
    69,
    1,
    [120, 120, 80, 60],
    area=9000
)


# -------------------------------------------------
# Save COCO JSON
# -------------------------------------------------

dataset = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

json_path = ANNOTATION_DIR / "instances.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(dataset, file, indent=2)


print("\nVISIONPORT SESSION 6 DATASET CREATED")
print("------------------------------------")
print(f"Images       : {len(images)}")
print(f"Annotations  : {len(annotations)}")
print(f"Categories   : {len(categories)}")
print(f"JSON file    : {json_path}")
print("\nSynthetic practice data with deliberate defects.")
