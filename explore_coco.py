
"""
VisionPort - Session 5
COCO Dataset Exploration

Implementation A: Read and explore COCO JSON
Implementation B: Visualise bounding boxes
Implementation C: Count objects and verify annotations

Run from project root:
python explore_coco.py
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image, ImageDraw


# ============================================================
# PROJECT PATHS
# ============================================================

project_dir = Path(__file__).resolve().parent

json_path = (
    project_dir
    / "data"
    / "idd"
    / "annotations"
    / "instances.json"
)

image_dir = project_dir / "data" / "idd" / "images"

output_dir = project_dir / "docs"
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPLEMENTATION A - READ AND EXPLORE COCO JSON
# ============================================================

print("\n========================================")
print("VISIONPORT - SESSION 5")
print("========================================")

# Check annotation file
if not json_path.exists():
    raise FileNotFoundError(
        f"COCO annotation file not found: {json_path}"
    )

# Load COCO JSON
with open(json_path, "r", encoding="utf-8") as file:
    coco = json.load(file)

print("\nCOCO JSON loaded successfully!")

# Read three main sections
images = coco["images"]
categories = coco["categories"]
annotations = coco["annotations"]

# Dataset summary
print("\nDATASET SUMMARY")
print("------------------------------")
print("Images:", len(images))
print("Categories:", len(categories))
print("Annotations:", len(annotations))

# Display categories
print("\nAVAILABLE CATEGORIES")
print("------------------------------")

for category in categories:
    print(category["id"], ":", category["name"])

# Display image information
print("\nIMAGE INFORMATION")
print("------------------------------")

for image_info in images:
    print(
        "Image ID:", image_info["id"],
        "| File:", image_info["file_name"],
        "| Width:", image_info["width"],
        "| Height:", image_info["height"]
    )

# Display annotations
print("\nOBJECT ANNOTATIONS")
print("------------------------------")

for annotation in annotations:
    print(
        "Annotation ID:", annotation["id"],
        "| Image ID:", annotation["image_id"],
        "| Category ID:", annotation["category_id"],
        "| Bounding Box:", annotation["bbox"]
    )


# ============================================================
# IMPLEMENTATION B - VISUALISE BOUNDING BOXES
# ============================================================

print("\n========================================")
print("BOUNDING BOX VISUALIZATION")
print("========================================")

# Convert category IDs into category names
category_names = {
    category["id"]: category["name"]
    for category in categories
}

# Group annotations by image ID
annotations_by_image = defaultdict(list)

for annotation in annotations:
    annotations_by_image[annotation["image_id"]].append(annotation)

# Generate preview for every image
generated_previews = []

for image_info in images:

    image_id = image_info["id"]
    filename = image_info["file_name"]

    image_path = image_dir / filename

    print(f"\nProcessing Image ID: {image_id}")
    print("Image:", filename)

    # Check image exists
    if not image_path.exists():
        print("ERROR: Image file not found:", image_path)
        continue

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Check dimensions
    if image.size != (
        image_info["width"],
        image_info["height"]
    ):
        print("WARNING: Image dimensions differ from COCO JSON.")

    draw = ImageDraw.Draw(image)

    image_annotations = annotations_by_image.get(
        image_id, []
    )

    if not image_annotations:
        print("No annotations found for this image.")

    # Draw bounding boxes
    for annotation in image_annotations:

        # COCO format: x, y, width, height
        x, y, width, height = annotation["bbox"]

        # Convert COCO format to XYXY
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        # Get category label
        label = category_names.get(
            annotation["category_id"],
            "Unknown"
        )

        # Draw rectangle
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        # Label position
        label_position = (
            x1,
            max(0, y1 - 18)
        )

        # Draw label background
        text_box = draw.textbbox(
            label_position,
            label
        )

        draw.rectangle(
            text_box,
            fill="red"
        )

        # Draw category name
        draw.text(
            label_position,
            label,
            fill="white"
        )

        print(
            f"  Detected annotation: {label} "
            f"| BBox: {[x1, y1, x2, y2]}"
        )

    # Save preview
    output_path = (
        output_dir / f"coco_preview_{image_id}.jpg"
    )

    image.save(output_path, quality=95)

    generated_previews.append(output_path)

    print("Preview saved successfully:")
    print(output_path)


# ============================================================
# IMPLEMENTATION C - COUNT OBJECTS PER CATEGORY
# ============================================================

print("\n========================================")
print("OBJECTS PER CATEGORY")
print("========================================")

category_counts = Counter(
    annotation["category_id"]
    for annotation in annotations
)

for category_id, name in category_names.items():

    count = category_counts.get(
        category_id, 0
    )

    print(
        f"{category_id:>3} : {name:<12} {count}"
    )


# ============================================================
# IMAGES WITH AND WITHOUT ANNOTATIONS
# ============================================================

images_with_annotations = sum(
    1 for image_info in images
    if annotations_by_image.get(image_info["id"])
)

images_without_annotations = (
    len(images) - images_with_annotations
)

print("\nIMAGE ANNOTATION SUMMARY")
print("------------------------------")

print(
    "Images with annotations:",
    images_with_annotations
)

print(
    "Images without annotations:",
    images_without_annotations
)


# ============================================================
# IMPLEMENTATION C - VERIFICATION CHECKS
# ============================================================

print("\n========================================")
print("VERIFICATION CHECKS")
print("========================================")

# Create image lookup dictionary
image_by_id = {
    image_info["id"]: image_info
    for image_info in images
}

# Check 1: Invalid image references
bad_image = [
    a["id"]
    for a in annotations
    if a["image_id"] not in image_by_id
]

# Check 2: Invalid category references
bad_category = [
    a["id"]
    for a in annotations
    if a["category_id"] not in category_names
]

# Check 3: Invalid bounding box dimensions
bad_size = [
    a["id"]
    for a in annotations
    if (
        len(a["bbox"]) != 4
        or a["bbox"][2] <= 0
        or a["bbox"][3] <= 0
    )
]

# Check 4: Bounding boxes outside image boundaries
outside = []

for annotation in annotations:

    image_id = annotation["image_id"]

    if image_id not in image_by_id:
        continue

    bbox = annotation["bbox"]

    if len(bbox) != 4:
        continue

    x, y, width, height = bbox

    image_info = image_by_id[image_id]

    if (
        x < 0
        or y < 0
        or x + width > image_info["width"]
        or y + height > image_info["height"]
    ):
        outside.append(annotation["id"])


# Display results
checks = [
    (
        "Every annotation points to an existing image",
        bad_image
    ),
    (
        "Every annotation points to an existing category",
        bad_category
    ),
    (
        "Every box has width > 0 and height > 0",
        bad_size
    ),
    (
        "Every box lies inside its image",
        outside
    )
]

all_passed = True

for check_name, bad_ids in checks:

    if not bad_ids:
        result = "PASS"
    else:
        result = f"FAIL {bad_ids[:5]}"
        all_passed = False

    print(
        f"{check_name:<50} {result}"
    )


# ============================================================
# FINAL SESSION 5 REPORT
# ============================================================

print("\n========================================")
print("SESSION 5 EXECUTION SUMMARY")
print("========================================")

print("Total images:", len(images))
print("Total categories:", len(categories))
print("Total annotations:", len(annotations))

print(
    "Generated preview images:",
    len(generated_previews)
)

print("Preview directory:", output_dir.resolve())

print(
    "Annotation verification:",
    "ALL PASSED" if all_passed else "SOME CHECKS FAILED"
)

print("\nGenerated Preview Files:")

for preview in generated_previews:
    print(preview.name)

print("\nVISIONPORT SESSION 5 COMPLETED")
print("========================================")
