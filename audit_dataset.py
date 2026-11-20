
"""VisionPort Session 6: COCO Dataset Audit and Visualisation.

Run:
    python audit_dataset.py

Audit Session 5 data:
    python audit_dataset.py --ann data/idd/annotations/instances.json --images data/idd/images --out docs/s5_audit
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# -------------------------------------------------
# Audit thresholds
# -------------------------------------------------

MIN_BOX_SIDE = 10
DARK_MEAN = 50
BLUR_VAR = 100
RARE_SHARE = 0.05
AREA_TOLERANCE = 0.05


# -------------------------------------------------
# Command-line options
# -------------------------------------------------

parser = argparse.ArgumentParser(
    description="Audit a COCO-format dataset"
)

parser.add_argument(
    "--ann",
    default="data/audit/annotations/instances.json"
)

parser.add_argument(
    "--images",
    default="data/audit/images"
)

parser.add_argument(
    "--out",
    default="docs"
)

args = parser.parse_args()

ann_path = Path(args.ann)
image_dir = Path(args.images)
output_dir = Path(args.out)

if not ann_path.exists():
    raise SystemExit(
        f"Annotation file not found: {ann_path}"
    )

output_dir.mkdir(parents=True, exist_ok=True)

report = []


# Print and save report lines together
def say(line=""):
    print(line)
    report.append(line)


# -------------------------------------------------
# Load COCO dataset
# -------------------------------------------------

with open(ann_path, "r", encoding="utf-8") as file:
    coco = json.load(file)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

category_names = {
    category["id"]: category["name"]
    for category in categories
}

image_by_id = {
    image["id"]: image
    for image in images
}


say("# VisionPort Dataset Audit")
say()
say(f"Annotation file : {ann_path}")
say(f"Image folder    : {image_dir}")


# -------------------------------------------------
# 1. Dataset overview
# -------------------------------------------------

say("\n## 1. Overview")

say(
    f"Images: {len(images)} | "
    f"Annotations: {len(annotations)} | "
    f"Categories: {len(categories)}"
)

annotations_by_image = defaultdict(list)

for annotation in annotations:
    annotations_by_image[
        annotation["image_id"]
    ].append(annotation)

images_without_annotations = [
    image["id"]
    for image in images
    if not annotations_by_image.get(image["id"])
]

say(
    f"Images with no annotation: "
    f"{len(images_without_annotations)} "
    f"{images_without_annotations[:10]}"
)

image_sizes = Counter(
    (image["width"], image["height"])
    for image in images
)

size_summary = ", ".join(
    f"{width}x{height} ({count})"
    for (width, height), count
    in image_sizes.most_common()
)

say(f"Image sizes in JSON: {size_summary}")


# -------------------------------------------------
# 2. Class distribution
# -------------------------------------------------

say("\n## 2. Class Distribution")

class_counts = Counter(
    annotation["category_id"]
    for annotation in annotations
    if annotation["category_id"] in category_names
)

total_valid_annotations = sum(class_counts.values())

say(
    f"{'ID':>3} {'Class':<14}"
    f"{'Count':>6} {'Share':>8}"
)

for category_id, name in sorted(category_names.items()):

    count = class_counts.get(category_id, 0)

    share = (
        count / total_valid_annotations
        if total_valid_annotations
        else 0
    )

    bar = "#" * round(share * 50)

    say(
        f"{category_id:>3} "
        f"{name:<14}"
        f"{count:>6} "
        f"{share:>7.1%} "
        f"{bar}"
    )

if class_counts:

    largest = max(class_counts.values())
    smallest = min(class_counts.values())

    if smallest > 0:
        imbalance_ratio = largest / smallest
        say(
            f"Imbalance ratio "
            f"(largest / smallest): "
            f"{imbalance_ratio:.1f} : 1"
        )

rare_classes = [
    category_names[category_id]
    for category_id in category_names
    if (
        class_counts.get(category_id, 0)
        / max(total_valid_annotations, 1)
    ) < RARE_SHARE
]

say(
    f"Rare classes (under {RARE_SHARE:.0%}): "
    f"{rare_classes if rare_classes else 'none'}"
)


# -------------------------------------------------
# 3. Annotation validation
# -------------------------------------------------

say("\n## 3. Annotation Validation")

issues = defaultdict(list)

annotation_id_counts = Counter(
    annotation["id"]
    for annotation in annotations
)

box_counts = Counter(
    (
        annotation["image_id"],
        annotation["category_id"],
        tuple(annotation.get("bbox", []))
    )
    for annotation in annotations
)

for annotation in annotations:

    annotation_id = annotation["id"]
    box = annotation.get("bbox", [])

    if annotation_id_counts[annotation_id] > 1:
        issues["duplicate annotation id"].append(
            annotation_id
        )

    if annotation["image_id"] not in image_by_id:
        issues["points to a missing image"].append(
            annotation_id
        )

    if annotation["category_id"] not in category_names:
        issues["points to an unknown category"].append(
            annotation_id
        )

    if len(box) != 4:
        issues["bbox does not have 4 numbers"].append(
            annotation_id
        )
        continue

    x, y, width, height = box

    # Zero or negative box dimensions
    if width <= 0 or height <= 0:

        issues["zero or negative width/height"].append(
            annotation_id
        )

    # Tiny boxes
    elif min(width, height) < MIN_BOX_SIDE:

        issues[
            f"tiny box (side under {MIN_BOX_SIDE} px)"
        ].append(annotation_id)

    # Check image boundaries
    if annotation["image_id"] in image_by_id:

        image = image_by_id[annotation["image_id"]]

        if (
            x < 0
            or y < 0
            or x + width > image["width"]
            or y + height > image["height"]
        ):
            issues["box outside the image"].append(
                annotation_id
            )

    # Check stored area
    if width > 0 and height > 0:

        actual_area = width * height
        stored_area = annotation.get("area", actual_area)

        if (
            abs(stored_area - actual_area)
            > AREA_TOLERANCE * actual_area
        ):
            issues[
                "area differs from width x height"
            ].append(annotation_id)

    # Check duplicate boxes
    if box_counts[
        (
            annotation["image_id"],
            annotation["category_id"],
            tuple(box)
        )
    ] > 1:

        issues["duplicate box"].append(annotation_id)


if not issues:
    say("No annotation problems found.")

for problem, annotation_ids in sorted(issues.items()):

    say(
        f"- {problem}: {len(annotation_ids)} "
        f"e.g. annotation ids {annotation_ids[:5]}"
    )


# -------------------------------------------------
# 4. Image file and quality validation
# -------------------------------------------------

say("\n## 4. Image Files and Quality")

missing_images = []
corrupt_images = []
wrong_size_images = []
dark_images = []
blurry_images = []

quality_scores = []

for image in images:

    image_path = image_dir / image["file_name"]

    # Check missing file
    if not image_path.exists():

        missing_images.append(image["id"])
        continue

    # Check whether the image opens
    try:

        with Image.open(image_path) as picture:
            picture = picture.convert("RGB")

    except Exception:

        corrupt_images.append(image["id"])
        continue

    # Check actual dimensions
    if picture.size != (
        image["width"],
        image["height"]
    ):

        wrong_size_images.append(image["id"])

    # Convert image to grayscale
    grey = np.asarray(
        picture.convert("L"),
        dtype=float
    )

    # Brightness score
    brightness = grey.mean()

    # Laplacian sharpness score
    laplacian = (
        4 * grey[1:-1, 1:-1]
        - grey[:-2, 1:-1]
        - grey[2:, 1:-1]
        - grey[1:-1, :-2]
        - grey[1:-1, 2:]
    )

    sharpness = laplacian.var()

    quality_scores.append(
        (image["id"], brightness, sharpness)
    )

    if brightness < DARK_MEAN:
        dark_images.append(image["id"])

    if sharpness < BLUR_VAR:
        blurry_images.append(image["id"])


image_findings = [
    ("file missing", missing_images),
    ("file cannot be opened (corrupt)", corrupt_images),
    ("size differs from the JSON", wrong_size_images),
    (
        f"too dark (brightness under {DARK_MEAN})",
        dark_images
    ),
    (
        f"blurry (sharpness under {BLUR_VAR})",
        blurry_images
    )
]

for label, ids in image_findings:

    say(
        f"- {label}: {len(ids)} "
        f"e.g. image ids {ids[:8]}"
    )


# -------------------------------------------------
# 5. Object size classification
# -------------------------------------------------

say("\n## 5. Object Size Mix")

small = 0
medium = 0
large = 0

for annotation in annotations:

    box = annotation.get("bbox", [])

    if (
        len(box) == 4
        and box[2] > 0
        and box[3] > 0
    ):

        area = box[2] * box[3]

        if area < 32 * 32:
            small += 1

        elif area < 96 * 96:
            medium += 1

        else:
            large += 1

say(
    f"Small (under 32x32): {small} | "
    f"Medium (under 96x96): {medium} | "
    f"Large: {large}"
)


# -------------------------------------------------
# 6. Generate visualisation charts
# -------------------------------------------------

figure, axes = plt.subplots(
    1, 3,
    figsize=(15, 4)
)

# Chart 1: class distribution
names = [
    category_names[cid]
    for cid in sorted(category_names)
]

counts = [
    class_counts.get(cid, 0)
    for cid in sorted(category_names)
]

axes[0].bar(names, counts, color="#378ADD")
axes[0].set_title("Annotations per Class")
axes[0].tick_params(
    axis="x",
    rotation=30
)

# Chart 2: object size distribution
axes[1].bar(
    ["Small", "Medium", "Large"],
    [small, medium, large],
    color="#1D9E75"
)

axes[1].set_title("Object Size Mix")

# Chart 3: image quality
if quality_scores:

    ids, brightness_values, sharpness_values = zip(
        *quality_scores
    )

    flagged = [
        image_id in dark_images
        or image_id in blurry_images
        for image_id in ids
    ]

    point_colours = [
        "#E24B4A" if is_flagged else "#378ADD"
        for is_flagged in flagged
    ]

    axes[2].scatter(
        brightness_values,
        np.maximum(sharpness_values, 0.1),
        c=point_colours,
        s=18
    )

    axes[2].axvline(
        DARK_MEAN,
        color="grey",
        linestyle="--"
    )

    axes[2].axhline(
        BLUR_VAR,
        color="grey",
        linestyle="--"
    )

    axes[2].set_yscale("log")

axes[2].set_xlabel("Brightness")
axes[2].set_ylabel("Sharpness (log)")
axes[2].set_title("Image Quality (red = flagged)")

plt.tight_layout()

chart_path = output_dir / "audit_charts.png"

plt.savefig(
    chart_path,
    dpi=150
)

plt.close(figure)


# -------------------------------------------------
# 7. Final audit summary
# -------------------------------------------------

say("\n## 6. Summary")

annotations_with_problems = len({
    annotation_id
    for annotation_ids in issues.values()
    for annotation_id in annotation_ids
})

images_with_problems = len(
    set(missing_images)
    | set(corrupt_images)
    | set(wrong_size_images)
    | set(dark_images)
    | set(blurry_images)
)

say(
    f"Annotations with at least one problem: "
    f"{annotations_with_problems} of {len(annotations)}"
)

say(
    f"Images with at least one problem: "
    f"{images_with_problems} of {len(images)}"
)

say(
    "Suggested actions: repair or remove flagged annotations, "
    "replace missing or corrupt images, review dark and blurry "
    "images by eye, and collect more examples of rare classes."
)


# -------------------------------------------------
# Save Markdown report
# -------------------------------------------------

report_path = output_dir / "audit_report.md"

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nAUDIT COMPLETED SUCCESSFULLY")
print("----------------------------")
print(f"Report : {report_path}")
print(f"Chart  : {chart_path}")
