
"""
VisionPort Session 7: Model Artefact Lifecycle Inspector

Inspects:
1. Project artefact inventory
2. PyTorch checkpoint (.pt)
3. Model architecture and weights
4. ONNX graph (.onnx)
5. TensorRT engine (.engine / .plan)
6. Artefact comparison

Run from the VisionPort project root:

    python inspect_artefacts.py

Optional ONNX export:

    python inspect_artefacts.py --export-onnx

Inspect an existing ONNX file:

    python inspect_artefacts.py --onnx yolov8n.onnx
"""

import argparse
import os
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------
# 0. COMMAND-LINE OPTIONS AND ARTEFACT ROLES
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Inspect VisionPort model artefacts"
)

parser.add_argument(
    "--weights",
    default="yolov8n.pt",
    help="PyTorch checkpoint to inspect"
)

parser.add_argument(
    "--onnx",
    default=None,
    help="Path of an existing ONNX file to inspect"
)

parser.add_argument(
    "--export-onnx",
    action="store_true",
    help="Export the checkpoint to ONNX before inspecting it"
)

args = parser.parse_args()


ROLE = {
    ".pt": (
        "PyTorch checkpoint: trained weights + architecture. "
        "Used for training and export."
    ),
    ".onnx": (
        "ONNX graph: framework-neutral model file. "
        "Used by ONNX Runtime and TensorRT."
    ),
    ".engine": (
        "TensorRT engine: compiled for one GPU and precision. "
        "Used for fast serving."
    ),
    ".plan": (
        "TensorRT engine (Triton naming). Used for fast serving."
    ),
}


# ---------------------------------------------------------
# 1. INVENTORY OF ARTEFACTS IN THE PROJECT
# ---------------------------------------------------------

print("\n=== 1. ARTEFACT INVENTORY ===")

found = []

for folder, subfolders, files in os.walk("."):

    # Skip virtual environments and Git folders
    subfolders[:] = [
        folder_name
        for folder_name in subfolders
        if folder_name not in (
            ".venv",
            "venv",
            ".git",
            "__pycache__",
            ".idea"
        )
    ]

    for name in files:

        if Path(name).suffix.lower() in ROLE:

            found.append(
                Path(folder) / name
            )


if not found:
    print(
        "No .pt, .onnx, .engine or .plan files "
        "found in the project folder yet."
    )

for path in found:

    size_mb = path.stat().st_size / 1e6

    print(
        f"{str(path):<40} "
        f"{size_mb:>8.1f} MB   "
        f"{ROLE[path.suffix.lower()]}"
    )


# ---------------------------------------------------------
# 2. PYTORCH CHECKPOINT INSPECTION
# ---------------------------------------------------------

print(
    f"\n=== 2. PYTORCH CHECKPOINT: {args.weights} ==="
)

yolo = None
model = None

try:

    import torch
    from ultralytics import YOLO

    # Load the YOLO model.
    # Ultralytics downloads the official weights if needed.
    yolo = YOLO(args.weights)

    weights_path = Path(args.weights)

    # Only load trusted checkpoint files.
    # weights_only=False is needed because this checkpoint
    # contains Python objects in addition to tensors.
    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False
    )

    # 2a. Contents of the checkpoint
    print("\nTop-level contents of the checkpoint file:")

    for key, value in checkpoint.items():

        kind = type(value).__name__

        detail = (
            f"{len(value)} entries"
            if isinstance(value, dict)
            else ""
        )

        print(
            f" {key:<14} {kind:<16} {detail}"
        )

    print(
        "Training state (optimizer) present:",
        "yes" if checkpoint.get("optimizer")
        else "no (stripped, for inference)"
    )

    stored_model = (
        checkpoint.get("ema")
        or checkpoint.get("model")
    )

    if stored_model is not None:

        print(
            "Weights stored in the file as:",
            next(stored_model.parameters()).dtype
        )

    # 2b. Architecture
    model = yolo.model

    print(
        f"\nArchitecture: {type(model).__name__}, "
        f"{len(model.model)} top-level layers"
    )

    for index in (
        0,
        1,
        2,
        len(model.model) - 1
    ):

        layer = model.model[index]

        parameter_count = sum(
            parameter.numel()
            for parameter in layer.parameters()
        )

        print(
            f" layer {index:>2}: "
            f"{type(layer).__name__:<10} "
            f"{parameter_count:>10,} parameters"
        )

    # 2c. Weights
    state = model.state_dict()

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    first_parameter = next(
        model.parameters()
    )

    print(
        f"\nWeights: {len(state)} named tensors, "
        f"{number_of_parameters:,} parameters, "
        f"loaded as {first_parameter.dtype}"
    )

    memory_mb = (
        number_of_parameters
        * first_parameter.element_size()
        / 1e6
    )

    print(
        f"Approx. weight memory: {memory_mb:.1f} MB"
    )

    for name in list(state)[:3]:

        print(
            f" {name:<36} "
            f"shape {tuple(state[name].shape)}"
        )

    print(
        "Classes the model predicts:",
        len(model.names),
        "e.g.",
        list(model.names.values())[:5]
    )

except Exception as error:

    print(
        "Could not inspect the checkpoint:",
        type(error).__name__,
        error
    )

    print(
        "Check that torch and ultralytics are installed "
        "and that the weights file exists."
    )


# ---------------------------------------------------------
# 3. ONNX GRAPH INSPECTION
# ---------------------------------------------------------

onnx_path = args.onnx

# Export checkpoint to ONNX when requested
if args.export_onnx:

    print(
        "\nExporting to ONNX "
        "(this may install additional packages)..."
    )

    if yolo is None:

        print(
            "Export skipped because the PyTorch "
            "checkpoint could not be loaded."
        )

    else:

        try:

            onnx_path = str(
                yolo.export(
                    format="onnx",
                    imgsz=640
                )
            )

            print(
                "ONNX export completed:",
                onnx_path
            )

        except Exception as error:

            print(
                "Export failed:",
                type(error).__name__,
                error
            )


# If no ONNX path was supplied, look beside the checkpoint
if onnx_path is None:

    guessed_path = Path(
        args.weights
    ).with_suffix(".onnx")

    if guessed_path.exists():
        onnx_path = str(guessed_path)


print("\n=== 3. ONNX GRAPH ===")

if onnx_path is None:

    print(
        "No .onnx file yet. Create one with:"
    )

    print(
        "python inspect_artefacts.py --export-onnx"
    )

elif not Path(onnx_path).exists():

    print(
        f"ONNX file does not exist: {onnx_path}"
    )

else:

    try:

        import onnx

        model_proto = onnx.load(onnx_path)

        # Validate the ONNX graph
        onnx.checker.check_model(model_proto)

        print(
            f"File           : {onnx_path} "
            f"({Path(onnx_path).stat().st_size / 1e6:.1f} MB)"
        )

        print("Format check   : passed")

        print(
            "Opset version  :",
            model_proto.opset_import[0].version
        )

        # Describe input and output tensors
        def describe(tensor_infos):

            rows = []

            for tensor in tensor_infos:

                dimensions = [
                    (
                        dimension.dim_value
                        if dimension.HasField("dim_value")
                        else dimension.dim_param
                    )
                    for dimension
                    in tensor.type.tensor_type.shape.dim
                ]

                rows.append(
                    f"{tensor.name} {dimensions}"
                )

            return rows

        # Initializers are model weights, not real inputs
        weights = {
            weight.name
            for weight in model_proto.graph.initializer
        }

        real_inputs = [
            item
            for item in model_proto.graph.input
            if item.name not in weights
        ]

        print(
            "Inputs         :",
            describe(real_inputs)
        )

        print(
            "Outputs        :",
            describe(model_proto.graph.output)
        )

        operations = Counter(
            node.op_type
            for node in model_proto.graph.node
        )

        print(
            f"Graph          : "
            f"{len(model_proto.graph.node)} operations, "
            f"{len(model_proto.graph.initializer)} weight tensors"
        )

        print(
            "Most common operations:",
            ", ".join(
                f"{name} x{count}"
                for name, count
                in operations.most_common(6)
            )
        )

    except ImportError:

        print(
            "ONNX package is missing. Install it with:"
        )

        print("python -m pip install onnx")

    except Exception as error:

        print(
            "Could not inspect the ONNX file:",
            type(error).__name__,
            error
        )


# ---------------------------------------------------------
# 4. TENSORRT ENGINE
# ---------------------------------------------------------

print("\n=== 4. TENSORRT ENGINE ===")

engines = [
    path
    for path in found
    if path.suffix.lower() in (".engine", ".plan")
]

if engines:

    for path in engines:

        print(
            f"{path} "
            f"({path.stat().st_size / 1e6:.1f} MB)"
        )

        print(
            "It cannot be opened with PyTorch or ONNX tools."
        )

        print(
            "Only TensorRT on a matching NVIDIA GPU can load it."
        )

else:

    print(
        "No engine found. This is expected on a computer "
        "without an NVIDIA GPU."
    )

    print(
        "An engine is built from the ONNX file in Session 13, "
        "on the GPU that will run it."
    )


# ---------------------------------------------------------
# 5. WHY THE ARTEFACTS ARE NOT INTERCHANGEABLE
# ---------------------------------------------------------

print("\n=== 5. COMPARISON ===")

rows = [
    (
        "Checkpoint (.pt)",
        "weights + architecture",
        "PyTorch + Python",
        "training, fine-tuning, export"
    ),
    (
        "ONNX (.onnx)",
        "graph + weights, no Python",
        "ONNX Runtime or TensorRT",
        "portable hand-over format"
    ),
    (
        "Engine (.engine)",
        "optimised, compiled graph",
        "NVIDIA GPU + TensorRT",
        "fast serving on that GPU"
    ),
]

print(
    f"{'artefact':<20}"
    f"{'contains':<30}"
    f"{'needs':<28}"
    f"used for"
)

for row in rows:

    print(
        f"{row[0]:<20}"
        f"{row[1]:<30}"
        f"{row[2]:<28}"
        f"{row[3]}"
    )


# ---------------------------------------------------------
# 6. COMPLETION
# ---------------------------------------------------------

print("\n==========================================")
print("VISIONPORT SESSION 7 INSPECTION COMPLETED")
print("==========================================")
