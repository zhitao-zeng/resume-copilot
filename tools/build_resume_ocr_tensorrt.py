#!/usr/bin/env python3
"""Build deployment-local TensorRT engines for resume PP-OCRv6.

Engines are hardware and TensorRT-version specific and must stay outside Git.
FP32 is the default because dates and business metrics are immutable resume
facts; FP16 remains available for controlled accuracy/latency experiments.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


Shape = tuple[int, int, int, int]


@dataclass(frozen=True)
class ShapeProfile:
    minimum: Shape
    optimum: Shape
    maximum: Shape


DETECTOR_PROFILES = (
    ShapeProfile(
        (1, 3, 1600, 448),
        (1, 3, 1600, 1120),
        (1, 3, 1600, 1216),
    ),
)
RECOGNIZER_PROFILES = (
    ShapeProfile(
        (1, 3, 48, 320),
        (4, 3, 48, 320),
        (8, 3, 48, 320),
    ),
    ShapeProfile(
        (1, 3, 48, 328),
        (1, 3, 48, 640),
        (1, 3, 48, 2048),
    ),
)
CLASSIFIER_PROFILES = (
    ShapeProfile(
        (1, 3, 48, 192),
        (4, 3, 48, 192),
        (8, 3, 48, 192),
    ),
)


def _validate_profiles(profiles: tuple[ShapeProfile, ...]) -> None:
    for profile in profiles:
        for lower, optimum, upper in zip(
            profile.minimum, profile.optimum, profile.maximum
        ):
            if not lower <= optimum <= upper:
                raise ValueError(f"invalid TensorRT profile: {profile}")


def build_engine(
    onnx_path: Path,
    output_path: Path,
    profiles: tuple[ShapeProfile, ...],
    *,
    precision: str,
    workspace_mb: int,
    optimization_level: int,
) -> None:
    import tensorrt as trt

    _validate_profiles(profiles)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")
    if network.num_inputs != 1:
        raise RuntimeError(
            f"OCR ONNX must have exactly one input, got {network.num_inputs}"
        )
    input_tensor = network.get_input(0)
    if len(tuple(input_tensor.shape)) != 4:
        raise RuntimeError(
            f"OCR ONNX input must be rank 4, got {input_tensor.shape}"
        )
    static_shape = []
    for axis in range(4):
        values = {
            value
            for profile in profiles
            for value in (profile.minimum[axis], profile.maximum[axis])
        }
        static_shape.append(values.pop() if len(values) == 1 else -1)
    input_tensor.shape = tuple(static_shape)

    config = builder.create_builder_config()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * 1024 * 1024
    )
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(optimization_level)
    for shape_profile in profiles:
        profile = builder.create_optimization_profile()
        accepted = profile.set_shape(
            input_tensor.name,
            shape_profile.minimum,
            shape_profile.optimum,
            shape_profile.maximum,
        )
        if accepted is False:
            raise RuntimeError(f"TensorRT rejected profile {shape_profile}")
        profile_index = config.add_optimization_profile(profile)
        if isinstance(profile_index, int) and profile_index < 0:
            raise RuntimeError(f"TensorRT failed to add profile {shape_profile}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(bytes(serialized))
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, output_path)
        os.chmod(output_path, 0o644)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-onnx", type=Path, required=True)
    parser.add_argument("--cls-onnx", type=Path, required=True)
    parser.add_argument("--primary-rec-onnx", type=Path, required=True)
    parser.add_argument("--secondary-rec-onnx", type=Path, required=True)
    parser.add_argument("--medium-rec-onnx", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--workspace-mb", type=int, default=1024)
    parser.add_argument(
        "--builder-optimization-level", type=int, choices=range(0, 6), default=0
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "det.engine": (args.det_onnx, DETECTOR_PROFILES),
        "cls.engine": (args.cls_onnx, CLASSIFIER_PROFILES),
        "primary-rec.engine": (args.primary_rec_onnx, RECOGNIZER_PROFILES),
        "secondary-rec.engine": (args.secondary_rec_onnx, RECOGNIZER_PROFILES),
        "medium-rec.engine": (args.medium_rec_onnx, RECOGNIZER_PROFILES),
    }
    for source, _profiles in inputs.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    if not args.keys.is_file():
        raise FileNotFoundError(args.keys)
    if args.workspace_mb <= 0:
        raise ValueError("--workspace-mb must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output_name, (source, profiles) in inputs.items():
        print(
            f"Building {output_name} ({args.precision}) from {source}",
            flush=True,
        )
        build_engine(
            source,
            args.output_dir / output_name,
            profiles,
            precision=args.precision,
            workspace_mb=args.workspace_mb,
            optimization_level=args.builder_optimization_level,
        )
    (args.output_dir / "keys.txt").write_bytes(args.keys.read_bytes())


if __name__ == "__main__":
    main()
