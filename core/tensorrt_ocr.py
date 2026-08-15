"""Native TensorRT PP-OCRv6 inference with numeric fact consensus.

The implementation intentionally keeps TensorRT imports lazy so the ordinary
CPU image remains usable.  The production GPU path keeps the established CPU
detector/crop geometry, runs the Medium recognizer over every line, and asks
two independent Small recognizers to verify only numeric lines.  A parity mode
that keeps the complete CPU Small primary is retained as a conservative
fallback.
"""

from __future__ import annotations

import copy
import ctypes
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


logger = logging.getLogger(__name__)

_OCR_MEAN = (127.5, 127.5, 127.5)
_OCR_NORMAL = (1 / 127.5, 1 / 127.5, 1 / 127.5)

TENSORRT_CONSENSUS_FILES = (
    "det.engine",
    "cls.engine",
    "primary-rec.engine",
    "secondary-rec.engine",
    "medium-rec.engine",
    "keys.txt",
)
TENSORRT_RECOGNITION_CONSENSUS_FILES = (
    "secondary-rec.engine",
    "medium-rec.engine",
    "keys.txt",
)
TENSORRT_MEDIUM_PRIMARY_FILES = (
    "primary-rec.engine",
    "secondary-rec.engine",
    "medium-rec.engine",
    "keys.txt",
)


class TensorRTOCRShapeError(ValueError):
    """Raised when an OCR tensor cannot be represented by an engine profile."""


@dataclass
class TensorRTOCRResult:
    boxes: object
    txts: tuple[str, ...]
    scores: tuple[float, ...]
    numeric_consensus_applied: bool = True


def is_complete_tensorrt_bundle(root: Path) -> bool:
    try:
        return root.is_dir() and all((root / name).is_file() for name in TENSORRT_CONSENSUS_FILES)
    except OSError:
        return False


def is_complete_tensorrt_recognition_bundle(root: Path) -> bool:
    try:
        return root.is_dir() and all(
            (root / name).is_file()
            for name in TENSORRT_RECOGNITION_CONSENSUS_FILES
        )
    except OSError:
        return False


def is_complete_tensorrt_medium_primary_bundle(root: Path) -> bool:
    try:
        return root.is_dir() and all(
            (root / name).is_file()
            for name in TENSORRT_MEDIUM_PRIMARY_FILES
        )
    except OSError:
        return False


class _CudaRuntime:
    """Small CUDA Runtime API wrapper used only for TensorRT buffers."""

    _MEMCPY_HOST_TO_DEVICE = 1
    _MEMCPY_DEVICE_TO_HOST = 2
    _STREAM_NON_BLOCKING = 1

    def __init__(self, device_id: int):
        self._device_id = int(device_id)
        self._library = ctypes.CDLL("libcudart.so")
        self._bind_functions()
        self._stream = ctypes.c_void_p()
        self._set_device()
        self._check(
            self._library.cudaStreamCreateWithFlags(
                ctypes.byref(self._stream), self._STREAM_NON_BLOCKING
            ),
            "cudaStreamCreateWithFlags",
        )

    def _bind_functions(self) -> None:
        library = self._library
        library.cudaGetErrorString.argtypes = [ctypes.c_int]
        library.cudaGetErrorString.restype = ctypes.c_char_p
        library.cudaSetDevice.argtypes = [ctypes.c_int]
        library.cudaSetDevice.restype = ctypes.c_int
        library.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        library.cudaStreamCreateWithFlags.restype = ctypes.c_int
        library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        library.cudaStreamDestroy.restype = ctypes.c_int
        library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        library.cudaStreamSynchronize.restype = ctypes.c_int
        library.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        library.cudaMalloc.restype = ctypes.c_int
        library.cudaFree.argtypes = [ctypes.c_void_p]
        library.cudaFree.restype = ctypes.c_int
        library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        library.cudaMemcpyAsync.restype = ctypes.c_int

    def _check(self, code: int, operation: str) -> None:
        if code == 0:
            return
        message = self._library.cudaGetErrorString(code)
        detail = message.decode("utf-8", errors="replace") if message else "unknown"
        raise RuntimeError(f"{operation} failed with CUDA error {code}: {detail}")

    def _set_device(self) -> None:
        self._check(self._library.cudaSetDevice(self._device_id), "cudaSetDevice")

    @property
    def stream_handle(self) -> int:
        return int(self._stream.value or 0)

    def malloc(self, size: int) -> int:
        self._set_device()
        pointer = ctypes.c_void_p()
        self._check(
            self._library.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(size)),
            "cudaMalloc",
        )
        if not pointer.value:
            raise RuntimeError("cudaMalloc returned a null device pointer")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        if not pointer:
            return
        self._set_device()
        self._check(self._library.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def copy_host_to_device(self, pointer: int, array) -> None:
        self._set_device()
        self._check(
            self._library.cudaMemcpyAsync(
                ctypes.c_void_p(pointer),
                ctypes.c_void_p(array.ctypes.data),
                ctypes.c_size_t(array.nbytes),
                self._MEMCPY_HOST_TO_DEVICE,
                self._stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )

    def copy_device_to_host(self, array, pointer: int) -> None:
        self._set_device()
        self._check(
            self._library.cudaMemcpyAsync(
                ctypes.c_void_p(array.ctypes.data),
                ctypes.c_void_p(pointer),
                ctypes.c_size_t(array.nbytes),
                self._MEMCPY_DEVICE_TO_HOST,
                self._stream,
            ),
            "cudaMemcpyAsync(D2H)",
        )

    def synchronize(self) -> None:
        self._set_device()
        self._check(
            self._library.cudaStreamSynchronize(self._stream),
            "cudaStreamSynchronize",
        )

    def close(self) -> None:
        if not self._stream.value:
            return
        self._set_device()
        stream = self._stream
        self._stream = ctypes.c_void_p()
        self._check(self._library.cudaStreamDestroy(stream), "cudaStreamDestroy")


class _TensorRTModelSession:
    """One TensorRT engine with reusable host/device buffers."""

    def __init__(self, engine_path: Path, *, device_id: int):
        import numpy as np
        import tensorrt as trt

        self._np = np
        self._cuda = None
        self._input_device = 0
        self._input_capacity = 0
        self._output_device = 0
        self._output_capacity = 0
        self._context = None
        self._engine = None
        self._runtime = None

        try:
            self._cuda = _CudaRuntime(device_id)
            trt_logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(trt_logger)
            self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if self._engine is None:
                raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(f"failed to create TensorRT context: {engine_path}")
        except Exception:
            self.close()
            raise

        inputs: list[str] = []
        outputs: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            self.close()
            raise RuntimeError(
                "OCR TensorRT engine must have exactly one input and output; "
                f"got inputs={inputs}, outputs={outputs}"
            )
        self._input_name = inputs[0]
        self._output_name = outputs[0]
        self._input_numpy_dtype = np.dtype(
            trt.nptype(self._engine.get_tensor_dtype(self._input_name))
        )
        self._output_numpy_dtype = np.dtype(
            trt.nptype(self._engine.get_tensor_dtype(self._output_name))
        )
        self._mean = np.asarray(_OCR_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        self._normal = np.asarray(_OCR_NORMAL, dtype=np.float32).reshape(1, 3, 1, 1)
        self._profiles = self._read_profiles()
        self._active_profile = 0

    def _read_profiles(self):
        static_shape = tuple(
            int(value) for value in self._engine.get_tensor_shape(self._input_name)
        )
        if static_shape and all(value > 0 for value in static_shape):
            return [(static_shape, static_shape, static_shape)]

        profiles = []
        for index in range(self._engine.num_optimization_profiles):
            minimum, optimum, maximum = self._engine.get_tensor_profile_shape(
                self._input_name, index
            )
            profiles.append(
                tuple(
                    tuple(int(value) for value in shape)
                    for shape in (minimum, optimum, maximum)
                )
            )
        if not profiles:
            raise RuntimeError("OCR TensorRT engine has no optimization profile")
        return profiles

    @property
    def optimization_shape(self) -> tuple[int, ...]:
        return self._profiles[0][1]

    @property
    def primary_profile(self):
        return self._profiles[0]

    def _select_profile(self, shape: tuple[int, ...]) -> int:
        candidates = []
        for index, (minimum, optimum, maximum) in enumerate(self._profiles):
            if len(shape) != len(minimum):
                continue
            if all(
                lower <= value <= upper
                for value, lower, upper in zip(shape, minimum, maximum)
            ):
                distance = sum(
                    abs(math.log(max(1, value) / max(1, ideal)))
                    for value, ideal in zip(shape, optimum)
                )
                candidates.append((distance, index))
        if candidates:
            return min(candidates)[1]
        ranges = [
            {"min": minimum, "max": maximum}
            for minimum, _optimum, maximum in self._profiles
        ]
        raise TensorRTOCRShapeError(
            f"OCR TensorRT input shape {shape} is outside profiles {ranges}"
        )

    def max_batch_size(self, height: int, width: int) -> int:
        compatible = [
            maximum[0]
            for minimum, _optimum, maximum in self._profiles
            if len(minimum) == 4
            and minimum[1] <= 3 <= maximum[1]
            and minimum[2] <= height <= maximum[2]
            and minimum[3] <= width <= maximum[3]
            and minimum[0] <= 1 <= maximum[0]
        ]
        if not compatible:
            self._select_profile((1, 3, int(height), int(width)))
        return max(compatible)

    def run_uint8(self, image, shape: tuple[int, ...]):
        return self.run_uint8_batch(image[None], shape)

    def run_uint8_batch(self, images, shape: tuple[int, ...]):
        images = self._np.ascontiguousarray(images, dtype=self._np.uint8)
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"invalid OCR TensorRT NCHW shape: {shape}")
        expected = (shape[0], shape[2], shape[3], shape[1])
        if images.ndim != 4 or images.shape != expected:
            raise ValueError(
                f"OCR TensorRT image batch/shape mismatch: {images.shape} vs {shape}"
            )
        nchw = images.transpose(0, 3, 1, 2)
        if self._input_numpy_dtype == self._np.dtype(self._np.float32):
            array = self._np.empty(nchw.shape, dtype=self._np.float32)
            self._np.subtract(nchw, self._mean, out=array)
            self._np.multiply(array, self._normal, out=array)
        else:
            value = (nchw.astype(self._np.float32) - self._mean) * self._normal
            array = self._np.ascontiguousarray(value, dtype=self._input_numpy_dtype)
        return self._run(array)

    def run_normalized_batch(self, images, shape: tuple[int, ...]):
        """Run NHWC images already normalized to RapidOCR's [-1, 1]."""

        images = self._np.ascontiguousarray(images, dtype=self._np.float32)
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"invalid OCR TensorRT NCHW shape: {shape}")
        expected = (shape[0], shape[2], shape[3], shape[1])
        if images.ndim != 4 or images.shape != expected:
            raise ValueError(
                f"OCR TensorRT normalized batch/shape mismatch: "
                f"{images.shape} vs {shape}"
            )
        array = self._np.ascontiguousarray(
            images.transpose(0, 3, 1, 2), dtype=self._input_numpy_dtype
        )
        return self._run(array)

    def _run(self, array):
        shape = tuple(int(value) for value in array.shape)
        profile = self._select_profile(shape)
        cuda = self._cuda
        if profile != self._active_profile:
            if not self._context.set_optimization_profile_async(
                profile, cuda.stream_handle
            ):
                raise RuntimeError(f"failed to select TensorRT profile {profile}")
            self._active_profile = profile
        if not self._context.set_input_shape(self._input_name, shape):
            raise RuntimeError(f"TensorRT rejected OCR input shape {shape}")
        output_shape = tuple(
            int(value) for value in self._context.get_tensor_shape(self._output_name)
        )
        if any(value < 0 for value in output_shape):
            raise RuntimeError(f"TensorRT produced unresolved output shape {output_shape}")

        self._input_device = self._ensure_capacity(
            "_input_device", "_input_capacity", array.nbytes
        )
        output = self._np.empty(output_shape, dtype=self._output_numpy_dtype)
        self._output_device = self._ensure_capacity(
            "_output_device", "_output_capacity", output.nbytes
        )
        cuda.copy_host_to_device(self._input_device, array)
        self._context.set_tensor_address(self._input_name, self._input_device)
        self._context.set_tensor_address(self._output_name, self._output_device)
        if not self._context.execute_async_v3(cuda.stream_handle):
            raise RuntimeError("TensorRT OCR execution failed")
        cuda.copy_device_to_host(output, self._output_device)
        cuda.synchronize()
        return output

    def _ensure_capacity(
        self, pointer_attribute: str, capacity_attribute: str, required: int
    ) -> int:
        pointer = getattr(self, pointer_attribute)
        capacity = getattr(self, capacity_attribute)
        if required <= capacity:
            return pointer
        if pointer:
            self._cuda.free(pointer)
            setattr(self, pointer_attribute, 0)
            setattr(self, capacity_attribute, 0)
        replacement = self._cuda.malloc(required)
        setattr(self, pointer_attribute, replacement)
        setattr(self, capacity_attribute, required)
        return replacement

    def close(self) -> None:
        cuda = self._cuda
        if cuda is not None:
            if self._input_device:
                cuda.free(self._input_device)
                self._input_device = 0
                self._input_capacity = 0
            if self._output_device:
                cuda.free(self._output_device)
                self._output_device = 0
                self._output_capacity = 0
        self._context = None
        self._engine = None
        self._runtime = None
        if cuda is not None:
            cuda.close()
            self._cuda = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TensorRTConsensusOCR:
    """PP-OCRv6 pipeline optimized for fact-safe resume extraction."""

    _RECOGNITION_PREPROCESS_MODES = {
        "standard-resize",
        "native-height-pad",
    }

    def __init__(
        self,
        root: Path,
        *,
        device_id: int = 0,
        text_score: float = 0.5,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.5,
        det_unclip_ratio: float = 1.6,
        cls_thresh: float = 0.9,
        max_side_len: int = 1600,
        recognition_preprocess: str = "standard-resize",
        numeric_signature: Callable[[object], tuple[str, ...]],
        suppress_numeric: Callable[[object], str],
        repair_text: Callable[[object], str],
    ):
        import numpy as np
        from rapidocr.ch_ppocr_det.utils import DBPostProcess
        from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
        from rapidocr.utils.process_img import get_rotate_crop_image

        root = Path(root)
        if not is_complete_tensorrt_bundle(root):
            raise FileNotFoundError(f"incomplete PP-OCRv6 TensorRT bundle: {root}")
        self._sessions: list[_TensorRTModelSession] = []
        try:
            self._det = self._new_session(root / "det.engine", device_id)
            self._cls = self._new_session(root / "cls.engine", device_id)
            self._primary_rec = self._new_session(root / "primary-rec.engine", device_id)
            self._secondary_rec = self._new_session(root / "secondary-rec.engine", device_id)
            self._medium_rec = self._new_session(root / "medium-rec.engine", device_id)
        except Exception:
            self.close()
            raise
        self._det_postprocess = DBPostProcess(
            thresh=float(det_thresh),
            box_thresh=float(det_box_thresh),
            max_candidates=1000,
            unclip_ratio=float(det_unclip_ratio),
            use_dilation=True,
            score_mode="fast",
        )
        self._rec_decode = CTCLabelDecode(character_path=root / "keys.txt")
        self._crop = get_rotate_crop_image
        self._text_score = float(text_score)
        self._cls_thresh = float(cls_thresh)
        self._max_side_len = max(32, int(max_side_len))
        self._recognition_preprocess = self._validate_recognition_preprocess(
            recognition_preprocess
        )
        self._numeric_signature = numeric_signature
        self._suppress_numeric = suppress_numeric
        self._repair_text = repair_text
        self._np = np

    @classmethod
    def _validate_recognition_preprocess(cls, value: object) -> str:
        mode = str(value or "standard-resize").strip().lower()
        if mode not in cls._RECOGNITION_PREPROCESS_MODES:
            raise ValueError(
                "unknown TensorRT OCR recognition preprocessing mode: "
                f"{value!r}"
            )
        return mode

    def _new_session(self, path: Path, device_id: int) -> _TensorRTModelSession:
        session = _TensorRTModelSession(path, device_id=device_id)
        self._sessions.append(session)
        return session

    @staticmethod
    def _multiple_of_32(value: float) -> int:
        return max(32, int(round(value / 32)) * 32)

    def _detector_input(self, image):
        import cv2

        height, width = image.shape[:2]
        minimum, optimum, maximum = self._det.primary_profile
        target_height = int(optimum[2])
        target_width = self._multiple_of_32(width * target_height / max(1, height))
        # The A100 document engine intentionally has a narrow portrait profile.
        # Clamping keeps uncommon aspect ratios on the accelerated path; only
        # detection is stretched, while recognition still sees original crops.
        target_width = min(max(target_width, int(minimum[3])), int(maximum[3]))
        if (target_width, target_height) == (width, height):
            return image
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA if target_height < height else cv2.INTER_LINEAR,
        )

    def _detect(self, image):
        detector_input = self._detector_input(image)
        height, width = detector_input.shape[:2]
        prediction = self._det.run_uint8(
            detector_input, (1, 3, height, width)
        )
        boxes, scores = self._det_postprocess(prediction, image.shape[:2])
        if len(boxes) == 0:
            return self._np.empty((0, 4, 2), dtype=self._np.float32), []
        order = sorted(
            range(len(boxes)),
            key=lambda index: (boxes[index][0][1], boxes[index][0][0]),
        )
        ordered_boxes = [boxes[index] for index in order]
        ordered_scores = [float(scores[index]) for index in order]
        # Match PaddleOCR/RapidOCR's same-row correction exactly. A plain
        # ``(y, x)`` key reverses nearby two-column rows when their y values
        # differ by only a few pixels.
        for index in range(1, len(ordered_boxes)):
            cursor = index
            while cursor > 0:
                current = ordered_boxes[cursor]
                previous = ordered_boxes[cursor - 1]
                if not (
                    abs(float(current[0][1]) - float(previous[0][1])) < 10
                    and float(current[0][0]) < float(previous[0][0])
                ):
                    break
                ordered_boxes[cursor], ordered_boxes[cursor - 1] = (
                    ordered_boxes[cursor - 1], ordered_boxes[cursor]
                )
                ordered_scores[cursor], ordered_scores[cursor - 1] = (
                    ordered_scores[cursor - 1], ordered_scores[cursor]
                )
                cursor -= 1
        return self._np.asarray(ordered_boxes), ordered_scores

    @staticmethod
    def _prepare_classifier_crop(crop):
        import cv2
        import numpy as np

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return None
        target_height, target_width = 48, 192
        resized_width = min(
            target_width,
            max(1, int(math.ceil(target_height * width / float(height)))),
        )
        resized = cv2.resize(
            crop, (resized_width, target_height), interpolation=cv2.INTER_LINEAR
        )
        resized = resized.astype(np.float32) / 255.0
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((target_height, target_width, 3), dtype=np.float32)
        padded[:, :resized_width] = resized
        return padded

    def _orient_crops(self, crops):
        import cv2

        oriented = list(crops)
        prepared = []
        for index, crop in enumerate(crops):
            value = self._prepare_classifier_crop(crop)
            if value is not None:
                prepared.append((index, value))
        max_batch = self._cls.max_batch_size(48, 192)
        for offset in range(0, len(prepared), max_batch):
            chunk = prepared[offset:offset + max_batch]
            images = self._np.stack([entry[1] for entry in chunk])
            prediction = self._cls.run_normalized_batch(
                images, (len(chunk), 3, 48, 192)
            )
            for entry, probabilities in zip(chunk, prediction):
                label = int(self._np.argmax(probabilities))
                score = float(probabilities[label])
                if label == 1 and score > self._cls_thresh:
                    oriented[entry[0]] = cv2.rotate(
                        crops[entry[0]], cv2.ROTATE_180
                    )
        return oriented

    @staticmethod
    def _prepare_recognition_crop(
        crop,
        *,
        target_width: int | None = None,
        preprocess_mode: str = "standard-resize",
    ):
        import cv2
        import numpy as np

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return None
        preprocess_mode = TensorRTConsensusOCR._validate_recognition_preprocess(
            preprocess_mode
        )
        ratio = width / float(height)
        target_height = 48
        preserve_native = (
            preprocess_mode == "native-height-pad"
            and height <= target_height
            and width <= 2048
        )
        if target_width is None:
            content_width = (
                width
                if preserve_native
                else int(math.ceil(target_height * ratio))
            )
            target_width = max(320, content_width)
            target_width = int(math.ceil(target_width / 8)) * 8
        target_width = min(2048, max(320, int(target_width)))
        # The deployed recognition profile has a deliberate 320/328 split.
        if 320 < target_width < 328:
            target_width = 328
        if preserve_native and width <= target_width:
            normalized = crop.astype(np.float32) / 255.0
            normalized = (normalized - 0.5) / 0.5
            padded = np.zeros(
                (target_height, target_width, 3), dtype=np.float32
            )
            # Preserve the detector's source raster. The recognizer still
            # receives its required 48 px input height, and unused cells keep
            # RapidOCR's normalized-zero padding value.
            padded[:height, :width] = normalized
            return padded, min(ratio, target_width / target_height)
        resized_width = min(
            target_width, max(1, int(math.ceil(target_height * ratio)))
        )
        resized = cv2.resize(
            crop, (resized_width, target_height), interpolation=cv2.INTER_LINEAR
        )
        resized = resized.astype(np.float32) / 255.0
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((target_height, target_width, 3), dtype=np.float32)
        padded[:, :resized_width] = resized
        return padded, min(ratio, target_width / target_height)

    def _oriented_crops(self, image, boxes):
        crops = [self._crop(image, copy.deepcopy(box)) for box in boxes]
        return self._orient_crops(crops)

    def _prepare_recognition_crops(
        self,
        crops,
        *,
        result_indexes: Sequence[int] | None = None,
        batch_size: int = 6,
        preprocess_mode: str | None = None,
    ):
        """Prepare recognition crops without losing their source indexes.

        ``standard-resize`` reproduces RapidOCR's ratio-sorted, per-batch
        width contract. ``native-height-pad`` gives each crop its smallest
        compatible engine width so crops no taller than 48 px can retain
        their original pixels. Width-identical tensors are still grouped by
        :meth:`_recognize`, including the common 320 px profile.
        """

        result_indexes = list(result_indexes) if result_indexes is not None else list(range(len(crops)))
        if len(result_indexes) != len(crops):
            raise ValueError("OCR crop indexes must match crop count")
        ratios = [
            crop.shape[1] / float(max(1, crop.shape[0]))
            for crop in crops
        ]
        order = self._np.argsort(self._np.asarray(ratios))
        prepared = []
        preprocess_mode = self._validate_recognition_preprocess(
            preprocess_mode
            or getattr(self, "_recognition_preprocess", "standard-resize")
        )
        if preprocess_mode == "native-height-pad":
            for local_index in order:
                local_index = int(local_index)
                value = self._prepare_recognition_crop(
                    crops[local_index],
                    preprocess_mode=preprocess_mode,
                )
                if value is not None:
                    padded, ratio = value
                    prepared.append(
                        (result_indexes[local_index], padded, ratio)
                    )
            return prepared
        for offset in range(0, len(order), max(1, int(batch_size))):
            chunk = order[offset:offset + max(1, int(batch_size))]
            max_ratio = max(
                [320 / 48]
                + [ratios[int(local_index)] for local_index in chunk]
            )
            target_width = min(2048, int(48 * max_ratio))
            if 320 < target_width < 328:
                target_width = 328
            for local_index in chunk:
                local_index = int(local_index)
                value = self._prepare_recognition_crop(
                    crops[local_index],
                    target_width=target_width,
                    preprocess_mode=preprocess_mode,
                )
                if value is not None:
                    padded, ratio = value
                    prepared.append((result_indexes[local_index], padded, ratio))
        return prepared

    def _prepare_crops(self, image, boxes):
        return self._prepare_recognition_crops(
            self._oriented_crops(image, boxes)
        )

    def _recognize(
        self,
        session: _TensorRTModelSession,
        prepared: Sequence[tuple[int, object, float]],
        result_size: int,
    ) -> list[tuple[str, float]]:
        recognized = [("", 0.0)] * result_size
        prepared_by_width = defaultdict(list)
        for entry in prepared:
            prepared_by_width[entry[1].shape[1]].append(entry)
        for target_width, group in prepared_by_width.items():
            max_batch = session.max_batch_size(48, target_width)
            for offset in range(0, len(group), max_batch):
                chunk = group[offset:offset + max_batch]
                images = self._np.stack([entry[1] for entry in chunk])
                ratios = tuple(entry[2] for entry in chunk)
                prediction = session.run_normalized_batch(
                    images, (len(chunk), 3, 48, target_width)
                )
                line_results, _word_results = self._rec_decode(
                    prediction,
                    False,
                    wh_ratio_list=ratios,
                    max_wh_ratio=target_width / 48,
                )
                for entry, result in zip(chunk, line_results):
                    text, score = result
                    recognized[entry[0]] = str(text), float(score)
        return recognized

    def __call__(self, image) -> TensorRTOCRResult:
        from rapidocr.utils.process_img import resize_image_within_bounds

        started = time.perf_counter()
        image = self._np.ascontiguousarray(image, dtype=self._np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"TensorRT OCR expects HWC uint8 RGB/BGR, got {image.shape}")

        original_height, original_width = image.shape[:2]
        working_image, ratio_h, ratio_w = resize_image_within_bounds(
            image, 30, self._max_side_len
        )

        detect_started = time.perf_counter()
        boxes, _det_scores = self._detect(working_image)
        detect_s = time.perf_counter() - detect_started
        if len(boxes) == 0:
            return TensorRTOCRResult(boxes=boxes, txts=(), scores=())

        crop_started = time.perf_counter()
        oriented_crops = self._oriented_crops(working_image, boxes)
        prepared = self._prepare_recognition_crops(oriented_crops)
        crop_cls_s = time.perf_counter() - crop_started

        primary_started = time.perf_counter()
        primary = self._recognize(self._primary_rec, prepared, len(boxes))
        primary_s = time.perf_counter() - primary_started
        secondary_started = time.perf_counter()
        secondary = self._recognize(self._secondary_rec, prepared, len(boxes))
        secondary_s = time.perf_counter() - secondary_started

        kept_indexes = [
            index
            for index, (text, score) in enumerate(primary)
            if text.strip() and score >= self._text_score
        ]
        numeric_indexes = [
            index
            for index in kept_indexes
            if self._numeric_signature(primary[index][0])
            or self._numeric_signature(secondary[index][0])
        ]
        medium_indexes = [
            index
            for index in numeric_indexes
            if self._numeric_signature(primary[index][0])
            == self._numeric_signature(secondary[index][0])
        ]
        medium_prepared = self._prepare_recognition_crops(
            [oriented_crops[index] for index in medium_indexes],
            result_indexes=medium_indexes,
        )
        medium_started = time.perf_counter()
        medium = self._recognize(
            self._medium_rec, medium_prepared, len(boxes)
        ) if medium_prepared else [("", 0.0)] * len(boxes)
        medium_s = time.perf_counter() - medium_started

        reconciled: dict[int, str] = {}
        disputed = 0
        for index in numeric_indexes:
            primary_signature = self._numeric_signature(primary[index][0])
            secondary_signature = self._numeric_signature(secondary[index][0])
            medium_signature = self._numeric_signature(medium[index][0])
            if (
                index in medium_indexes
                and primary_signature == secondary_signature == medium_signature
            ):
                reconciled[index] = self._repair_text(primary[index][0])
            else:
                reconciled[index] = self._suppress_numeric(primary[index][0])
                disputed += 1

        final_boxes = []
        final_texts: list[str] = []
        final_scores: list[float] = []
        for index in kept_indexes:
            text = reconciled.get(index, primary[index][0]).strip()
            if not text:
                continue
            final_boxes.append(boxes[index])
            final_texts.append(text)
            final_scores.append(primary[index][1])

        final_boxes_array = self._np.asarray(final_boxes, dtype=self._np.float32)
        if len(final_boxes_array):
            final_boxes_array[:, :, 0] *= ratio_w
            final_boxes_array[:, :, 1] *= ratio_h
            final_boxes_array[:, :, 0] = self._np.clip(
                final_boxes_array[:, :, 0], 0, original_width
            )
            final_boxes_array[:, :, 1] = self._np.clip(
                final_boxes_array[:, :, 1], 0, original_height
            )

        logger.info(
            "TensorRT OCR consensus finished | lines=%s numeric_lines=%s "
            "medium_lines=%s disputed_lines=%s det_s=%.3f crop_cls_s=%.3f "
            "primary_s=%.3f secondary_s=%.3f medium_s=%.3f total_s=%.3f",
            len(final_texts),
            len(numeric_indexes),
            len(medium_indexes),
            disputed,
            detect_s,
            crop_cls_s,
            primary_s,
            secondary_s,
            medium_s,
            time.perf_counter() - started,
        )
        return TensorRTOCRResult(
            boxes=final_boxes_array,
            txts=tuple(final_texts),
            scores=tuple(final_scores),
        )

    def warm_up(self) -> None:
        det_shape = self._det.optimization_shape
        self._det.run_uint8_batch(
            self._np.zeros(
                (det_shape[0], det_shape[2], det_shape[3], 3), dtype=self._np.uint8
            ),
            det_shape,
        )
        cls_shape = self._cls.optimization_shape
        self._cls.run_uint8_batch(
            self._np.zeros(
                (cls_shape[0], cls_shape[2], cls_shape[3], 3), dtype=self._np.uint8
            ),
            cls_shape,
        )
        for session in (self._primary_rec, self._secondary_rec, self._medium_rec):
            rec_shape = session.optimization_shape
            session.run_uint8_batch(
                self._np.zeros(
                    (rec_shape[0], rec_shape[2], rec_shape[3], 3),
                    dtype=self._np.uint8,
                ),
                rec_shape,
            )

    def close(self) -> None:
        while self._sessions:
            self._sessions.pop().close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TensorRTRecognitionConsensus(TensorRTConsensusOCR):
    """Keep the established CPU primary and accelerate its two witnesses.

    This hybrid is intentionally asymmetric. The CPU primary retains the
    exact detector boxes, primary text and layout behavior already validated
    in production. TensorRT only replaces the expensive secondary-small and
    selective medium recognition calls, using the same high-resolution crops
    and batch padding contract as RapidOCR.
    """

    def __init__(
        self,
        root: Path,
        *,
        primary_ocr,
        device_id: int = 0,
        recognition_preprocess: str = "standard-resize",
        numeric_signature: Callable[[object], tuple[str, ...]],
        suppress_numeric: Callable[[object], str],
        repair_text: Callable[[object], str],
    ):
        import numpy as np
        from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode

        root = Path(root)
        if not is_complete_tensorrt_recognition_bundle(root):
            raise FileNotFoundError(
                f"incomplete PP-OCRv6 TensorRT recognition bundle: {root}"
            )
        self._sessions: list[_TensorRTModelSession] = []
        self._primary_ocr = primary_ocr
        self._np = np
        self._recognition_preprocess = self._validate_recognition_preprocess(
            recognition_preprocess
        )
        try:
            self._secondary_rec = self._new_session(
                root / "secondary-rec.engine", device_id
            )
            self._medium_rec = self._new_session(
                root / "medium-rec.engine", device_id
            )
        except Exception:
            self.close()
            raise
        self._rec_decode = CTCLabelDecode(character_path=root / "keys.txt")
        self._numeric_signature = numeric_signature
        self._suppress_numeric = suppress_numeric
        self._repair_text = repair_text

    def __call__(self, image) -> TensorRTOCRResult:
        started = time.perf_counter()
        image = self._np.ascontiguousarray(image, dtype=self._np.uint8)

        primary_started = time.perf_counter()
        result = self._primary_ocr(image)
        primary_s = time.perf_counter() - primary_started
        if (
            result is None
            or getattr(result, "boxes", None) is None
            or getattr(result, "txts", None) is None
            or len(result.boxes) == 0
        ):
            return result

        crop_started = time.perf_counter()
        crops = self._primary_ocr.crop_text_regions(image, result.boxes)
        crops, _classification = self._primary_ocr.cls_and_rotate(crops)
        prepared = self._prepare_recognition_crops(crops)
        crop_cls_s = time.perf_counter() - crop_started

        secondary_started = time.perf_counter()
        secondary = self._recognize(
            self._secondary_rec, prepared, len(result.txts)
        )
        secondary_s = time.perf_counter() - secondary_started
        numeric_indexes = [
            index
            for index, primary_text in enumerate(result.txts)
            if self._numeric_signature(primary_text)
            or self._numeric_signature(secondary[index][0])
        ]
        medium_indexes = [
            index
            for index in numeric_indexes
            if self._numeric_signature(result.txts[index])
            == self._numeric_signature(secondary[index][0])
        ]
        medium_prepared = self._prepare_recognition_crops(
            [crops[index] for index in medium_indexes],
            result_indexes=medium_indexes,
        )
        medium_started = time.perf_counter()
        medium = self._recognize(
            self._medium_rec, medium_prepared, len(result.txts)
        ) if medium_prepared else [("", 0.0)] * len(result.txts)
        medium_s = time.perf_counter() - medium_started

        reconciled = list(result.txts)
        disputed = 0
        for index in numeric_indexes:
            primary_signature = self._numeric_signature(result.txts[index])
            secondary_signature = self._numeric_signature(secondary[index][0])
            medium_signature = self._numeric_signature(medium[index][0])
            if (
                index in medium_indexes
                and primary_signature == secondary_signature == medium_signature
            ):
                reconciled[index] = self._repair_text(result.txts[index])
            else:
                reconciled[index] = self._suppress_numeric(result.txts[index])
                disputed += 1
        result.txts = tuple(reconciled)
        result.numeric_consensus_applied = True
        logger.info(
            "TensorRT recognition consensus finished | lines=%s "
            "numeric_lines=%s medium_lines=%s disputed_lines=%s "
            "primary_cpu_s=%.3f crop_cls_s=%.3f secondary_trt_s=%.3f "
            "medium_trt_s=%.3f total_s=%.3f",
            len(result.txts),
            len(numeric_indexes),
            len(medium_indexes),
            disputed,
            primary_s,
            crop_cls_s,
            secondary_s,
            medium_s,
            time.perf_counter() - started,
        )
        return result

    def warm_up(self) -> None:
        for session in (self._secondary_rec, self._medium_rec):
            shape = session.optimization_shape
            session.run_normalized_batch(
                self._np.zeros(
                    (shape[0], shape[2], shape[3], 3), dtype=self._np.float32
                ),
                shape,
            )


class TensorRTMediumRecognitionConsensus(TensorRTRecognitionConsensus):
    """Use Medium for all text and two Small heads as numeric witnesses.

    Detection and angle classification stay on the established CPU Small
    frontend.  This avoids the crop/layout drift observed when replacing the
    complete OCR pipeline, while moving all expensive recognition work to the
    GPU.  Medium is allowed to improve ordinary text, but a number is emitted
    only when both Small witnesses produce the identical numeric signature.
    """

    def __init__(
        self,
        root: Path,
        *,
        primary_ocr,
        device_id: int = 0,
        recognition_preprocess: str = "standard-resize",
        witness_preprocess: str = "standard-resize",
        numeric_signature: Callable[[object], tuple[str, ...]],
        suppress_numeric: Callable[[object], str],
        repair_text: Callable[[object], str],
    ):
        root = Path(root)
        if not is_complete_tensorrt_medium_primary_bundle(root):
            raise FileNotFoundError(
                f"incomplete PP-OCRv6 TensorRT Medium-primary bundle: {root}"
            )
        super().__init__(
            root,
            primary_ocr=primary_ocr,
            device_id=device_id,
            recognition_preprocess=recognition_preprocess,
            numeric_signature=numeric_signature,
            suppress_numeric=suppress_numeric,
            repair_text=repair_text,
        )
        self._witness_preprocess = self._validate_recognition_preprocess(
            witness_preprocess
        )
        try:
            self._primary_rec = self._new_session(
                root / "primary-rec.engine", device_id
            )
        except Exception:
            self.close()
            raise

    def __call__(self, image) -> TensorRTOCRResult:
        started = time.perf_counter()
        image = self._np.ascontiguousarray(image, dtype=self._np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"TensorRT OCR expects HWC uint8 RGB/BGR, got {image.shape}"
            )

        detect_started = time.perf_counter()
        detected = self._primary_ocr(
            image,
            use_det=True,
            use_cls=False,
            use_rec=False,
        )
        detect_s = time.perf_counter() - detect_started
        boxes = getattr(detected, "boxes", None)
        if boxes is None or len(boxes) == 0:
            # Preserve RapidOCR's established empty-result contract instead of
            # manufacturing a partially populated result object.
            return self._primary_ocr(
                image,
                use_det=True,
                use_cls=True,
                use_rec=True,
            )

        crop_started = time.perf_counter()
        crops = self._primary_ocr.crop_text_regions(image, boxes)
        crops, _classification = self._primary_ocr.cls_and_rotate(crops)
        prepared = self._prepare_recognition_crops(crops)
        crop_cls_s = time.perf_counter() - crop_started

        medium_started = time.perf_counter()
        medium = self._recognize(self._medium_rec, prepared, len(boxes))
        medium_s = time.perf_counter() - medium_started
        text_score = float(getattr(self._primary_ocr, "text_score", 0.5))
        kept_indexes = [
            index
            for index, (text, score) in enumerate(medium)
            if text.strip() and score >= text_score
        ]
        numeric_indexes = [
            index
            for index in kept_indexes
            if self._numeric_signature(medium[index][0])
        ]

        witness_prepared = self._prepare_recognition_crops(
            [crops[index] for index in numeric_indexes],
            result_indexes=numeric_indexes,
            preprocess_mode=getattr(
                self, "_witness_preprocess", "standard-resize"
            ),
        )
        witness_started = time.perf_counter()
        if witness_prepared:
            primary_small = self._recognize(
                self._primary_rec, witness_prepared, len(boxes)
            )
            secondary_small = self._recognize(
                self._secondary_rec, witness_prepared, len(boxes)
            )
        else:
            primary_small = [("", 0.0)] * len(boxes)
            secondary_small = [("", 0.0)] * len(boxes)
        witness_s = time.perf_counter() - witness_started

        reconciled: dict[int, str] = {}
        disputed = 0
        for index in numeric_indexes:
            medium_signature = self._numeric_signature(medium[index][0])
            primary_signature = self._numeric_signature(primary_small[index][0])
            secondary_signature = self._numeric_signature(
                secondary_small[index][0]
            )
            if medium_signature == primary_signature == secondary_signature:
                reconciled[index] = self._repair_text(medium[index][0])
            else:
                reconciled[index] = self._suppress_numeric(medium[index][0])
                disputed += 1

        final_boxes = []
        final_texts: list[str] = []
        final_scores: list[float] = []
        for index in kept_indexes:
            text = reconciled.get(index, medium[index][0]).strip()
            if not text:
                continue
            final_boxes.append(boxes[index])
            final_texts.append(text)
            final_scores.append(float(medium[index][1]))

        logger.info(
            "TensorRT Medium recognition consensus finished | lines=%s "
            "numeric_lines=%s disputed_lines=%s det_cpu_s=%.3f "
            "crop_cls_cpu_s=%.3f medium_trt_s=%.3f witnesses_trt_s=%.3f "
            "total_s=%.3f preprocess=%s witness_preprocess=%s "
            "native_height_lines=%s",
            len(final_texts),
            len(numeric_indexes),
            disputed,
            detect_s,
            crop_cls_s,
            medium_s,
            witness_s,
            time.perf_counter() - started,
            getattr(
                self, "_recognition_preprocess", "standard-resize"
            ),
            getattr(self, "_witness_preprocess", "standard-resize"),
            sum(
                1
                for crop in crops
                if crop.shape[0] <= 48 and crop.shape[1] <= 2048
            ),
        )
        return TensorRTOCRResult(
            boxes=self._np.asarray(final_boxes, dtype=self._np.float32),
            txts=tuple(final_texts),
            scores=tuple(final_scores),
        )

    def warm_up(self) -> None:
        for session in (
            self._primary_rec,
            self._secondary_rec,
            self._medium_rec,
        ):
            shape = session.optimization_shape
            session.run_normalized_batch(
                self._np.zeros(
                    (shape[0], shape[2], shape[3], 3), dtype=self._np.float32
                ),
                shape,
            )
