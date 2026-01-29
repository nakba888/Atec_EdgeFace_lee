"""
YuNet Face Detector with DeepX NPU Support
DeepX NPU를 사용한 YuNet 얼굴 검출기
"""

import os

try:
    import cv2 as cv
except ImportError:
    print("OpenCV가 설치되지 않았습니다. pip install opencv-python을 실행하세요.")
    cv = None

try:
    import numpy as np
except ImportError:
    print("NumPy가 설치되지 않았습니다. pip install numpy를 실행하세요.")
    np = None

try:
    from PIL import Image
except ImportError:
    print("PIL이 설치되지 않았습니다. pip install Pillow를 실행하세요.")
    Image = None

# Check if we should even try to import dx_engine
# Skip on Jetson (TensorRT environment) or if explicitly disabled
_SKIP_NPU = os.environ.get('SKIP_NPU', '0') == '1'
_IS_JETSON = os.path.exists('/etc/nv_tegra_release')

DXENGINE_AVAILABLE = False
InferenceEngine = None

if not _SKIP_NPU and not _IS_JETSON:
    try:
        from dx_engine import InferenceEngine
        DXENGINE_AVAILABLE = True
    except ImportError:
        print("dx_engine을 가져올 수 없습니다. DeepX NPU SDK가 설치되어 있는지 확인하세요.")
    except Exception as e:
        print(f"dx_engine import 중 오류 발생: {e}")


import sys
import os

# MTCNN의 align_trans 모듈을 재사용
sys.path.insert(0, os.path.dirname(__file__))
try:
    from mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points
except ImportError:
    print("MTCNN align_trans 모듈을 가져올 수 없습니다.")
    warp_and_crop_face = None
    get_reference_facial_points = None


class YuNetNPUDetector:
    """YuNet face detector using DeepX NPU"""

    def __init__(self, model_path, device='npu', crop_size=(112, 112)):
        """
        Initialize YuNet detector with DeepX NPU

        Args:
            model_path: Path to YuNet DXNN model (e.g., face_detection_yunet_2023mar.dxnn)
            device: 'npu' (only npu is supported)
            crop_size: Output face crop size (width, height)
        """
        # 필수 라이브러리 확인
        if cv is None:
            raise ImportError("OpenCV가 설치되지 않았습니다.")
        if np is None:
            raise ImportError("NumPy가 설치되지 않았습니다.")
        if Image is None:
            raise ImportError("PIL이 설치되지 않았습니다.")
        if warp_and_crop_face is None:
            raise ImportError("MTCNN align_trans 모듈을 사용할 수 없습니다.")
        if not DXENGINE_AVAILABLE:
            raise ImportError("dx_engine을 사용할 수 없습니다. DeepX NPU SDK를 설치하세요.")

        self.device = device
        self.crop_size = crop_size
        self.model_path = model_path

        # Input size for YuNet (from calibration config)
        self.input_size = (320, 320)

        # Detection thresholds
        # Lower threshold for NPU due to quantization effects
        # NPU quantization reduces confidence scores, especially for distant/small faces
        self.score_threshold = 0.5  # Tuned: 0.4 provides good balance (distant detection vs false positives)
        self.nms_threshold = 0.2    # Keeps duplicate removal effective

        # Initialize DeepX NPU Inference Engine
        try:
            print(f"YuNet NPU: Loading model from {model_path}...")
            self.inference_engine = InferenceEngine(model_path)
            print(f"YuNet NPU: Model loaded successfully")
            print(f"YuNet NPU: Input size: {self.inference_engine.input_size()}")
            print(f"YuNet NPU: Output dtype: {self.inference_engine.output_dtype()}")
        except Exception as e:
            raise RuntimeError(f"YuNet NPU detector 초기화 실패: {e}")

    def _preprocess_image(self, cv_image):
        """
        Preprocess image for YuNet NPU inference

        Args:
            cv_image: OpenCV BGR image

        Returns:
            preprocessed: Preprocessed image tensor ready for NPU
            scale_x, scale_y: Scale factors for coordinate conversion
        """
        orig_h, orig_w = cv_image.shape[:2]

        # Resize to input size (640x640 as per test_npu_inference.py)
        # Updated from 320x320 to match actual model input
        self.input_size = (640, 640)
        resized = cv.resize(cv_image, self.input_size)

        # Convert BGR to RGB (as per test_npu_inference.py)
        rgb_image = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

        # YuNet expects HWC format, not CHW! (as per test_npu_inference.py)
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        input_tensor = np.expand_dims(rgb_image, axis=0)

        # Calculate scale factors for coordinate conversion
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]

        return input_tensor.astype(np.uint8), scale_x, scale_y

    def _decode_outputs(self, outputs, scale_x, scale_y):
        """
        Decode YuNet NPU outputs to face detections

        YuNet outputs 13 tensors from 3 scales (FPN):
        - Scale 1 (80x80): outputs 1, 3, 10 (landmarks, cls, loc)
        - Scale 2 (40x40): outputs 6, 8, 12 (landmarks, cls, loc)
        - Scale 3 (20x20): outputs 7, 9, 11 (landmarks, cls, loc)

        Args:
            outputs: List of output tensors from NPU (each may be wrapped in a list)
            scale_x, scale_y: Scale factors for coordinate conversion

        Returns:
            faces: List of detected faces [x, y, w, h, x1, y1, ..., x5, y5, confidence]
        """
        faces = []

        if len(outputs) == 0:
            return faces

        # Unwrap list wrappers from all outputs
        unwrapped_outputs = []
        for out in outputs:
            if isinstance(out, list) and len(out) > 0:
                unwrapped_outputs.append(out[0] if isinstance(out[0], np.ndarray) else out)
            elif isinstance(out, np.ndarray):
                unwrapped_outputs.append(out)
            else:
                unwrapped_outputs.append(out)

        # Debug output count (disabled for performance)
        # print(f"[DEBUG] YuNet NPU outputs count: {len(unwrapped_outputs)}")

        if len(unwrapped_outputs) != 13:
            print(f"[WARNING] Expected 13 outputs, got {len(unwrapped_outputs)}")
            return faces

        # Extract outputs for each scale based on ONNX model structure
        # ONNX outputs: cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
        #
        # NPU outputs (your compiled version):
        # Output 0: (1, 1, 80, 80) - spatial feature map (unused)
        # Output 1: (1, 6400, 1) - cls_8 (0-0.207)
        # Output 2: (1, 1600, 1) - cls_16 (0-0.0027, very low)
        # Output 3: (1, 6400, 4) - bbox_8
        # Output 4: (1, 1600, 4) - bbox_16
        # Output 5: (1, 1600, 1) - obj_16 (0.37-0.63)
        # Output 6: (1, 400, 10) - kps_32
        # Output 7: (1, 6400, 10) - kps_8
        # Output 8: (1, 1600, 10) - kps_16
        # Output 9: (1, 400, 1) - obj_32 (0.37-0.72)
        # Output 10: (1, 400, 4) - bbox_32
        # Output 11: (1, 6400, 1) - obj_8 (0.38-0.89, objectness)
        # Output 12: (1, 400, 1) - cls_32 (0-0.92)
        #
        # Final confidence = cls * obj

        try:
            # Process all 3 scales
            all_detections = []

            # Scale 1: stride 8 (80x80 feature map, 6400 anchors)
            cls_8 = unwrapped_outputs[1].squeeze()    # (6400, 1) -> (6400,)
            obj_8 = unwrapped_outputs[11].squeeze()   # (6400, 1) -> (6400,)
            bbox_8 = unwrapped_outputs[3].squeeze()   # (6400, 4)
            kps_8 = unwrapped_outputs[7].squeeze()    # (6400, 10)

            # Combine cls and obj scores
            score_8 = cls_8 * obj_8
            scale_detections = self._process_scale(score_8, bbox_8, kps_8, stride=8, input_size=self.input_size[0])
            all_detections.extend(scale_detections)

            # Scale 2: stride 16 (40x40 feature map, 1600 anchors)
            cls_16 = unwrapped_outputs[2].squeeze()   # (1600, 1) -> (1600,)
            obj_16 = unwrapped_outputs[5].squeeze()   # (1600, 1) -> (1600,)
            bbox_16 = unwrapped_outputs[4].squeeze()  # (1600, 4)
            kps_16 = unwrapped_outputs[8].squeeze()   # (1600, 10)

            score_16 = cls_16 * obj_16
            scale_detections = self._process_scale(score_16, bbox_16, kps_16, stride=16, input_size=self.input_size[0])
            all_detections.extend(scale_detections)

            # Scale 3: stride 32 (20x20 feature map, 400 anchors)
            cls_32 = unwrapped_outputs[12].squeeze()  # (400, 1) -> (400,)
            obj_32 = unwrapped_outputs[9].squeeze()   # (400, 1) -> (400,)
            bbox_32 = unwrapped_outputs[10].squeeze() # (400, 4)
            kps_32 = unwrapped_outputs[6].squeeze()   # (400, 10)

            score_32 = cls_32 * obj_32
            scale_detections = self._process_scale(score_32, bbox_32, kps_32, stride=32, input_size=self.input_size[0])
            all_detections.extend(scale_detections)

            # Apply NMS across all scales
            if len(all_detections) > 0:
                # Scale coordinates back to original image size
                for detection in all_detections:
                    detection[0] *= scale_x  # x
                    detection[1] *= scale_y  # y
                    detection[2] *= scale_x  # w
                    detection[3] *= scale_y  # h
                    for i in range(4, 14, 2):
                        detection[i] *= scale_x      # landmark x
                        detection[i+1] *= scale_y    # landmark y

                faces = self._apply_nms(all_detections, self.nms_threshold)

        except Exception as e:
            print(f"[ERROR] Failed to decode YuNet outputs: {e}")
            import traceback
            traceback.print_exc()
            return faces

        return faces

    def _process_scale(self, cls_scores, bboxes, landmarks, stride, input_size):
        """
        Process detections from a single FPN scale

        Args:
            cls_scores: Classification scores (N,) or (N, 1)
            bboxes: Bounding boxes (N, 4) - format depends on YuNet encoding
            landmarks: Facial landmarks (N, 10)
            stride: Feature map stride (8, 16, or 32)
            input_size: Model input size (e.g., 640)

        Returns:
            List of detections [x, y, w, h, x1, y1, ..., x5, y5, confidence]
        """
        detections = []

        # Ensure correct shapes
        if cls_scores.ndim > 1:
            cls_scores = cls_scores.flatten()
        if bboxes.ndim == 3:
            bboxes = bboxes.squeeze(0)
        if landmarks.ndim == 3:
            landmarks = landmarks.squeeze(0)

        # Calculate feature map size
        feat_size = input_size // stride

        # Filter by confidence threshold
        valid_mask = cls_scores >= self.score_threshold
        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            confidence = float(cls_scores[idx])

            # Decode bounding box from anchor
            bbox = bboxes[idx]

            # Calculate anchor position (grid top-left corner, no 0.5 offset)
            anchor_y = (idx // feat_size)
            anchor_x = (idx % feat_size)

            # YuNet bbox encoding for NPU outputs
            # Bbox offsets are relative to anchor position
            # Width/height use linear scaling with prior size

            # Decode center with direct offset (no scaling)
            cx = (anchor_x + bbox[0] * 1.0) * stride
            cy = (anchor_y + bbox[1] * 1.0) * stride

            # Decode size with aspect ratio (face is typically taller than wide)
            prior_w = stride * 3
            prior_h = stride * 3.6  # 1:1.2 aspect ratio for faces
            w = bbox[2] * prior_w
            h = bbox[3] * prior_h

            # Convert to top-left corner format
            x = cx - w / 2
            y = cy - h / 2

            # Decode landmarks (10 values: 5 points * 2 coordinates)
            # Landmarks use same anchor-based offset as bbox center
            lms = landmarks[idx]
            decoded_lms = []
            for i in range(5):
                lm_x = (anchor_x + lms[i*2] * 1) * stride
                lm_y = (anchor_y + lms[i*2 + 1] * 1) * stride
                decoded_lms.extend([lm_x, lm_y])

            # Build detection: [x, y, w, h, x1, y1, ..., x5, y5, confidence]
            detection = [x, y, w, h] + decoded_lms + [confidence]
            detections.append(detection)

        return detections

    def _apply_nms(self, faces, nms_threshold):
        """
        Apply Non-Maximum Suppression to remove overlapping detections

        Args:
            faces: List of face detections
            nms_threshold: IoU threshold for NMS

        Returns:
            Filtered list of faces
        """
        if len(faces) == 0:
            return faces

        # Convert to numpy array for easier processing
        faces_array = np.array(faces)

        # Extract bounding boxes and scores
        x = faces_array[:, 0]
        y = faces_array[:, 1]
        w = faces_array[:, 2]
        h = faces_array[:, 3]
        scores = faces_array[:, 14]

        # Calculate areas
        x2 = x + w
        y2 = y + h
        areas = w * h

        # Sort by confidence (descending)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            inter = w_inter * h_inter

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]

        return [faces[i] for i in keep]

    def detect_faces(self, image):
        """
        Detect faces and landmarks using YuNet NPU

        Args:
            image: PIL Image or numpy array

        Returns:
            faces: list of detected faces with landmarks
                   each face: [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, confidence]
        """
        if isinstance(image, Image.Image):
            cv_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        else:
            cv_image = image

        # Preprocess image
        input_tensor, scale_x, scale_y = self._preprocess_image(cv_image)

        # Make input contiguous to avoid NPU warning
        input_tensor = np.ascontiguousarray(input_tensor)

        # Run inference on NPU using run() + get_all_task_outputs() pattern
        try:
            self.inference_engine.run(input_tensor)
            outputs = self.inference_engine.get_all_task_outputs()
        except Exception as e:
            print(f"YuNet NPU inference error: {e}")
            import traceback
            traceback.print_exc()
            return []

        # Decode outputs to face detections
        faces = self._decode_outputs(outputs, scale_x, scale_y)

        return faces

    def align(self, img, return_landmarks=False):
        """
        Detect and align face using YuNet NPU (compatible with MTCNN interface)

        Args:
            img: PIL Image
            return_landmarks: whether to return landmarks

        Returns:
            aligned_face: PIL Image of aligned face or None if no face detected
            landmarks: facial landmarks if return_landmarks=True
        """
        faces = self.detect_faces(img)

        if len(faces) == 0:
            return None if not return_landmarks else (None, None)

        # Select the best face (highest confidence)
        best_face = max(faces, key=lambda x: x[-1])

        # Extract landmarks from the best face
        landmarks = []
        for i in range(5):
            x = best_face[4 + i * 2]
            y = best_face[4 + i * 2 + 1]
            landmarks.append([x, y])

        best_landmarks = np.array(landmarks, dtype=np.float32)

        # Apply landmark stabilization
        facial_pts = self._stabilize_landmarks(best_landmarks)

        # Align face using the same transformation as MTCNN
        try:
            ref_pts = get_reference_facial_points(
                output_size=self.crop_size,
                inner_padding_factor=0,
                outer_padding=(0, 0),
                default_square=(self.crop_size[0] == self.crop_size[1])
            )

            aligned_face = warp_and_crop_face(
                np.array(img),
                facial_pts,
                reference_pts=ref_pts,
                crop_size=self.crop_size,
                align_type='similarity'
            )
        except Exception as e:
            try:
                # Fallback: use affine transformation
                aligned_face = warp_and_crop_face(
                    np.array(img),
                    facial_pts,
                    reference_pts=ref_pts,
                    crop_size=self.crop_size,
                    align_type='affine'
                )
            except Exception as e2:
                # Final fallback: use basic crop
                face = best_face
                x, y, w, h = face[:4]
                x, y, w, h = int(x), int(y), int(w), int(h)

                padding = int(min(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img.size[0], x + w + padding)
                y2 = min(img.size[1], y + h + padding)

                cropped = img.crop((x1, y1, x2, y2))
                aligned_face = np.array(cropped.resize(self.crop_size))

        aligned_face_pil = Image.fromarray(aligned_face)

        if return_landmarks:
            return aligned_face_pil, facial_pts.tolist()
        else:
            return aligned_face_pil

    def _stabilize_landmarks(self, landmarks):
        """
        Apply landmark stabilization for better alignment quality
        (Same as yunet.py implementation)
        """
        stabilized = landmarks.copy()

        # 1. Basic sanity check: ensure left eye is to the left of right eye
        if stabilized[0, 0] > stabilized[1, 0]:
            stabilized[[0, 1]] = stabilized[[1, 0]]

        # 2. Validate landmark positions
        face_center_x = np.mean(stabilized[:, 0])
        face_center_y = np.mean(stabilized[:, 1])

        eye_distance = np.linalg.norm(stabilized[1] - stabilized[0])
        face_size = eye_distance * 3.0

        # 3. Validate each landmark distance from center
        for i in range(5):
            dist_from_center = np.linalg.norm(stabilized[i] - [face_center_x, face_center_y])
            if dist_from_center > face_size:
                direction = (stabilized[i] - [face_center_x, face_center_y]) / dist_from_center
                stabilized[i] = [face_center_x, face_center_y] + direction * (face_size * 0.8)

        # 4. Ensure nose is centered
        expected_nose_x = (stabilized[0, 0] + stabilized[1, 0]) / 2
        nose_x_diff = abs(stabilized[2, 0] - expected_nose_x)
        if nose_x_diff > eye_distance * 0.3:
            stabilized[2, 0] = expected_nose_x + np.sign(stabilized[2, 0] - expected_nose_x) * eye_distance * 0.3

        # 5. Ensure mouth landmarks are below nose
        nose_y = stabilized[2, 1]
        for i in [3, 4]:
            if stabilized[i, 1] < nose_y:
                stabilized[i, 1] = nose_y + eye_distance * 0.3

        # 6. Ensure symmetric mouth position
        mouth_center_x = (stabilized[3, 0] + stabilized[4, 0]) / 2
        expected_mouth_x = expected_nose_x
        mouth_offset = mouth_center_x - expected_mouth_x
        if abs(mouth_offset) > eye_distance * 0.2:
            correction = np.sign(mouth_offset) * eye_distance * 0.2 - mouth_offset
            stabilized[3, 0] += correction / 2
            stabilized[4, 0] += correction / 2

        return stabilized

    def align_face(self, image, landmarks):
        """
        Align face using provided landmarks

        Args:
            image: PIL Image
            landmarks: Facial landmarks array [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]

        Returns:
            Aligned face image or None if alignment fails
        """
        # Convert flat landmarks to (5, 2) array
        if len(landmarks) == 10:
            facial_pts = np.array([[landmarks[i], landmarks[i+1]] for i in range(0, 10, 2)], dtype=np.float32)
        else:
            facial_pts = np.array(landmarks, dtype=np.float32)

        # Apply stabilization
        facial_pts = self._stabilize_landmarks(facial_pts)

        try:
            ref_pts = get_reference_facial_points(
                output_size=self.crop_size,
                inner_padding_factor=0,
                outer_padding=(0, 0),
                default_square=(self.crop_size[0] == self.crop_size[1])
            )

            aligned_face = warp_and_crop_face(
                np.array(image),
                facial_pts,
                reference_pts=ref_pts,
                crop_size=self.crop_size,
                align_type='similarity'
            )

            return Image.fromarray(aligned_face)
        except Exception as e:
            print(f"Face alignment failed: {e}")
            return None

    def align_multi(self, img, limit=None):
        """
        Detect and align multiple faces (compatible with MTCNN interface)

        Args:
            img: PIL Image
            limit: maximum number of faces to process

        Returns:
            bboxes: list of bounding boxes
            faces: list of aligned face PIL Images
        """
        detected_faces = self.detect_faces(img)

        if len(detected_faces) == 0:
            return [], []

        if limit is not None:
            detected_faces = detected_faces[:limit]

        bboxes = []
        aligned_faces = []

        for face in detected_faces:
            # Extract bounding box
            x, y, w, h = face[:4]
            bbox = [x, y, x + w, y + h, face[-1]]
            bboxes.append(bbox)

            # Extract landmarks
            landmarks = []
            for i in range(5):
                x_pt = face[4 + i * 2]
                y_pt = face[4 + i * 2 + 1]
                landmarks.append([x_pt, y_pt])

            # Align face
            facial_pts = np.array(landmarks, dtype=np.float32)
            facial_pts = self._stabilize_landmarks(facial_pts)

            try:
                ref_pts = get_reference_facial_points(
                    output_size=self.crop_size,
                    inner_padding_factor=0,
                    outer_padding=(0, 0),
                    default_square=True
                )
                aligned_face = warp_and_crop_face(
                    np.array(img),
                    facial_pts,
                    reference_pts=ref_pts,
                    crop_size=self.crop_size,
                    align_type='similarity'
                )
                aligned_faces.append(Image.fromarray(aligned_face))
            except Exception as e2:
                # Skip this face if alignment fails
                continue

        return bboxes, aligned_faces
