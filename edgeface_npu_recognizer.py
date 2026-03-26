"""
EdgeFace Face Recognition with DeepX NPU Support
DeepX NPU를 사용한 EdgeFace 얼굴 인식 모듈
"""

import os
import numpy as np
import cv2

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



class EdgeFaceNPURecognizer:
    """EdgeFace based face recognition module using DeepX NPU"""

    def __init__(self, model_path: str, model_name: str = 'edgeface_xs_gamma_06', device: str = 'npu'):
        """
        Initialize EdgeFace recognizer with DeepX NPU

        Args:
            model_path: Path to EdgeFace DXNN model (e.g., edgeface_xs_gamma_06.dxnn)
            model_name: Model architecture name (for compatibility)
            device: 'npu' (only npu is supported)
        """
        if not DXENGINE_AVAILABLE:
            raise ImportError("dx_engine을 사용할 수 없습니다. DeepX NPU SDK를 설치하세요.")

        self.device = device
        self.model_name = model_name
        self.model_path = model_path

        # Input size for EdgeFace (from calibration config)
        self.input_size = (112, 112)

        # Initialize DeepX NPU Inference Engine
        try:
            print(f"EdgeFace NPU: Loading model from {model_path}...")
            self.inference_engine = InferenceEngine(model_path)
            print(f"EdgeFace NPU: Model loaded successfully")
            print(f"EdgeFace NPU: Input size: {self.inference_engine.input_size()}")
            print(f"EdgeFace NPU: Output dtype: {self.inference_engine.output_dtype()}")
        except Exception as e:
            raise RuntimeError(f"EdgeFace NPU recognizer 초기화 실패: {e}")

        print(f"✅ EdgeFace NPU model loaded: {model_name} on {device}")

    def _preprocess_image(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for EdgeFace NPU inference

        Args:
            face_img: Aligned face image (112x112x3) in BGR format

        Returns:
            preprocessed: Preprocessed image tensor ready for NPU
        """
        # Resize if needed
        if face_img.shape[:2] != self.input_size:
            face_img = cv2.resize(face_img, self.input_size)

        # Convert BGR to RGB (as per calibration config)
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Convert to float and normalize to [0, 1] (as per calibration config: div 255.0)
        # img_float = rgb_img.astype(np.float32) / 255.0

        # Normalize with mean=0.5, std=0.5 (as per calibration config)
        # normalized = (img / 255.0 - 0.5) / 0.5
        # img_normalized = (img_float - 0.5) / 0.5

        # Transpose to CHW format (as per calibration config)
        # Shape: (H, W, C) -> (C, H, W)
        # chw_img = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        input_tensor = np.expand_dims(rgb_img, axis=0)

        # Convert to float32 for NPU
        return input_tensor.astype(np.uint8)

    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from aligned face image using NPU

        Args:
            face_img: Aligned face image (112x112x3) in BGR format

        Returns:
            Face embedding vector (512-d)
        """
        # Preprocess
        input_tensor = self._preprocess_image(face_img)

        # Make input contiguous to avoid NPU warning
        input_tensor = np.ascontiguousarray(input_tensor)

        # Run inference on NPU
        try:
            # Use run() + get_all_task_outputs() pattern as per test_npu_inference.py
            self.inference_engine.run(input_tensor)
            outputs = self.inference_engine.get_all_task_outputs()
        except Exception as e:
            print(f"EdgeFace NPU inference error: {e}")
            raise

        # Extract embedding from output
        # Based on test output: Output 1 is list with first element being (1, 512) tensor
        # outputs[1] is the embedding output (Output 1 from test)
        if len(outputs) < 2:
            raise RuntimeError(f"EdgeFace NPU: Expected at least 2 outputs, got {len(outputs)}")

        embedding_output = outputs[1]

        # Handle list wrapper: unwrap if it's a list with one element
        if isinstance(embedding_output, list):
            if len(embedding_output) == 0:
                raise RuntimeError("EdgeFace NPU: Embedding output list is empty")
            embedding_output = embedding_output[0]

        # Now embedding_output should be a numpy array with shape (1, 512)
        if not isinstance(embedding_output, np.ndarray):
            raise RuntimeError(f"EdgeFace NPU: Expected numpy array, got {type(embedding_output)}")

        # Flatten to 1D vector
        embedding = embedding_output.flatten()

        # Verify size
        if embedding.size != 512:
            raise RuntimeError(f"EdgeFace NPU: Expected 512-d embedding, got {embedding.size}-d")

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def extract_embedding_timed(self, face_img: np.ndarray) -> tuple:
        """
        Extract embedding and return pure inference time
        Returns: (embedding_vector, pure_inference_time_seconds)
        """
        import time
        input_tensor = self._preprocess_image(face_img)
        input_tensor = np.ascontiguousarray(input_tensor)

        try:
            start_time = time.perf_counter()
            self.inference_engine.run(input_tensor)
            outputs = self.inference_engine.get_all_task_outputs()
            pure_inference_time = time.perf_counter() - start_time
        except Exception as e:
            print(f"EdgeFace NPU inference error: {e}")
            raise

        if len(outputs) < 2:
            raise RuntimeError(f"Expected at least 2 outputs, got {len(outputs)}")

        embedding_output = outputs[1]
        if isinstance(embedding_output, list):
            embedding_output = embedding_output[0]
        
        embedding = embedding_output.flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding, pure_inference_time

    def extract_embeddings_batch(self, face_imgs: list) -> list:
        """
        Extract face embeddings from multiple aligned face images

        Note: DeepX NPU inference engine runs one image at a time,
              so we process each image sequentially

        Args:
            face_imgs: List of aligned face images (112x112x3) in BGR format

        Returns:
            List of face embedding vectors (512-d each)
        """
        if not face_imgs:
            return []

        embeddings = []
        for face_img in face_imgs:
            try:
                embedding = self.extract_embedding(face_img)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error extracting embedding: {e}")
                # Return zero embedding for failed cases
                embeddings.append(np.zeros(512, dtype=np.float32))

        return embeddings

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return float(np.dot(emb1, emb2))
