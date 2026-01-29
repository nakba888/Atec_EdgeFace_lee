#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeFace Face Recognition with TensorRT for Jetson Orin Nano
Jetson Orin Nano용 TensorRT EdgeFace 얼굴 인식 모듈

This module provides TensorRT-accelerated face recognition using EdgeFace model.
Compatible interface with EdgeFaceNPURecognizer for easy switching.

Usage:
    from edgeface_jetson_recognizer import EdgeFaceJetsonRecognizer
    
    recognizer = EdgeFaceJetsonRecognizer('checkpoints/edgeface_xs_gamma_06.trt')
    embedding = recognizer.extract_embedding(face_image)
"""

import os
import numpy as np
import cv2
import time
from typing import Optional, Tuple, List

# TensorRT imports
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_VERSION = trt.__version__
except ImportError:
    TRT_AVAILABLE = False
    TRT_VERSION = None
    print("⚠️ TensorRT를 가져올 수 없습니다. JetPack이 설치되어 있는지 확인하세요.")

# CUDA imports for memory management
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ PyCUDA를 가져올 수 없습니다. 'pip install pycuda' 로 설치하세요.")


class TensorRTEngine:
    """TensorRT Engine wrapper for inference"""
    
    def __init__(self, engine_path: str):
        """
        Load TensorRT engine from file
        
        Args:
            engine_path: Path to TensorRT engine file (.trt)
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install JetPack.")
        
        if not CUDA_AVAILABLE:
            raise ImportError("PyCUDA is not available. Please install pycuda.")
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get input/output info
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(shape)
                print(f"  Input: {name}, shape: {shape}")
            else:
                self.output_name = name
                self.output_shape = tuple(shape)
                print(f"  Output: {name}, shape: {shape}")
        
        # Allocate GPU memory
        self._allocate_buffers()
        
        print(f"✅ TensorRT engine loaded successfully")
    
    def _allocate_buffers(self):
        """Allocate GPU memory for input/output"""
        # Calculate buffer sizes
        input_size = int(np.prod(self.input_shape))
        output_size = int(np.prod(self.output_shape))
        
        # Allocate host memory
        self.h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
        
        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Create CUDA stream
        self.stream = cuda.Stream()
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data
        
        Args:
            input_data: Input tensor (must match engine input shape)
            
        Returns:
            Output tensor
        """
        # Ensure input is contiguous and correct dtype
        input_data = np.ascontiguousarray(input_data.astype(np.float32))
        
        # Copy input to host buffer
        np.copyto(self.h_input, input_data.ravel())
        
        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output to host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        # Reshape output
        return self.h_output.reshape(self.output_shape)
    
    def __del__(self):
        """Cleanup GPU resources"""
        try:
            if hasattr(self, 'd_input'):
                self.d_input.free()
            if hasattr(self, 'd_output'):
                self.d_output.free()
        except:
            pass


class EdgeFaceJetsonRecognizer:
    """EdgeFace based face recognition module using TensorRT on Jetson"""
    
    def __init__(self, model_path: str, model_name: str = 'edgeface_xs_gamma_06', device: str = 'jetson'):
        """
        Initialize EdgeFace recognizer with TensorRT
        
        Args:
            model_path: Path to EdgeFace TensorRT engine (e.g., edgeface_xs_gamma_06.trt)
            model_name: Model architecture name (for compatibility)
            device: 'jetson' (TensorRT), 'cuda' (fallback to PyTorch)
        """
        self.device = device
        self.model_name = model_name
        self.model_path = model_path
        
        # Input size for EdgeFace
        self.input_size = (112, 112)
        
        # Timing info
        self.last_inference_time = 0.0
        
        # Check if using TensorRT or PyTorch fallback
        if device == 'jetson' or model_path.endswith('.trt'):
            if not TRT_AVAILABLE:
                raise ImportError("TensorRT를 사용할 수 없습니다. JetPack을 설치하세요.")
            
            print(f"EdgeFace Jetson: Loading TensorRT engine from {model_path}...")
            self.engine = TensorRTEngine(model_path)
            self.use_tensorrt = True
            print(f"✅ EdgeFace TensorRT model loaded: {model_name}")
            
        else:
            # Fallback to PyTorch (for comparison)
            print(f"EdgeFace Jetson: Loading PyTorch model from {model_path}...")
            self._load_pytorch_model(model_path, model_name)
            self.use_tensorrt = False
            print(f"✅ EdgeFace PyTorch model loaded: {model_name}")
    
    def _load_pytorch_model(self, model_path: str, model_name: str):
        """Load PyTorch model as fallback"""
        import torch
        from backbones import get_model
        
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = get_model(model_name, fp16=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.torch_device))
        self.model.to(self.torch_device)
        self.model.eval()
    
    def _preprocess_image(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for EdgeFace inference
        
        Args:
            face_img: Aligned face image (112x112x3) in BGR format
            
        Returns:
            preprocessed: Preprocessed image tensor (1, 3, 112, 112)
        """
        # Resize if needed
        if face_img.shape[:2] != self.input_size:
            face_img = cv2.resize(face_img, self.input_size)
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1
        img_float = rgb_img.astype(np.float32) / 255.0
        img_normalized = (img_float - 0.5) / 0.5
        
        # Transpose to CHW format: (H, W, C) -> (C, H, W)
        chw_img = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        input_tensor = np.expand_dims(chw_img, axis=0)
        
        return input_tensor.astype(np.float32)
    
    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from aligned face image using TensorRT
        
        Args:
            face_img: Aligned face image (112x112x3) in BGR format
            
        Returns:
            Face embedding vector (512-d)
        """
        # Preprocess
        input_tensor = self._preprocess_image(face_img)
        
        # Record inference time
        start_time = time.time()
        
        try:
            if self.use_tensorrt:
                # TensorRT inference
                output = self.engine.infer(input_tensor)
                embedding = output.flatten()
            else:
                # PyTorch inference (fallback)
                import torch
                with torch.no_grad():
                    tensor = torch.from_numpy(input_tensor).to(self.torch_device)
                    output = self.model(tensor)
                    embedding = output.cpu().numpy().flatten()
            
            self.last_inference_time = (time.time() - start_time) * 1000  # ms
            
        except Exception as e:
            print(f"EdgeFace inference error: {e}")
            raise
        
        # Verify embedding size
        if embedding.size != 512:
            raise RuntimeError(f"EdgeFace: Expected 512-d embedding, got {embedding.size}-d")
        
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def extract_embeddings_batch(self, face_imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract face embeddings from multiple aligned face images
        
        Note: Current implementation processes images sequentially.
              For batch processing with TensorRT, engine needs to support dynamic batch.
        
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
    
    def get_inference_time(self) -> float:
        """Get last inference time in milliseconds"""
        return self.last_inference_time
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return float(np.dot(emb1, emb2))


def test_recognizer():
    """Test EdgeFaceJetsonRecognizer"""
    import sys
    
    # Check for TensorRT engine
    trt_engine_path = "checkpoints/edgeface_xs_gamma_06.trt"
    pytorch_model_path = "checkpoints/edgeface_xs_gamma_06.pt"
    
    # Create dummy test image
    test_image = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    
    print("=" * 60)
    print("EdgeFace Jetson Recognizer Test")
    print("=" * 60)
    
    # Test TensorRT if available
    if os.path.exists(trt_engine_path):
        print(f"\n[TensorRT] Testing with {trt_engine_path}")
        try:
            recognizer_trt = EdgeFaceJetsonRecognizer(trt_engine_path)
            
            # Warmup
            for _ in range(5):
                _ = recognizer_trt.extract_embedding(test_image)
            
            # Benchmark
            times = []
            for _ in range(50):
                start = time.time()
                emb = recognizer_trt.extract_embedding(test_image)
                times.append((time.time() - start) * 1000)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"  ✅ TensorRT inference successful")
            print(f"  Embedding shape: {emb.shape}")
            print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")
            print(f"  Avg inference time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"  Throughput: {1000/avg_time:.1f} FPS")
            
        except Exception as e:
            print(f"  ❌ TensorRT test failed: {e}")
    else:
        print(f"\n[TensorRT] Engine not found: {trt_engine_path}")
        print("  Run: python onnx_to_tensorrt.py --input checkpoints/edgeface_xs_gamma_06.onnx")
    
    # Test PyTorch if available
    if os.path.exists(pytorch_model_path):
        print(f"\n[PyTorch] Testing with {pytorch_model_path}")
        try:
            recognizer_pt = EdgeFaceJetsonRecognizer(pytorch_model_path, device='cuda')
            
            # Warmup
            for _ in range(5):
                _ = recognizer_pt.extract_embedding(test_image)
            
            # Benchmark
            times = []
            for _ in range(50):
                start = time.time()
                emb = recognizer_pt.extract_embedding(test_image)
                times.append((time.time() - start) * 1000)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"  ✅ PyTorch inference successful")
            print(f"  Embedding shape: {emb.shape}")
            print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")
            print(f"  Avg inference time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"  Throughput: {1000/avg_time:.1f} FPS")
            
        except Exception as e:
            print(f"  ❌ PyTorch test failed: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_recognizer()
