#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Face Recognition System for Jetson Orin Nano
Jetson Orin Nano용 실시간 얼굴 인식 시스템

Features:
- TensorRT accelerated EdgeFace inference
- PyTorch fallback mode for comparison
- Real-time performance monitoring
- Compatible with existing reference database
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

# Import base components from original system
from face_recognition_system import (
    FaceAngleCalculator,
    ReferenceDatabase,
    EdgeFaceRecognizer
)
from face_alignment.unified_detector import UnifiedFaceDetector

# Import TensorRT recognizer
try:
    from edgeface_jetson_recognizer import EdgeFaceJetsonRecognizer, TRT_AVAILABLE
    JETSON_AVAILABLE = TRT_AVAILABLE
except ImportError:
    JETSON_AVAILABLE = False
    EdgeFaceJetsonRecognizer = None
    print("⚠️ EdgeFaceJetsonRecognizer를 가져올 수 없습니다.")

# Import PyTorch for fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PerformanceMonitor:
    """Real-time performance monitoring for Jetson"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.detection_times = []
        self.recognition_times = []
        self.total_times = []
        self.fps_history = []
        
        self.max_history = 100  # Keep last 100 measurements
        
        # Jetson-specific monitoring
        self.gpu_memory = 0
        self.temperature = 0
        self.power = 0
        
        # Try to import jtop for Jetson stats
        try:
            from jtop import jtop
            self.jtop = jtop()
            self.jtop_available = True
        except ImportError:
            self.jtop = None
            self.jtop_available = False
    
    def update(self, detection_time: float, recognition_time: float):
        """
        Update performance metrics
        
        Args:
            detection_time: Face detection time in seconds
            recognition_time: Face recognition time in seconds
        """
        total_time = detection_time + recognition_time
        
        self.detection_times.append(detection_time)
        self.recognition_times.append(recognition_time)
        self.total_times.append(total_time)
        
        # Keep history limited
        if len(self.detection_times) > self.max_history:
            self.detection_times.pop(0)
            self.recognition_times.pop(0)
            self.total_times.pop(0)
        
        # Update FPS
        if total_time > 0:
            self.fps_history.append(1.0 / total_time)
            if len(self.fps_history) > self.max_history:
                self.fps_history.pop(0)
        
        # Update Jetson stats
        self._update_jetson_stats()
    
    def _update_jetson_stats(self):
        """Update Jetson-specific statistics"""
        if self.jtop_available and self.jtop:
            try:
                with self.jtop as jetson:
                    self.gpu_memory = jetson.memory.get('GPU', {}).get('used', 0)
                    # Get GPU temperature
                    temps = jetson.temperature
                    self.temperature = temps.get('GPU', temps.get('gpu', 0))
                    # Get power consumption
                    self.power = jetson.power.get('total', {}).get('power', 0)
            except:
                pass
        else:
            # Try to read from sysfs directly
            try:
                # GPU memory from tegrastats
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    self.temperature = int(f.read().strip()) / 1000.0
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        if not self.total_times:
            return {
                'fps': 0.0,
                'detection_ms': 0.0,
                'recognition_ms': 0.0,
                'total_ms': 0.0,
                'gpu_memory_mb': self.gpu_memory,
                'temperature_c': self.temperature,
                'power_w': self.power
            }
        
        return {
            'fps': np.mean(self.fps_history[-30:]) if self.fps_history else 0.0,
            'detection_ms': np.mean(self.detection_times[-30:]) * 1000,
            'recognition_ms': np.mean(self.recognition_times[-30:]) * 1000,
            'total_ms': np.mean(self.total_times[-30:]) * 1000,
            'gpu_memory_mb': self.gpu_memory,
            'temperature_c': self.temperature,
            'power_w': self.power
        }
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed performance statistics for benchmarking"""
        if not self.total_times:
            return {}
        
        return {
            'fps': {
                'mean': np.mean(self.fps_history),
                'std': np.std(self.fps_history),
                'min': np.min(self.fps_history),
                'max': np.max(self.fps_history)
            },
            'detection_ms': {
                'mean': np.mean(self.detection_times) * 1000,
                'std': np.std(self.detection_times) * 1000,
                'min': np.min(self.detection_times) * 1000,
                'max': np.max(self.detection_times) * 1000
            },
            'recognition_ms': {
                'mean': np.mean(self.recognition_times) * 1000,
                'std': np.std(self.recognition_times) * 1000,
                'min': np.min(self.recognition_times) * 1000,
                'max': np.max(self.recognition_times) * 1000
            },
            'total_ms': {
                'mean': np.mean(self.total_times) * 1000,
                'std': np.std(self.total_times) * 1000,
                'min': np.min(self.total_times) * 1000,
                'max': np.max(self.total_times) * 1000
            }
        }


class FaceRecognitionJetsonSystem:
    """Face Recognition System optimized for Jetson Orin Nano"""
    
    def __init__(
        self,
        detector_method: str = 'yunet',
        edgeface_model_path: str = 'checkpoints/edgeface_xs_gamma_06.trt',
        edgeface_model_name: str = 'edgeface_xs_gamma_06',
        device: str = 'jetson',
        similarity_threshold: float = 0.5,
        use_tensorrt: bool = True,
        fp16: bool = True
    ):
        """
        Initialize Jetson face recognition system
        
        Args:
            detector_method: Face detection method ('mtcnn', 'yunet', 'yolov5_face', 'yolov8')
            edgeface_model_path: Path to EdgeFace model (.trt for TensorRT, .pt for PyTorch)
            edgeface_model_name: EdgeFace model architecture name
            device: 'jetson' (TensorRT), 'cuda' (PyTorch GPU), 'cpu' (PyTorch CPU)
            similarity_threshold: Minimum similarity for recognition
            use_tensorrt: Use TensorRT acceleration (requires .trt engine)
            fp16: Use FP16 precision (for PyTorch fallback)
        """
        print("🚀 Initializing Jetson Face Recognition System...")
        
        self.use_tensorrt = use_tensorrt
        self.device = device
        self.fp16 = fp16
        
        # Performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        # Initialize face detector
        print(f"📷 Initializing face detector: {detector_method}")
        detector_device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.detector = UnifiedFaceDetector(detector_method, device=detector_device)
        
        # Initialize EdgeFace recognizer
        print(f"🧠 Initializing EdgeFace recognizer...")
        self._init_recognizer(edgeface_model_path, edgeface_model_name)
        
        # Initialize reference database
        print(f"💾 Initializing reference database...")
        self.ref_db = ReferenceDatabase()
        
        # Settings
        self.similarity_threshold = similarity_threshold
        
        # FPS calculation
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Face tracking
        self.tracked_faces = {}
        self.next_track_id = 0
        self.max_tracking_frames = 10
        
        print(f"✅ System initialized successfully!")
        print(f"   Mode: {'TensorRT' if self.use_tensorrt else 'PyTorch'}")
        if hasattr(self.recognizer, 'get_inference_time'):
            print(f"   TensorRT available: {JETSON_AVAILABLE}")
    
    def _init_recognizer(self, model_path: str, model_name: str):
        """Initialize the appropriate recognizer"""
        
        # Determine model type from path
        is_trt_model = model_path.endswith('.trt')
        is_pt_model = model_path.endswith('.pt')
        
        if self.use_tensorrt and is_trt_model:
            # Use TensorRT
            if not JETSON_AVAILABLE:
                raise ImportError("TensorRT not available. Install JetPack on Jetson.")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"TensorRT engine not found: {model_path}\n"
                    f"Run: python onnx_to_tensorrt.py --input checkpoints/edgeface_xs_gamma_06.onnx --output {model_path}"
                )
            
            self.recognizer = EdgeFaceJetsonRecognizer(model_path, model_name, device='jetson')
            print(f"✅ Using TensorRT engine: {model_path}")
            
        elif self.use_tensorrt and is_pt_model:
            # User wants TensorRT but provided .pt file
            # Check if TRT version exists
            trt_path = model_path.replace('.pt', '.trt')
            if os.path.exists(trt_path):
                print(f"⚠️ Found TensorRT engine: {trt_path}, using it instead of .pt")
                self.recognizer = EdgeFaceJetsonRecognizer(trt_path, model_name, device='jetson')
            else:
                print(f"⚠️ TensorRT engine not found, falling back to PyTorch")
                self.use_tensorrt = False
                # Use 'cuda' for PyTorch (not 'jetson')
                pytorch_device = 'cuda' if self.device == 'jetson' else self.device
                self.recognizer = EdgeFaceRecognizer(model_path, model_name, pytorch_device)
        else:
            # Use PyTorch
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Use 'cuda' for PyTorch (not 'jetson')
            pytorch_device = 'cuda' if self.device == 'jetson' else self.device
            self.recognizer = EdgeFaceRecognizer(model_path, model_name, pytorch_device)
            print(f"✅ Using PyTorch model: {model_path}")
    
    def switch_mode(self, use_tensorrt: bool) -> bool:
        """
        Switch between TensorRT and PyTorch mode
        
        Args:
            use_tensorrt: True for TensorRT, False for PyTorch
            
        Returns:
            True if switch successful
        """
        if use_tensorrt == self.use_tensorrt:
            return True
        
        try:
            if use_tensorrt:
                # Switch to TensorRT
                trt_path = 'checkpoints/edgeface_xs_gamma_06.trt'
                if not os.path.exists(trt_path):
                    print(f"❌ TensorRT engine not found: {trt_path}")
                    return False
                
                self.recognizer = EdgeFaceJetsonRecognizer(trt_path, 'edgeface_xs_gamma_06', device='jetson')
                self.use_tensorrt = True
                print("✅ Switched to TensorRT mode")
            else:
                # Switch to PyTorch
                pt_path = 'checkpoints/edgeface_xs_gamma_06.pt'
                if not os.path.exists(pt_path):
                    print(f"❌ PyTorch model not found: {pt_path}")
                    return False
                
                # Use 'cuda' for PyTorch (not 'jetson')
                pytorch_device = 'cuda' if self.device == 'jetson' else self.device
                self.recognizer = EdgeFaceRecognizer(pt_path, 'edgeface_xs_gamma_06', pytorch_device)
                self.use_tensorrt = False
                print("✅ Switched to PyTorch mode")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to switch mode: {e}")
            return False
    
    def add_reference_from_image(self, image_path: str, person_id: str) -> bool:
        """Add reference person from image file"""
        try:
            img = Image.open(image_path)
            aligned_face = self.detector.align(img)
            
            if aligned_face is None:
                print(f"❌ No face detected in {image_path}")
                return False
            
            face_np = np.array(aligned_face)
            face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            
            embedding = self.recognizer.extract_embedding(face_np)
            self.ref_db.add_person(person_id, embedding)
            
            print(f"✅ Added {person_id} to reference database")
            return True
            
        except Exception as e:
            print(f"❌ Error adding reference: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame with performance monitoring
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            (annotated_frame, detections)
        """
        # Detection timing
        detect_start = time.time()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        bboxes, landmarks = self.detector.detect_faces(pil_img)
        
        detect_time = time.time() - detect_start
        
        # Recognition timing
        recog_start = time.time()
        
        detections = []
        
        if bboxes is not None and len(bboxes) > 0:
            aligned_faces = []
            valid_indices = []
            
            for i, (bbox, lm) in enumerate(zip(bboxes, landmarks)):
                aligned_face = self.detector.align_face(pil_img, lm)
                
                if aligned_face is not None:
                    face_np = np.array(aligned_face)
                    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    aligned_faces.append(face_np)
                    valid_indices.append(i)
            
            if aligned_faces:
                embeddings = self.recognizer.extract_embeddings_batch(aligned_faces)
                
                emb_idx = 0
                for i, (bbox, lm) in enumerate(zip(bboxes, landmarks)):
                    if i in valid_indices:
                        embedding = embeddings[emb_idx]
                        emb_idx += 1
                        
                        # Find match
                        person_id, similarity = self.ref_db.find_match(
                            embedding, self.similarity_threshold
                        )
                        
                        if person_id is None:
                            person_id = 'Unknown'
                            similarity = 0.0
                        
                        detections.append({
                            'person_id': person_id,
                            'similarity': similarity,
                            'bbox': bbox,
                            'landmarks': lm
                        })
        
        recog_time = time.time() - recog_start
        
        # Update performance monitor
        self.perf_monitor.update(detect_time, recog_time)
        
        # Update FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        # Annotate frame
        annotated_frame = self.draw_results(frame.copy(), detections)
        
        return annotated_frame, detections
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results and performance info on frame"""
        
        # Get performance stats
        stats = self.perf_monitor.get_stats()
        
        # Draw performance overlay
        overlay_y = 10
        line_height = 25
        
        # Background for performance info
        cv2.rectangle(frame, (5, 5), (250, 135), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (250, 135), (0, 255, 0), 2)
        
        # Mode indicator
        mode_text = "TensorRT" if self.use_tensorrt else "PyTorch"
        mode_color = (0, 255, 255) if self.use_tensorrt else (255, 165, 0)
        cv2.putText(frame, f"Mode: {mode_text}", (10, overlay_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, overlay_y + 2*line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Inference time
        cv2.putText(frame, f"Detect: {stats['detection_ms']:.1f}ms", (10, overlay_y + 3*line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Recog: {stats['recognition_ms']:.1f}ms", (10, overlay_y + 4*line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Temperature (if available)
        if stats['temperature_c'] > 0:
            temp_color = (0, 255, 0) if stats['temperature_c'] < 60 else (0, 165, 255) if stats['temperature_c'] < 75 else (0, 0, 255)
            cv2.putText(frame, f"Temp: {stats['temperature_c']:.0f}C", (10, overlay_y + 5*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, temp_color, 1)
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            color = (0, 255, 0) if det['person_id'] != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['person_id']}: {det['similarity']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Draw landmarks
            if det['landmarks'] is not None:
                for i in range(0, len(det['landmarks']), 2):
                    x, y = int(det['landmarks'][i]), int(det['landmarks'][i+1])
                    cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
        
        return frame
    
    def add_reference_from_frame(self, frame: np.ndarray, person_id: str, angle: str = 'front') -> bool:
        """Add reference person from camera frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            aligned_face = self.detector.align(pil_img)
            
            if aligned_face is None:
                print(f"❌ No face detected in frame")
                return False
            
            face_np = np.array(aligned_face)
            face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            
            embedding = self.recognizer.extract_embedding(face_np)
            self.ref_db.add_person(person_id, embedding, angle)
            
            # Save captured image
            os.makedirs(f"captured_references/{person_id}", exist_ok=True)
            save_path = f"captured_references/{person_id}/{angle}.jpg"
            cv2.imwrite(save_path, cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR))
            
            print(f"✅ Added {person_id} ({angle}) to reference database")
            return True
            
        except Exception as e:
            print(f"❌ Error adding reference: {e}")
            return False
    
    def run_benchmark(self, num_iterations: int = 100) -> Dict:
        """
        Run performance benchmark
        
        Args:
            num_iterations: Number of inference iterations
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Running Benchmark ({num_iterations} iterations)")
        print(f"Mode: {'TensorRT' if self.use_tensorrt else 'PyTorch'}")
        print(f"{'='*60}")
        
        # Create dummy input
        dummy_face = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            _ = self.recognizer.extract_embedding(dummy_face)
        
        # Benchmark
        print("Benchmarking...")
        times = []
        for i in range(num_iterations):
            start = time.time()
            _ = self.recognizer.extract_embedding(dummy_face)
            times.append((time.time() - start) * 1000)  # ms
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
        
        results = {
            'mode': 'TensorRT' if self.use_tensorrt else 'PyTorch',
            'iterations': num_iterations,
            'avg_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput_fps': 1000 / np.mean(times)
        }
        
        print(f"\nResults:")
        print(f"  Average: {results['avg_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"  Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
        print(f"  P50/P95/P99: {results['p50_ms']:.2f} / {results['p95_ms']:.2f} / {results['p99_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_fps']:.1f} FPS")
        print(f"{'='*60}\n")
        
        return results
    
    def run_camera(self, camera_id: int = 0):
        """Run face recognition on camera feed"""
        print(f"📹 Opening camera {camera_id}...")
        
        # V4L2 backend with MJPG format - confirmed working on Jetson
        cap = cv2.VideoCapture(f'/dev/video{camera_id}', cv2.CAP_V4L2)
        
        if cap.isOpened():
            # Set MJPG format first (important for USB cameras)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret:
                print(f"✅ Camera opened successfully with V4L2 backend")
                print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            else:
                cap.release()
                cap = None
        
        if not cap or not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        print("Controls:")
        print("  'q' - Quit")
        print("  't' - Toggle TensorRT/PyTorch")
        print("  'b' - Run benchmark")
        print("  'c' - Capture reference")
        
        try:
            while True:
                ret, frame = cap.read()

                
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Process frame
                annotated_frame, detections = self.process_frame(frame)
                
                # Display
                cv2.imshow('Jetson Face Recognition', annotated_frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.switch_mode(not self.use_tensorrt)
                elif key == ord('b'):
                    self.run_benchmark()
                elif key == ord('c'):
                    person_id = input("Enter person name/ID: ").strip()
                    if person_id:
                        self.add_reference_from_frame(frame, person_id)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Camera closed")


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Jetson Face Recognition System')
    parser.add_argument('--detector', type=str, default='yunet',
                       choices=['mtcnn', 'yunet', 'yolov5_face', 'yolov8'],
                       help='Face detection method')
    parser.add_argument('--model', type=str, default='checkpoints/edgeface_xs_gamma_06.trt',
                       help='EdgeFace model path (.trt or .pt)')
    parser.add_argument('--pytorch', action='store_true',
                       help='Use PyTorch instead of TensorRT')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark only')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.pytorch:
        model_path = args.model.replace('.trt', '.pt')
    else:
        model_path = args.model
    
    # Initialize system
    system = FaceRecognitionJetsonSystem(
        detector_method=args.detector,
        edgeface_model_path=model_path,
        similarity_threshold=args.threshold,
        use_tensorrt=not args.pytorch
    )
    
    if args.benchmark:
        # Run benchmark only
        system.run_benchmark(200)
    else:
        # Run camera
        system.run_camera(args.camera)


if __name__ == '__main__':
    main()
