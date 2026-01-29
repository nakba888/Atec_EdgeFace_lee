#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch vs TensorRT Performance Comparison for Jetson Orin Nano
Jetson Orin Nano에서 PyTorch vs TensorRT 성능 비교

This script compares:
1. Inference speed (latency, throughput)
2. Embedding accuracy (cosine similarity between PyTorch and TensorRT)
3. Memory usage
4. Temperature during inference

Usage:
    python jetson_pytorch_comparison.py
    python jetson_pytorch_comparison.py --iterations 200 --output results.json
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

# Matplotlib backend for headless mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Check available backends
PYTORCH_AVAILABLE = False
TENSORRT_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available")

try:
    from edgeface_jetson_recognizer import EdgeFaceJetsonRecognizer, TRT_AVAILABLE
    TENSORRT_AVAILABLE = TRT_AVAILABLE
except ImportError:
    print("⚠️ TensorRT recognizer not available")

from backbones import get_model


class PerformanceBenchmark:
    """Benchmark class for comparing PyTorch and TensorRT"""
    
    def __init__(self, 
                 pytorch_model_path: str = 'checkpoints/edgeface_xs_gamma_06.pt',
                 tensorrt_model_path: str = 'checkpoints/edgeface_xs_gamma_06.trt',
                 model_name: str = 'edgeface_xs_gamma_06'):
        """
        Initialize benchmark
        
        Args:
            pytorch_model_path: Path to PyTorch model
            tensorrt_model_path: Path to TensorRT engine
            model_name: Model architecture name
        """
        self.pytorch_model_path = pytorch_model_path
        self.tensorrt_model_path = tensorrt_model_path
        self.model_name = model_name
        
        self.pytorch_model = None
        self.tensorrt_model = None
        self.device = 'cuda' if PYTORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        print(f"{'='*60}")
        print("Performance Benchmark: PyTorch vs TensorRT")
        print(f"{'='*60}")
        print(f"PyTorch available: {PYTORCH_AVAILABLE}")
        print(f"TensorRT available: {TENSORRT_AVAILABLE}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
    
    def load_pytorch_model(self) -> bool:
        """Load PyTorch model"""
        if not PYTORCH_AVAILABLE:
            print("❌ PyTorch not available")
            return False
        
        if not os.path.exists(self.pytorch_model_path):
            print(f"❌ PyTorch model not found: {self.pytorch_model_path}")
            return False
        
        try:
            print(f"Loading PyTorch model: {self.pytorch_model_path}")
            self.pytorch_model = get_model(self.model_name, fp16=False)
            self.pytorch_model.load_state_dict(
                torch.load(self.pytorch_model_path, map_location=self.device)
            )
            self.pytorch_model.to(self.device)
            self.pytorch_model.eval()
            print("✅ PyTorch model loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load PyTorch model: {e}")
            return False
    
    def load_tensorrt_model(self) -> bool:
        """Load TensorRT model"""
        if not TENSORRT_AVAILABLE:
            print("❌ TensorRT not available")
            return False
        
        if not os.path.exists(self.tensorrt_model_path):
            print(f"❌ TensorRT engine not found: {self.tensorrt_model_path}")
            print(f"💡 Run: python onnx_to_tensorrt.py --input checkpoints/edgeface_xs_gamma_06.onnx")
            return False
        
        try:
            print(f"Loading TensorRT engine: {self.tensorrt_model_path}")
            self.tensorrt_model = EdgeFaceJetsonRecognizer(
                self.tensorrt_model_path, 
                self.model_name, 
                device='jetson'
            )
            print("✅ TensorRT engine loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load TensorRT engine: {e}")
            return False
    
    def pytorch_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run PyTorch inference"""
        with torch.no_grad():
            # Preprocess
            img = input_tensor.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
            
            # Inference
            output = self.pytorch_model(tensor).cpu().numpy().flatten()
            
            # L2 normalize
            output = output / np.linalg.norm(output)
            
        return output
    
    def tensorrt_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run TensorRT inference"""
        return self.tensorrt_model.extract_embedding(input_tensor)
    
    def benchmark_latency(self, num_iterations: int = 100, warmup: int = 10) -> Dict:
        """
        Benchmark inference latency
        
        Args:
            num_iterations: Number of inference iterations
            warmup: Number of warmup iterations
            
        Returns:
            Results dictionary
        """
        # Create random test image
        test_image = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        
        results = {
            'pytorch': None,
            'tensorrt': None
        }
        
        # Benchmark PyTorch
        if self.pytorch_model is not None:
            print(f"\nBenchmarking PyTorch ({num_iterations} iterations)...")
            
            # Warmup
            for _ in range(warmup):
                _ = self.pytorch_inference(test_image)
            
            # Benchmark
            times = []
            for i in range(num_iterations):
                start = time.time()
                _ = self.pytorch_inference(test_image)
                times.append((time.time() - start) * 1000)  # ms
                
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{num_iterations}")
            
            results['pytorch'] = {
                'avg_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'p50_ms': np.percentile(times, 50),
                'p95_ms': np.percentile(times, 95),
                'p99_ms': np.percentile(times, 99),
                'throughput_fps': 1000 / np.mean(times),
                'times': times
            }
            
            print(f"  ✅ PyTorch: {results['pytorch']['avg_ms']:.2f}ms avg, {results['pytorch']['throughput_fps']:.1f} FPS")
        
        # Benchmark TensorRT
        if self.tensorrt_model is not None:
            print(f"\nBenchmarking TensorRT ({num_iterations} iterations)...")
            
            # Warmup
            for _ in range(warmup):
                _ = self.tensorrt_inference(test_image)
            
            # Benchmark
            times = []
            for i in range(num_iterations):
                start = time.time()
                _ = self.tensorrt_inference(test_image)
                times.append((time.time() - start) * 1000)  # ms
                
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{num_iterations}")
            
            results['tensorrt'] = {
                'avg_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'p50_ms': np.percentile(times, 50),
                'p95_ms': np.percentile(times, 95),
                'p99_ms': np.percentile(times, 99),
                'throughput_fps': 1000 / np.mean(times),
                'times': times
            }
            
            print(f"  ✅ TensorRT: {results['tensorrt']['avg_ms']:.2f}ms avg, {results['tensorrt']['throughput_fps']:.1f} FPS")
        
        # Calculate speedup
        if results['pytorch'] and results['tensorrt']:
            speedup = results['pytorch']['avg_ms'] / results['tensorrt']['avg_ms']
            results['speedup'] = speedup
            print(f"\n🚀 TensorRT Speedup: {speedup:.2f}x faster than PyTorch")
        
        return results
    
    def benchmark_accuracy(self, num_samples: int = 100) -> Dict:
        """
        Compare embedding accuracy between PyTorch and TensorRT
        
        Args:
            num_samples: Number of test samples
            
        Returns:
            Accuracy comparison results
        """
        if self.pytorch_model is None or self.tensorrt_model is None:
            print("❌ Both PyTorch and TensorRT models required for accuracy comparison")
            return {}
        
        print(f"\nComparing embedding accuracy ({num_samples} samples)...")
        
        cosine_similarities = []
        l2_distances = []
        
        for i in range(num_samples):
            # Create random test image
            test_image = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
            
            # Get embeddings
            emb_pytorch = self.pytorch_inference(test_image)
            emb_tensorrt = self.tensorrt_inference(test_image)
            
            # Calculate cosine similarity
            cos_sim = np.dot(emb_pytorch, emb_tensorrt)
            cosine_similarities.append(cos_sim)
            
            # Calculate L2 distance
            l2_dist = np.linalg.norm(emb_pytorch - emb_tensorrt)
            l2_distances.append(l2_dist)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_samples}")
        
        results = {
            'num_samples': num_samples,
            'cosine_similarity': {
                'mean': float(np.mean(cosine_similarities)),
                'std': float(np.std(cosine_similarities)),
                'min': float(np.min(cosine_similarities)),
                'max': float(np.max(cosine_similarities))
            },
            'l2_distance': {
                'mean': float(np.mean(l2_distances)),
                'std': float(np.std(l2_distances)),
                'min': float(np.min(l2_distances)),
                'max': float(np.max(l2_distances))
            },
            'cosine_similarities': cosine_similarities,
            'l2_distances': l2_distances
        }
        
        print(f"\n📊 Accuracy Results:")
        print(f"  Cosine Similarity: {results['cosine_similarity']['mean']:.6f} ± {results['cosine_similarity']['std']:.6f}")
        print(f"  L2 Distance: {results['l2_distance']['mean']:.6f} ± {results['l2_distance']['std']:.6f}")
        
        if results['cosine_similarity']['mean'] > 0.99:
            print("  ✅ Embeddings are nearly identical (similarity > 0.99)")
        elif results['cosine_similarity']['mean'] > 0.95:
            print("  ⚠️ Embeddings are similar (similarity > 0.95)")
        else:
            print("  ❌ Significant embedding difference detected")
        
        return results
    
    def plot_results(self, latency_results: Dict, accuracy_results: Dict, output_dir: str = 'benchmark_results'):
        """
        Create visualization plots
        
        Args:
            latency_results: Latency benchmark results
            accuracy_results: Accuracy comparison results
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Latency comparison plot
        if latency_results.get('pytorch') and latency_results.get('tensorrt'):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Bar chart: Average latency
            labels = ['PyTorch', 'TensorRT']
            avgs = [latency_results['pytorch']['avg_ms'], latency_results['tensorrt']['avg_ms']]
            stds = [latency_results['pytorch']['std_ms'], latency_results['tensorrt']['std_ms']]
            colors = ['#2E86AB', '#A23B72']
            
            bars = axes[0].bar(labels, avgs, yerr=stds, capsize=5, color=colors, alpha=0.8)
            axes[0].set_ylabel('Latency (ms)')
            axes[0].set_title('Average Inference Latency')
            
            # Add value labels
            for bar, val in zip(bars, avgs):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.2f}ms', ha='center', va='bottom', fontsize=10)
            
            # Speedup annotation
            speedup = latency_results.get('speedup', 1.0)
            axes[0].text(0.5, 0.95, f'Speedup: {speedup:.2f}x', transform=axes[0].transAxes,
                        ha='center', va='top', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Histogram: Latency distribution
            if 'times' in latency_results['pytorch']:
                axes[1].hist(latency_results['pytorch']['times'], bins=30, alpha=0.7, 
                           label='PyTorch', color='#2E86AB')
            if 'times' in latency_results['tensorrt']:
                axes[1].hist(latency_results['tensorrt']['times'], bins=30, alpha=0.7,
                           label='TensorRT', color='#A23B72')
            axes[1].set_xlabel('Latency (ms)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Latency Distribution')
            axes[1].legend()
            
            # Throughput comparison
            throughputs = [latency_results['pytorch']['throughput_fps'],
                          latency_results['tensorrt']['throughput_fps']]
            bars = axes[2].bar(labels, throughputs, color=colors, alpha=0.8)
            axes[2].set_ylabel('FPS')
            axes[2].set_title('Throughput (FPS)')
            
            for bar, val in zip(bars, throughputs):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_dir}/latency_comparison.png")
            plt.close()
        
        # Accuracy plot
        if accuracy_results and 'cosine_similarities' in accuracy_results:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Cosine similarity histogram
            axes[0].hist(accuracy_results['cosine_similarities'], bins=30, alpha=0.7, color='green')
            axes[0].axvline(accuracy_results['cosine_similarity']['mean'], color='red',
                          linestyle='--', linewidth=2, 
                          label=f"Mean: {accuracy_results['cosine_similarity']['mean']:.6f}")
            axes[0].set_xlabel('Cosine Similarity')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('PyTorch vs TensorRT Embedding Similarity')
            axes[0].legend()
            
            # L2 distance histogram
            axes[1].hist(accuracy_results['l2_distances'], bins=30, alpha=0.7, color='orange')
            axes[1].axvline(accuracy_results['l2_distance']['mean'], color='red',
                          linestyle='--', linewidth=2,
                          label=f"Mean: {accuracy_results['l2_distance']['mean']:.6f}")
            axes[1].set_xlabel('L2 Distance')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('PyTorch vs TensorRT Embedding Distance')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_dir}/accuracy_comparison.png")
            plt.close()
    
    def run_full_benchmark(self, num_iterations: int = 100, num_accuracy_samples: int = 50,
                          output_dir: str = 'benchmark_results') -> Dict:
        """
        Run full benchmark suite
        
        Args:
            num_iterations: Number of latency iterations
            num_accuracy_samples: Number of accuracy test samples
            output_dir: Directory to save results
            
        Returns:
            Complete benchmark results
        """
        print(f"\n{'='*60}")
        print("Running Full Benchmark Suite")
        print(f"{'='*60}")
        
        # Load models
        pytorch_loaded = self.load_pytorch_model()
        tensorrt_loaded = self.load_tensorrt_model()
        
        if not pytorch_loaded and not tensorrt_loaded:
            print("❌ No models available for benchmarking")
            return {}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'pytorch_available': pytorch_loaded,
            'tensorrt_available': tensorrt_loaded,
            'num_iterations': num_iterations,
            'num_accuracy_samples': num_accuracy_samples
        }
        
        # Latency benchmark
        latency_results = self.benchmark_latency(num_iterations)
        results['latency'] = {
            k: {kk: vv for kk, vv in v.items() if kk != 'times'} if isinstance(v, dict) else v
            for k, v in latency_results.items()
        }
        
        # Accuracy benchmark (if both models available)
        if pytorch_loaded and tensorrt_loaded:
            accuracy_results = self.benchmark_accuracy(num_accuracy_samples)
            results['accuracy'] = {
                k: v for k, v in accuracy_results.items() 
                if k not in ['cosine_similarities', 'l2_distances']
            }
        else:
            accuracy_results = {}
        
        # Generate plots
        print(f"\nGenerating plots...")
        self.plot_results(latency_results, accuracy_results, output_dir)
        
        # Save JSON results
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'benchmark_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {json_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        if results['latency'].get('pytorch'):
            pt = results['latency']['pytorch']
            print(f"\nPyTorch:")
            print(f"  Latency: {pt['avg_ms']:.2f} ± {pt['std_ms']:.2f} ms")
            print(f"  Throughput: {pt['throughput_fps']:.1f} FPS")
        
        if results['latency'].get('tensorrt'):
            trt = results['latency']['tensorrt']
            print(f"\nTensorRT:")
            print(f"  Latency: {trt['avg_ms']:.2f} ± {trt['std_ms']:.2f} ms")
            print(f"  Throughput: {trt['throughput_fps']:.1f} FPS")
        
        if results['latency'].get('speedup'):
            print(f"\n🚀 TensorRT Speedup: {results['latency']['speedup']:.2f}x")
        
        if results.get('accuracy'):
            print(f"\nAccuracy (PyTorch vs TensorRT):")
            print(f"  Cosine Similarity: {results['accuracy']['cosine_similarity']['mean']:.6f}")
            print(f"  L2 Distance: {results['accuracy']['l2_distance']['mean']:.6f}")
        
        print(f"\n{'='*60}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='PyTorch vs TensorRT Performance Comparison')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of latency test iterations')
    parser.add_argument('--accuracy-samples', type=int, default=50,
                       help='Number of accuracy test samples')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--pytorch-model', type=str, 
                       default='checkpoints/edgeface_xs_gamma_06.pt',
                       help='Path to PyTorch model')
    parser.add_argument('--tensorrt-model', type=str,
                       default='checkpoints/edgeface_xs_gamma_06.trt',
                       help='Path to TensorRT engine')
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(
        pytorch_model_path=args.pytorch_model,
        tensorrt_model_path=args.tensorrt_model
    )
    
    results = benchmark.run_full_benchmark(
        num_iterations=args.iterations,
        num_accuracy_samples=args.accuracy_samples,
        output_dir=args.output
    )
    
    return 0 if results else 1


if __name__ == '__main__':
    exit(main())
