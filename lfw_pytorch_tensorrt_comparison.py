#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFW Dataset: PyTorch vs TensorRT Accuracy & Timing Comparison for Jetson Orin Nano
Jetson Orin Nano에서 PyTorch vs TensorRT 정확도 및 처리 시간 비교

평가 목표:
1. Face Verification 정확도: ROC AUC, Best Accuracy, EER
2. 처리 시간 분석: 전체 시간, 임베딩만 추출 시간
3. PyTorch vs TensorRT 임베딩 유사도 (양자화 오차 분석)

사용법:
    python lfw_pytorch_tensorrt_comparison.py
    python lfw_pytorch_tensorrt_comparison.py --max-pairs 1000 --output benchmark_results
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

# Matplotlib backend for headless mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import cv2

# ============================================================================
# 1. 환경 설정 및 초기화
# ============================================================================

# PyTorch 관련
PYTORCH_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available")

# TensorRT 관련
TENSORRT_AVAILABLE = False
try:
    from edgeface_jetson_recognizer import EdgeFaceJetsonRecognizer, TRT_AVAILABLE
    TENSORRT_AVAILABLE = TRT_AVAILABLE
except ImportError:
    print("⚠️ TensorRT recognizer not available")

# EdgeFace backbone
from backbones import get_model

# YuNet detector
from face_alignment.yunet import YuNetDetector

print("=" * 60)
print("LFW Evaluation: PyTorch vs TensorRT on Jetson Orin Nano")
print("=" * 60)

if PYTORCH_AVAILABLE:
    print(f"✅ PyTorch version: {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ PyTorch device: {device}")
else:
    device = 'cpu'

print(f"✅ TensorRT available: {TENSORRT_AVAILABLE}")
print("=" * 60)


# ============================================================================
# 2. 설정
# ============================================================================

# LFW 데이터셋 경로 (Jetson용)
LFW_DIR = "/lfw_dataset/lfw-deepfunneled"
PAIRS_FILE = "/lfw_dataset/pairs.csv"  # pairs.csv 또는 pairs.txt

# 모델 경로
PYTORCH_MODEL_PATH = "checkpoints/edgeface_xs_gamma_06.pt"
TENSORRT_MODEL_PATH = "checkpoints/edgeface_xs_gamma_06.trt"
MODEL_NAME = "edgeface_xs_gamma_06"

# YuNet 모델 경로
YUNET_MODEL_PATH = "face_alignment/models/face_detection_yunet_2023mar.onnx"


# ============================================================================
# 3. 데이터 로딩 함수
# ============================================================================

def load_lfw_pairs(pairs_file: str, lfw_dir: str) -> List[Tuple]:
    """
    LFW pairs 파일을 로드합니다.
    
    지원 형식:
    - pairs.txt: 표준 LFW format
    - pairs.csv: CSV format
    
    Returns:
        List of (is_same, img1_path, img2_path) tuples
    """
    pairs = []
    
    if not os.path.exists(pairs_file):
        print(f"❌ Pairs file not found: {pairs_file}")
        return pairs
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
    
    # Detect format
    is_csv = pairs_file.endswith('.csv')
    
    # Skip header for CSV
    start_idx = 1 if is_csv else 1  # First line is usually header or count
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        if is_csv:
            # CSV format
            if line.endswith(','):
                # Same person pair
                parts = line.rstrip(',').split(',')
                if len(parts) == 3:
                    try:
                        person = parts[0]
                        img1_num = int(parts[1])
                        img2_num = int(parts[2])
                        
                        img1_path = os.path.join(lfw_dir, person, f"{person}_{img1_num:04d}.jpg")
                        img2_path = os.path.join(lfw_dir, person, f"{person}_{img2_num:04d}.jpg")
                        pairs.append((True, img1_path, img2_path))
                    except ValueError:
                        continue
            else:
                # Different person pair
                parts = line.split(',')
                if len(parts) == 4:
                    try:
                        person1 = parts[0]
                        img1_num = int(parts[1])
                        person2 = parts[2]
                        img2_num = int(parts[3])
                        
                        img1_path = os.path.join(lfw_dir, person1, f"{person1}_{img1_num:04d}.jpg")
                        img2_path = os.path.join(lfw_dir, person2, f"{person2}_{img2_num:04d}.jpg")
                        pairs.append((False, img1_path, img2_path))
                    except ValueError:
                        continue
        else:
            # Standard LFW pairs.txt format
            parts = line.split('\t')
            
            if len(parts) == 3:
                # Same person: name, img1_num, img2_num
                try:
                    person = parts[0]
                    img1_num = int(parts[1])
                    img2_num = int(parts[2])
                    
                    img1_path = os.path.join(lfw_dir, person, f"{person}_{img1_num:04d}.jpg")
                    img2_path = os.path.join(lfw_dir, person, f"{person}_{img2_num:04d}.jpg")
                    pairs.append((True, img1_path, img2_path))
                except ValueError:
                    continue
            elif len(parts) == 4:
                # Different person: name1, img1_num, name2, img2_num
                try:
                    person1 = parts[0]
                    img1_num = int(parts[1])
                    person2 = parts[2]
                    img2_num = int(parts[3])
                    
                    img1_path = os.path.join(lfw_dir, person1, f"{person1}_{img1_num:04d}.jpg")
                    img2_path = os.path.join(lfw_dir, person2, f"{person2}_{img2_num:04d}.jpg")
                    pairs.append((False, img1_path, img2_path))
                except ValueError:
                    continue
    
    positive_pairs = sum(1 for p in pairs if p[0])
    negative_pairs = sum(1 for p in pairs if not p[0])
    print(f"✅ Loaded {len(pairs)} pairs (positive: {positive_pairs}, negative: {negative_pairs})")
    
    return pairs


# ============================================================================
# 4. 모델 로딩 함수
# ============================================================================

def load_pytorch_model(model_path: str, model_name: str, device: str):
    """PyTorch 모델 로드"""
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ PyTorch model not found: {model_path}")
        return None
    
    try:
        model = get_model(model_name, fp16=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ PyTorch model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load PyTorch model: {e}")
        return None


def load_tensorrt_model(model_path: str, model_name: str):
    """TensorRT 모델 로드"""
    if not TENSORRT_AVAILABLE:
        print("❌ TensorRT not available")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ TensorRT engine not found: {model_path}")
        return None
    
    try:
        model = EdgeFaceJetsonRecognizer(model_path, model_name, device='jetson')
        print(f"✅ TensorRT engine loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load TensorRT engine: {e}")
        return None


# ============================================================================
# 5. 임베딩 추출 함수 (시간 측정 포함)
# ============================================================================

def extract_embedding_pytorch(face_img: np.ndarray, model, device: str) -> Tuple[np.ndarray, float]:
    """
    PyTorch 모델로 임베딩 추출 (시간 측정 포함)
    
    Args:
        face_img: BGR 이미지 (112x112x3)
        model: PyTorch 모델
        device: 디바이스
        
    Returns:
        (512-d embedding vector, inference_time_ms)
    """
    # Resize if needed
    if face_img.shape[:2] != (112, 112):
        face_img = cv2.resize(face_img, (112, 112))
    
    # BGR to RGB
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Transpose to CHW
    img = np.transpose(img, (2, 0, 1))
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    # Normalize
    img_tensor.div_(255).sub_(0.5).div_(0.5)
    
    # Extract embedding with timing
    start_time = time.time()
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy().flatten()
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding, inference_time


def extract_embedding_tensorrt(face_img: np.ndarray, model) -> Tuple[np.ndarray, float]:
    """
    TensorRT 모델로 임베딩 추출 (시간 측정 포함)
    
    Args:
        face_img: BGR 이미지 (112x112x3)
        model: EdgeFaceJetsonRecognizer
        
    Returns:
        (512-d embedding vector, inference_time_ms)
    """
    start_time = time.time()
    embedding = model.extract_embedding(face_img)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    return embedding, inference_time


# ============================================================================
# 6. End-to-End 파이프라인 평가
# ============================================================================

def evaluate_pipeline(pairs: List[Tuple], detector, recognizer, pipeline_name: str,
                     device: str = 'cuda', use_tensorrt: bool = False,
                     max_pairs: Optional[int] = None) -> Dict:
    """
    End-to-end 파이프라인 평가 (시간 측정 포함)
    
    Returns:
        Dict with evaluation results (ROC AUC, accuracy, timing, etc.)
    """
    similarities = []
    labels = []
    total_times = []  # 전체 처리 시간 (detection + embedding)
    embedding_times = []  # 임베딩만 추출 시간
    failed_count = 0
    
    # Sample pairs
    if max_pairs:
        positive_pairs = [p for p in pairs if p[0] == True]
        negative_pairs = [p for p in pairs if p[0] == False]
        
        half_pairs = max_pairs // 2
        selected_positive = positive_pairs[:half_pairs]
        selected_negative = negative_pairs[:half_pairs]
        
        pairs_to_process = selected_positive + selected_negative
    else:
        pairs_to_process = pairs
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {pipeline_name}")
    print(f"Processing {len(pairs_to_process)} pairs...")
    print(f"{'='*60}")
    
    for is_same, img1_path, img2_path in tqdm(pairs_to_process, desc=pipeline_name):
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            failed_count += 1
            continue
        
        total_start = time.time()
        
        try:
            # Load images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Detect and align
            aligned1 = detector.align(img1)
            aligned2 = detector.align(img2)
            
            if aligned1 is None or aligned2 is None:
                failed_count += 1
                continue
            
            # Convert to numpy BGR
            face1_np = cv2.cvtColor(np.array(aligned1), cv2.COLOR_RGB2BGR)
            face2_np = cv2.cvtColor(np.array(aligned2), cv2.COLOR_RGB2BGR)
            
            # Extract embeddings with timing
            if use_tensorrt:
                emb1, time1 = extract_embedding_tensorrt(face1_np, recognizer)
                emb2, time2 = extract_embedding_tensorrt(face2_np, recognizer)
            else:
                emb1, time1 = extract_embedding_pytorch(face1_np, recognizer, device)
                emb2, time2 = extract_embedding_pytorch(face2_np, recognizer, device)
            
            embedding_times.append(time1 + time2)  # 두 이미지의 임베딩 시간 합
            
            # Calculate similarity
            similarity = np.dot(emb1, emb2)
            similarities.append(similarity)
            labels.append(1 if is_same else 0)
            
            total_times.append((time.time() - total_start) * 1000)  # ms
            
        except Exception as e:
            failed_count += 1
            continue
    
    # Calculate metrics
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    if len(similarities) == 0:
        print("❌ No valid pairs processed!")
        return None
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Best accuracy
    accuracies = []
    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        accuracies.append(accuracy)
    
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]
    
    # EER
    eer_idx = np.nanargmin(np.absolute(fpr - (1 - tpr)))
    eer = fpr[eer_idx]
    
    results = {
        'pipeline': pipeline_name,
        'num_pairs': len(similarities),
        'failed_pairs': failed_count,
        'success_rate': len(similarities) / len(pairs_to_process),
        'roc_auc': roc_auc,
        'best_accuracy': best_accuracy,
        'best_threshold': best_threshold,
        'eer': eer,
        # Timing
        'avg_total_time_ms': np.mean(total_times),
        'std_total_time_ms': np.std(total_times),
        'avg_embedding_time_ms': np.mean(embedding_times),
        'std_embedding_time_ms': np.std(embedding_times),
        'total_times': total_times,
        'embedding_times': embedding_times,
        # For ROC plot
        'similarities': similarities,
        'labels': labels,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }
    
    print(f"\n📊 {pipeline_name} Results:")
    print(f"  Pairs processed: {results['num_pairs']}")
    print(f"  Success rate: {results['success_rate']:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  🎯 ROC AUC: {results['roc_auc']:.4f}")
    print(f"  🎯 Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"  🎯 Best Threshold: {results['best_threshold']:.4f}")
    print(f"  🎯 EER: {results['eer']:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  ⏱️ Avg Total Time (per pair): {results['avg_total_time_ms']:.2f} ± {results['std_total_time_ms']:.2f} ms")
    print(f"  ⏱️ Avg Embedding Time (per pair): {results['avg_embedding_time_ms']:.2f} ± {results['std_embedding_time_ms']:.2f} ms")
    
    return results


# ============================================================================
# 7. PyTorch vs TensorRT 임베딩 비교
# ============================================================================

def compare_embeddings(pairs: List[Tuple], detector, pytorch_model, tensorrt_model,
                       device: str, max_pairs: int = 100) -> Dict:
    """
    동일 얼굴에 대한 PyTorch vs TensorRT 임베딩 비교
    
    Returns:
        Dict with embedding comparison statistics
    """
    results = {
        'cosine_similarities': [],
        'l2_distances': [],
        'pytorch_times': [],
        'tensorrt_times': [],
        'valid_samples': 0
    }
    
    print(f"\n{'='*60}")
    print(f"Comparing Embeddings: PyTorch vs TensorRT")
    print(f"{'='*60}")
    
    sampled_pairs = pairs[:max_pairs] if max_pairs else pairs
    
    for is_same, img1_path, img2_path in tqdm(sampled_pairs, desc="Embedding Comparison"):
        for img_path in [img1_path, img2_path]:
            if not os.path.exists(img_path):
                continue
            
            try:
                # Detect and align
                pil_img = Image.open(img_path).convert('RGB')
                aligned_face = detector.align(pil_img)
                
                if aligned_face is None:
                    continue
                
                # Convert to numpy BGR
                face_np = cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR)
                
                # Extract PyTorch embedding
                emb_pytorch, pt_time = extract_embedding_pytorch(face_np, pytorch_model, device)
                results['pytorch_times'].append(pt_time)
                
                # Extract TensorRT embedding
                emb_tensorrt, trt_time = extract_embedding_tensorrt(face_np, tensorrt_model)
                results['tensorrt_times'].append(trt_time)
                
                # Compare
                cos_sim = np.dot(emb_pytorch, emb_tensorrt)
                l2_dist = np.linalg.norm(emb_pytorch - emb_tensorrt)
                
                results['cosine_similarities'].append(cos_sim)
                results['l2_distances'].append(l2_dist)
                results['valid_samples'] += 1
                
            except Exception as e:
                continue
    
    # Calculate statistics
    if results['cosine_similarities']:
        results['avg_cosine_similarity'] = np.mean(results['cosine_similarities'])
        results['std_cosine_similarity'] = np.std(results['cosine_similarities'])
        results['min_cosine_similarity'] = np.min(results['cosine_similarities'])
        results['avg_l2_distance'] = np.mean(results['l2_distances'])
        results['avg_pytorch_time_ms'] = np.mean(results['pytorch_times'])
        results['avg_tensorrt_time_ms'] = np.mean(results['tensorrt_times'])
        results['speedup'] = results['avg_pytorch_time_ms'] / results['avg_tensorrt_time_ms']
    
    print(f"\n📊 Embedding Comparison Results:")
    print(f"  Valid samples: {results['valid_samples']}")
    if results['cosine_similarities']:
        print(f"  ─────────────────────────────────")
        print(f"  🔗 Avg Cosine Similarity: {results['avg_cosine_similarity']:.6f}")
        print(f"  🔗 Min Cosine Similarity: {results['min_cosine_similarity']:.6f}")
        print(f"  🔗 Avg L2 Distance: {results['avg_l2_distance']:.6f}")
        print(f"  ─────────────────────────────────")
        print(f"  ⏱️ Avg PyTorch Time: {results['avg_pytorch_time_ms']:.2f} ms")
        print(f"  ⏱️ Avg TensorRT Time: {results['avg_tensorrt_time_ms']:.2f} ms")
        print(f"  🚀 TensorRT Speedup: {results['speedup']:.2f}x")
        
        if results['avg_cosine_similarity'] > 0.99:
            print("  ✅ Embeddings are nearly identical (similarity > 0.99)")
        elif results['avg_cosine_similarity'] > 0.95:
            print("  ⚠️ Embeddings are similar (similarity > 0.95)")
        else:
            print("  ❌ Significant embedding difference detected")
    
    return results


# ============================================================================
# 8. 시각화 함수
# ============================================================================

def plot_roc_comparison(pytorch_results: Dict, tensorrt_results: Dict, output_dir: str):
    """ROC 커브 비교"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # PyTorch ROC
    if pytorch_results:
        ax.plot(pytorch_results['fpr'], pytorch_results['tpr'],
               label=f"PyTorch (AUC={pytorch_results['roc_auc']:.4f})",
               linewidth=2, color='blue')
    
    # TensorRT ROC
    if tensorrt_results:
        ax.plot(tensorrt_results['fpr'], tensorrt_results['tpr'],
               label=f"TensorRT (AUC={tensorrt_results['roc_auc']:.4f})",
               linewidth=2, color='red')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: PyTorch vs TensorRT on LFW', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pytorch_vs_tensorrt_roc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_metrics_comparison(pytorch_results: Dict, tensorrt_results: Dict, output_dir: str):
    """성능 메트릭 비교 막대 그래프"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    labels = ['PyTorch', 'TensorRT']
    colors = ['#2E86AB', '#A23B72']
    
    # Accuracy metrics
    if pytorch_results and tensorrt_results:
        metrics = [
            ('ROC AUC', [pytorch_results['roc_auc'], tensorrt_results['roc_auc']]),
            ('Best Accuracy', [pytorch_results['best_accuracy'], tensorrt_results['best_accuracy']]),
            ('EER (lower=better)', [pytorch_results['eer'], tensorrt_results['eer']])
        ]
        
        for ax, (metric_name, values) in zip(axes, metrics):
            bars = ax.bar(labels, values, color=colors, alpha=0.8)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(metric_name, fontsize=12)
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
            
            # Value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pytorch_vs_tensorrt_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_timing_comparison(pytorch_results: Dict, tensorrt_results: Dict, 
                          embedding_comparison: Dict, output_dir: str):
    """처리 시간 비교"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    labels = ['PyTorch', 'TensorRT']
    colors = ['#2E86AB', '#A23B72']
    
    # 1. Embedding Time (per pair)
    if pytorch_results and tensorrt_results:
        emb_times = [pytorch_results['avg_embedding_time_ms'], tensorrt_results['avg_embedding_time_ms']]
        emb_stds = [pytorch_results['std_embedding_time_ms'], tensorrt_results['std_embedding_time_ms']]
        
        bars = axes[0].bar(labels, emb_times, yerr=emb_stds, capsize=5, color=colors, alpha=0.8)
        axes[0].set_ylabel('Time (ms)', fontsize=11)
        axes[0].set_title('Embedding Time (per pair)', fontsize=12)
        
        for bar, val in zip(bars, emb_times):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        # Speedup
        speedup = emb_times[0] / emb_times[1] if emb_times[1] > 0 else 0
        axes[0].text(0.5, 0.95, f'Speedup: {speedup:.2f}x', transform=axes[0].transAxes,
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Total Time (per pair)
    if pytorch_results and tensorrt_results:
        total_times = [pytorch_results['avg_total_time_ms'], tensorrt_results['avg_total_time_ms']]
        total_stds = [pytorch_results['std_total_time_ms'], tensorrt_results['std_total_time_ms']]
        
        bars = axes[1].bar(labels, total_times, yerr=total_stds, capsize=5, color=colors, alpha=0.8)
        axes[1].set_ylabel('Time (ms)', fontsize=11)
        axes[1].set_title('Total Time (per pair)', fontsize=12)
        
        for bar, val in zip(bars, total_times):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        speedup = total_times[0] / total_times[1] if total_times[1] > 0 else 0
        axes[1].text(0.5, 0.95, f'Speedup: {speedup:.2f}x', transform=axes[1].transAxes,
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Embedding Time Distribution (histogram)
    if pytorch_results and tensorrt_results:
        if 'embedding_times' in pytorch_results:
            axes[2].hist(pytorch_results['embedding_times'], bins=30, alpha=0.6, 
                        label='PyTorch', color=colors[0])
        if 'embedding_times' in tensorrt_results:
            axes[2].hist(tensorrt_results['embedding_times'], bins=30, alpha=0.6,
                        label='TensorRT', color=colors[1])
        
        axes[2].set_xlabel('Time (ms)', fontsize=11)
        axes[2].set_ylabel('Frequency', fontsize=11)
        axes[2].set_title('Embedding Time Distribution', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pytorch_vs_tensorrt_timing.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_embedding_similarity(embedding_comparison: Dict, output_dir: str):
    """임베딩 유사도 분포"""
    if not embedding_comparison or 'cosine_similarities' not in embedding_comparison:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cosine similarity
    ax1.hist(embedding_comparison['cosine_similarities'], bins=30, alpha=0.7, color='green')
    ax1.axvline(embedding_comparison['avg_cosine_similarity'], color='red',
               linestyle='--', linewidth=2,
               label=f"Mean: {embedding_comparison['avg_cosine_similarity']:.6f}")
    ax1.set_xlabel('Cosine Similarity', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('PyTorch vs TensorRT Embedding Similarity', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # L2 distance
    ax2.hist(embedding_comparison['l2_distances'], bins=30, alpha=0.7, color='orange')
    ax2.axvline(embedding_comparison['avg_l2_distance'], color='red',
               linestyle='--', linewidth=2,
               label=f"Mean: {embedding_comparison['avg_l2_distance']:.6f}")
    ax2.set_xlabel('L2 Distance', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('PyTorch vs TensorRT Embedding Distance', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pytorch_vs_tensorrt_embedding_similarity.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# 9. 메인 함수
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LFW: PyTorch vs TensorRT Comparison')
    parser.add_argument('--lfw-dir', type=str, default=LFW_DIR,
                       help='Path to LFW dataset')
    parser.add_argument('--pairs-file', type=str, default=PAIRS_FILE,
                       help='Path to pairs file')
    parser.add_argument('--pytorch-model', type=str, default=PYTORCH_MODEL_PATH,
                       help='Path to PyTorch model')
    parser.add_argument('--tensorrt-model', type=str, default=TENSORRT_MODEL_PATH,
                       help='Path to TensorRT engine')
    parser.add_argument('--max-pairs', type=int, default=None,
                       help='Maximum pairs to evaluate (None for all)')
    parser.add_argument('--embedding-samples', type=int, default=200,
                       help='Number of samples for embedding comparison')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("LFW Evaluation: PyTorch vs TensorRT")
    print("=" * 60)
    
    # Check paths
    if not os.path.exists(args.lfw_dir):
        print(f"❌ LFW directory not found: {args.lfw_dir}")
        return 1
    
    if not os.path.exists(args.pairs_file):
        print(f"❌ Pairs file not found: {args.pairs_file}")
        return 1
    
    # Load pairs
    pairs = load_lfw_pairs(args.pairs_file, args.lfw_dir)
    if not pairs:
        print("❌ No pairs loaded")
        return 1
    
    # Load models
    pytorch_model = load_pytorch_model(args.pytorch_model, MODEL_NAME, device)
    tensorrt_model = load_tensorrt_model(args.tensorrt_model, MODEL_NAME)
    
    if not pytorch_model and not tensorrt_model:
        print("❌ No models available")
        return 1
    
    # Load YuNet detector
    print(f"\nLoading YuNet detector...")
    if not os.path.exists(YUNET_MODEL_PATH):
        print(f"❌ YuNet model not found: {YUNET_MODEL_PATH}")
        return 1
    
    detector = YuNetDetector(model_path=YUNET_MODEL_PATH)
    print(f"✅ YuNet detector loaded")
    
    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'lfw_dir': args.lfw_dir,
        'pairs_file': args.pairs_file,
        'num_pairs': len(pairs),
        'max_pairs': args.max_pairs,
        'pytorch_available': pytorch_model is not None,
        'tensorrt_available': tensorrt_model is not None
    }
    
    pytorch_results = None
    tensorrt_results = None
    embedding_comparison = None
    
    # Evaluate PyTorch pipeline
    if pytorch_model:
        pytorch_results = evaluate_pipeline(
            pairs, detector, pytorch_model, "YuNet + EdgeFace PyTorch",
            device=device, use_tensorrt=False, max_pairs=args.max_pairs
        )
        if pytorch_results:
            results['pytorch'] = {k: v for k, v in pytorch_results.items() 
                                 if k not in ['similarities', 'labels', 'fpr', 'tpr', 
                                             'thresholds', 'total_times', 'embedding_times']}
    
    # Evaluate TensorRT pipeline
    if tensorrt_model:
        tensorrt_results = evaluate_pipeline(
            pairs, detector, tensorrt_model, "YuNet + EdgeFace TensorRT",
            device=device, use_tensorrt=True, max_pairs=args.max_pairs
        )
        if tensorrt_results:
            results['tensorrt'] = {k: v for k, v in tensorrt_results.items()
                                  if k not in ['similarities', 'labels', 'fpr', 'tpr',
                                              'thresholds', 'total_times', 'embedding_times']}
    
    # Compare embeddings
    if pytorch_model and tensorrt_model:
        embedding_comparison = compare_embeddings(
            pairs, detector, pytorch_model, tensorrt_model,
            device, max_pairs=args.embedding_samples
        )
        if embedding_comparison:
            results['embedding_comparison'] = {k: v for k, v in embedding_comparison.items()
                                              if k not in ['cosine_similarities', 'l2_distances',
                                                          'pytorch_times', 'tensorrt_times']}
    
    # Generate plots
    print(f"\n📈 Generating plots...")
    
    if pytorch_results or tensorrt_results:
        plot_roc_comparison(pytorch_results, tensorrt_results, args.output)
        plot_metrics_comparison(pytorch_results, tensorrt_results, args.output)
        plot_timing_comparison(pytorch_results, tensorrt_results, embedding_comparison, args.output)
    
    if embedding_comparison:
        plot_embedding_similarity(embedding_comparison, args.output)
    
    # Save JSON results
    json_path = os.path.join(args.output, 'lfw_comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if pytorch_results:
        print(f"\n📌 PyTorch:")
        print(f"   ROC AUC: {pytorch_results['roc_auc']:.4f}")
        print(f"   Best Accuracy: {pytorch_results['best_accuracy']:.4f}")
        print(f"   EER: {pytorch_results['eer']:.4f}")
        print(f"   Avg Embedding Time: {pytorch_results['avg_embedding_time_ms']:.2f} ms/pair")
        print(f"   Avg Total Time: {pytorch_results['avg_total_time_ms']:.2f} ms/pair")
    
    if tensorrt_results:
        print(f"\n📌 TensorRT:")
        print(f"   ROC AUC: {tensorrt_results['roc_auc']:.4f}")
        print(f"   Best Accuracy: {tensorrt_results['best_accuracy']:.4f}")
        print(f"   EER: {tensorrt_results['eer']:.4f}")
        print(f"   Avg Embedding Time: {tensorrt_results['avg_embedding_time_ms']:.2f} ms/pair")
        print(f"   Avg Total Time: {tensorrt_results['avg_total_time_ms']:.2f} ms/pair")
    
    if pytorch_results and tensorrt_results:
        emb_speedup = pytorch_results['avg_embedding_time_ms'] / tensorrt_results['avg_embedding_time_ms']
        total_speedup = pytorch_results['avg_total_time_ms'] / tensorrt_results['avg_total_time_ms']
        print(f"\n🚀 TensorRT Speedup:")
        print(f"   Embedding: {emb_speedup:.2f}x faster")
        print(f"   Total: {total_speedup:.2f}x faster")
    
    if embedding_comparison and 'avg_cosine_similarity' in embedding_comparison:
        print(f"\n🔗 Embedding Similarity (PyTorch vs TensorRT):")
        print(f"   Cosine Similarity: {embedding_comparison['avg_cosine_similarity']:.6f}")
        print(f"   L2 Distance: {embedding_comparison['avg_l2_distance']:.6f}")
    
    print("\n" + "=" * 60)
    print(f"✅ Results saved to: {args.output}/")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
