"""
YuNet + EdgeFace CPU vs NPU 종합 평가

이 스크립트는 YuNet 얼굴 검출기와 EdgeFace 임베딩 모델의 CPU/NPU 버전을 LFW 데이터셋으로 종합 평가합니다.

평가 목표:
1. YuNet 검출 성능 비교: CPU vs NPU 얼굴 검출 정확도, landmark 정확도
2. EdgeFace 임베딩 비교: PyTorch vs NPU 임베딩 유사도
3. End-to-End 파이프라인: YuNet+EdgeFace 전체 파이프라인 성능 비교

평가 메트릭:
- YuNet 검출기: Detection Rate, Landmark Accuracy, Detection Threshold 영향 분석
- EdgeFace 임베딩: Embedding Similarity, Embedding Distance
- End-to-End 파이프라인: ROC AUC, Best Accuracy, EER, Cross-compatibility
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict

# matplotlib 백엔드를 Agg로 설정 (헤드리스 환경에서 실행 가능)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import cv2

# PyTorch 관련
import torch
from torchvision import transforms

# EdgeFace 백본 모델 임포트
sys.path.insert(0, 'face_alignment')
from backbones import get_model

# YuNet 검출기 임포트
from face_alignment.yunet import YuNetDetector
try:
    from face_alignment.yunet_npu import YuNetNPUDetector
    YUNET_NPU_AVAILABLE = True
except ImportError:
    print("⚠ YuNet NPU not available")
    YUNET_NPU_AVAILABLE = False

# EdgeFace NPU 임포트
try:
    from edgeface_npu_recognizer import EdgeFaceNPURecognizer
    EDGEFACE_NPU_AVAILABLE = True
except ImportError:
    print("⚠ EdgeFace NPU not available")
    EDGEFACE_NPU_AVAILABLE = False

# ============================================================================
# 1. 환경 설정 및 초기화
# ============================================================================

print(f"PyTorch version: {torch.__version__}")
print(f"YuNet NPU available: {YUNET_NPU_AVAILABLE}")
print(f"EdgeFace NPU available: {EDGEFACE_NPU_AVAILABLE}")

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# LFW 데이터셋 경로 설정
lfw_dir = "/home/dxdemo/Downloads/lfw"
pairs_file = "/home/dxdemo/Downloads/lfw/pairs.txt"

# 만약 txt 파일이 없고 csv 파일이 있다면 변경
if not os.path.exists(pairs_file) and os.path.exists("/home/dxdemo/Downloads/lfw/pairs.csv"):
    pairs_file = "/home/dxdemo/Downloads/lfw/pairs.csv"

# 모델 경로 설정
YUNET_CPU_MODEL = "face_alignment/models/face_detection_yunet_2023mar.onnx"
YUNET_NPU_MODEL = "face_alignment/models/face_detection_yunet_2023mar.dxnn"
EDGEFACE_PYTORCH_MODEL = "checkpoints/edgeface_xs_gamma_06.pt"
EDGEFACE_NPU_MODEL = "checkpoints/edgeface_xs_gamma_06.dxnn"

# 경로 확인
if not os.path.exists(lfw_dir):
    print(f"⚠ Warning: LFW directory not found at {lfw_dir}")
else:
    print(f"✓ LFW directory found: {lfw_dir}")

if not os.path.exists(pairs_file):
    print(f"⚠ Warning: Pairs file not found at {pairs_file}")
else:
    print(f"✓ Pairs file found: {pairs_file}")


# ============================================================================
# 2. 데이터 로딩 함수
# ============================================================================

def load_lfw_pairs(pairs_file: str, lfw_dir: str) -> List[Tuple]:
    """
    LFW pairs 파일을 로드합니다.

    Returns:
        List of (is_same, img1_path, img2_path) tuples
    """
    pairs = []

    if not os.path.exists(pairs_file):
        return pairs

    is_csv = pairs_file.endswith('.csv')
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
        
    start_idx = 1 # Skip header line for both csv and txt
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue

        # Split by comma if CSV, otherwise by whitespace (tabs/spaces)
        parts = line.rstrip(',').split(',') if is_csv else line.split()

        if len(parts) == 3:
            # 같은 사람 쌍: Name, img1, img2
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
            # 다른 사람 쌍: Name1, img1, Name2, img2
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

    return pairs


# ============================================================================
# 3. 임베딩 추출 함수
# ============================================================================

def extract_embedding_pytorch(face_img: np.ndarray, model: torch.nn.Module, device: str) -> np.ndarray:
    """
    PyTorch 모델로 임베딩 추출

    Args:
        face_img: BGR 이미지 (112x112x3)
        model: PyTorch 모델
        device: 디바이스

    Returns:
        512-d embedding vector
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

    # Extract embedding
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy().flatten()

    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

def extract_embedding_pytorch_timed(face_img: np.ndarray, model: torch.nn.Module, device: str) -> Tuple[np.ndarray, float]:
    import time
    if face_img.shape[:2] != (112, 112):
        face_img = cv2.resize(face_img, (112, 112))
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img_tensor.div_(255).sub_(0.5).div_(0.5)

    try:
        start_time = time.perf_counter()
        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy().flatten()
        pure_time = time.perf_counter() - start_time
    except Exception as e:
        print(f"EdgeFace PyTorch inference error: {e}")
        raise

    embedding = embedding / np.linalg.norm(embedding)
    return embedding, pure_time


def extract_embedding_npu(face_img: np.ndarray, recognizer) -> np.ndarray:
    """
    NPU 모델로 임베딩 추출

    Args:
        face_img: BGR 이미지 (112x112x3)
        recognizer: EdgeFaceNPURecognizer

    Returns:
        512-d embedding vector
    """
    return recognizer.extract_embedding(face_img)


# ============================================================================
# 4. YuNet 검출 성능 평가
# ============================================================================

def evaluate_yunet_detection(pairs: List[Tuple], detector, detector_name: str, max_pairs: int = 500) -> Dict:
    """
    YuNet 검출 성능 평가

    Returns:
        Dict with detection statistics
    """
    results = {
        'detector': detector_name,
        'total_images': 0,
        'detected': 0,
        'failed': 0,
        'detection_times': [],
        'face_counts': [],  # Number of faces detected per image
        'confidences': [],
        'aligned_faces': []  # Store aligned faces for later use
    }

    print(f"\n=== Evaluating {detector_name} Detection ===")

    # Sample pairs for evaluation
    sampled_pairs = pairs[:max_pairs] if max_pairs else pairs

    for is_same, img1_path, img2_path in tqdm(sampled_pairs, desc=f"{detector_name} Detection"):
        for img_path in [img1_path, img2_path]:
            if not os.path.exists(img_path):
                results['failed'] += 1
                continue

            results['total_images'] += 1

            try:
                # Load image
                pil_img = Image.open(img_path).convert('RGB')

                # Detect face
                start_time = time.time()
                aligned_face = detector.align(pil_img)
                detection_time = time.time() - start_time

                results['detection_times'].append(detection_time)

                if aligned_face is not None:
                    results['detected'] += 1
                    results['aligned_faces'].append(aligned_face)

                    # Get detection info (if available)
                    faces = detector.detect_faces(pil_img)
                    if faces is not None and len(faces) > 0:
                        results['face_counts'].append(len(faces))
                        # Get confidence of best face
                        confidences = [f[-1] for f in faces]
                        results['confidences'].append(max(confidences))
                else:
                    results['failed'] += 1

            except Exception as e:
                results['failed'] += 1

    # Calculate statistics
    results['detection_rate'] = results['detected'] / results['total_images'] if results['total_images'] > 0 else 0
    results['avg_detection_time'] = np.mean(results['detection_times']) if results['detection_times'] else 0
    results['avg_faces_per_image'] = np.mean(results['face_counts']) if results['face_counts'] else 0
    results['avg_confidence'] = np.mean(results['confidences']) if results['confidences'] else 0

    print(f"\n{detector_name} Results:")
    print(f"  Total images: {results['total_images']}")
    print(f"  Detected: {results['detected']} ({results['detection_rate']:.2%})")
    print(f"  Failed: {results['failed']}")
    print(f"  Avg detection time: {results['avg_detection_time']:.4f}s")
    print(f"  Avg faces per image: {results['avg_faces_per_image']:.2f}")
    print(f"  Avg confidence: {results['avg_confidence']:.4f}")

    return results


# ============================================================================
# 5. Landmark 비교 함수
# ============================================================================

def compare_landmarks(pairs: List[Tuple], detector_cpu, detector_npu, max_pairs: int = 100) -> Dict:
    """
    CPU와 NPU의 landmark 차이 분석

    Returns:
        Dict with landmark comparison statistics
    """
    results = {
        'landmark_distances': [],  # L2 distance between CPU and NPU landmarks
        'landmark_distances_per_point': [[] for _ in range(5)],  # Per landmark point
        'both_detected': 0,
        'only_cpu_detected': 0,
        'only_npu_detected': 0,
        'neither_detected': 0
    }

    print(f"\n=== Comparing Landmarks (CPU vs NPU) ===")

    sampled_pairs = pairs[:max_pairs] if max_pairs else pairs

    for is_same, img1_path, img2_path in tqdm(sampled_pairs, desc="Landmark Comparison"):
        for img_path in [img1_path, img2_path]:
            if not os.path.exists(img_path):
                continue

            try:
                pil_img = Image.open(img_path).convert('RGB')

                # CPU detection with landmarks
                aligned_cpu, landmarks_cpu = detector_cpu.align(pil_img, return_landmarks=True)

                # NPU detection with landmarks
                aligned_npu, landmarks_npu = detector_npu.align(pil_img, return_landmarks=True)

                # Check detection status
                if aligned_cpu is not None and aligned_npu is not None:
                    results['both_detected'] += 1

                    # Compare landmarks
                    if landmarks_cpu is not None and landmarks_npu is not None:
                        landmarks_cpu = np.array(landmarks_cpu)
                        landmarks_npu = np.array(landmarks_npu)

                        # Overall L2 distance
                        dist = np.linalg.norm(landmarks_cpu - landmarks_npu)
                        results['landmark_distances'].append(dist)

                        # Per-point distance
                        for i in range(min(5, len(landmarks_cpu))):
                            point_dist = np.linalg.norm(landmarks_cpu[i] - landmarks_npu[i])
                            results['landmark_distances_per_point'][i].append(point_dist)

                elif aligned_cpu is not None:
                    results['only_cpu_detected'] += 1
                elif aligned_npu is not None:
                    results['only_npu_detected'] += 1
                else:
                    results['neither_detected'] += 1

            except Exception as e:
                continue

    # Calculate statistics
    if results['landmark_distances']:
        results['avg_landmark_distance'] = np.mean(results['landmark_distances'])
        results['std_landmark_distance'] = np.std(results['landmark_distances'])
        results['max_landmark_distance'] = np.max(results['landmark_distances'])

        # Per-point statistics
        landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
        results['per_point_stats'] = []
        for i, name in enumerate(landmark_names):
            if results['landmark_distances_per_point'][i]:
                results['per_point_stats'].append({
                    'name': name,
                    'avg': np.mean(results['landmark_distances_per_point'][i]),
                    'std': np.std(results['landmark_distances_per_point'][i]),
                    'max': np.max(results['landmark_distances_per_point'][i])
                })

    print(f"\nLandmark Comparison Results:")
    print(f"  Both detected: {results['both_detected']}")
    print(f"  Only CPU detected: {results['only_cpu_detected']}")
    print(f"  Only NPU detected: {results['only_npu_detected']}")
    print(f"  Neither detected: {results['neither_detected']}")
    if results['landmark_distances']:
        print(f"  Avg landmark distance: {results['avg_landmark_distance']:.2f} pixels")
        print(f"  Max landmark distance: {results['max_landmark_distance']:.2f} pixels")

    return results


# ============================================================================
# 6. 임베딩 비교 함수
# ============================================================================

def compare_embeddings(pairs: List[Tuple], detector, edgeface_pytorch, edgeface_npu,
                       device: str, detector_name: str, max_pairs: int = 100) -> Dict:
    """
    같은 얼굴에 대한 PyTorch vs NPU 임베딩 비교

    Args:
        pairs: Image pairs
        detector: YuNet detector (CPU or NPU)
        edgeface_pytorch: EdgeFace PyTorch model
        edgeface_npu: EdgeFace NPU model
        device: PyTorch device
        detector_name: Name of detector for logging
        max_pairs: Maximum pairs to evaluate

    Returns:
        Dict with embedding comparison statistics
    """
    results = {
        'detector': detector_name,
        'cosine_similarities': [],  # Cosine similarity between PyTorch and NPU embeddings
        'l2_distances': [],  # L2 distance
        'pytorch_times': [],
        'npu_times': [],
        'pytorch_pure_times': [],
        'npu_pure_times': [],
        'pytorch_embeddings': [],  # Store embeddings for distribution analysis
        'npu_embeddings': [],
        'valid_pairs': 0
    }

    print(f"\n=== Comparing EdgeFace Embeddings (PyTorch vs NPU) with {detector_name} ===")

    sampled_pairs = pairs[:max_pairs] if max_pairs else pairs

    for is_same, img1_path, img2_path in tqdm(sampled_pairs, desc=f"Embedding Comparison ({detector_name})"):
        for img_path in [img1_path, img2_path]:
            if not os.path.exists(img_path):
                continue

            try:
                # Detect and align face
                pil_img = Image.open(img_path).convert('RGB')
                aligned_face = detector.align(pil_img)

                if aligned_face is None:
                    continue

                # Convert to numpy BGR
                face_np = np.array(aligned_face)
                face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

                # Extract PyTorch embedding
                start_time = time.perf_counter()
                emb_pytorch, pure_pt = extract_embedding_pytorch_timed(face_np, edgeface_pytorch, device)
                pytorch_time = time.perf_counter() - start_time
                results['pytorch_times'].append(pytorch_time)
                results['pytorch_pure_times'].append(pure_pt)
                results['pytorch_embeddings'].append(emb_pytorch)

                # Extract NPU embedding
                start_time = time.perf_counter()
                try:
                    if hasattr(edgeface_npu, 'extract_embedding_timed'):
                        emb_npu, pure_npu = edgeface_npu.extract_embedding_timed(face_np)
                    else:
                        emb_npu = extract_embedding_npu(face_np, edgeface_npu)
                        pure_npu = 0.0
                except:
                    emb_npu = extract_embedding_npu(face_np, edgeface_npu)
                    pure_npu = 0.0

                npu_time = time.perf_counter() - start_time
                results['npu_times'].append(npu_time)
                results['npu_pure_times'].append(pure_npu)
                results['npu_embeddings'].append(emb_npu)

                # Compare embeddings
                cosine_sim = np.dot(emb_pytorch, emb_npu)
                l2_dist = np.linalg.norm(emb_pytorch - emb_npu)

                results['cosine_similarities'].append(cosine_sim)
                results['l2_distances'].append(l2_dist)
                results['valid_pairs'] += 1

            except Exception as e:
                continue

    # Convert to numpy arrays for analysis
    results['pytorch_embeddings'] = np.array(results['pytorch_embeddings'])
    results['npu_embeddings'] = np.array(results['npu_embeddings'])

    # Calculate statistics
    if results['cosine_similarities']:
        results['avg_cosine_similarity'] = np.mean(results['cosine_similarities'])
        results['std_cosine_similarity'] = np.std(results['cosine_similarities'])
        results['min_cosine_similarity'] = np.min(results['cosine_similarities'])
        results['avg_l2_distance'] = np.mean(results['l2_distances'])
        results['avg_pytorch_time'] = np.mean(results['pytorch_times'])
        results['avg_npu_time'] = np.mean(results['npu_times'])
        results['avg_pytorch_pure_time'] = np.mean(results['pytorch_pure_times']) if results.get('pytorch_pure_times') else 0
        results['avg_npu_pure_time'] = np.mean(results['npu_pure_times']) if results.get('npu_pure_times') else 0

    print(f"\nEmbedding Comparison Results ({detector_name}):")
    print(f"  Valid pairs: {results['valid_pairs']}")
    if results['cosine_similarities']:
        print(f"  Avg cosine similarity: {results['avg_cosine_similarity']:.4f}")
        print(f"  Min cosine similarity: {results['min_cosine_similarity']:.4f}")
        print(f"  Avg L2 distance: {results['avg_l2_distance']:.4f}")
        print(f"  Avg PyTorch total time: {results['avg_pytorch_time']:.4f}s (Pure Inference: {results['avg_pytorch_pure_time']:.4f}s)")
        print(f"  Avg NPU total time: {results['avg_npu_time']:.4f}s (Pure Inference: {results['avg_npu_pure_time']:.4f}s)")

    return results


# ============================================================================
# 7. End-to-End 파이프라인 평가
# ============================================================================

def evaluate_pipeline(pairs: List[Tuple], detector, recognizer, pipeline_name: str,
                     device: str = 'cuda', max_pairs: Optional[int] = None) -> Dict:
    """
    End-to-end 파이프라인 평가

    Returns:
        Dict with evaluation results (ROC AUC, accuracy, etc.)
    """
    similarities = []
    labels = []
    processing_times = []
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

    print(f"\n=== Evaluating {pipeline_name} ===")
    print(f"Processing {len(pairs_to_process)} pairs...")

    # Determine if using NPU or PyTorch recognizer
    is_npu_recognizer = hasattr(recognizer, 'inference_engine')

    for is_same, img1_path, img2_path in tqdm(pairs_to_process, desc=pipeline_name):
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            failed_count += 1
            continue

        start_time = time.time()

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

            # Extract embeddings
            if is_npu_recognizer:
                emb1 = extract_embedding_npu(face1_np, recognizer)
                emb2 = extract_embedding_npu(face2_np, recognizer)
            else:
                emb1 = extract_embedding_pytorch(face1_np, recognizer, device)
                emb2 = extract_embedding_pytorch(face2_np, recognizer, device)

            # Calculate similarity
            similarity = np.dot(emb1, emb2)
            similarities.append(similarity)
            labels.append(1 if is_same else 0)

            processing_times.append(time.time() - start_time)

        except Exception as e:
            failed_count += 1
            continue

    # Calculate metrics
    similarities = np.array(similarities)
    labels = np.array(labels)

    if len(similarities) == 0:
        print("No valid pairs processed!")
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
        'avg_processing_time': np.mean(processing_times),
        'similarities': similarities,
        'labels': labels,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

    print(f"\n{pipeline_name} Results:")
    print(f"  Pairs processed: {results['num_pairs']}")
    print(f"  Success rate: {results['success_rate']:.4f}")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")
    print(f"  Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"  Best Threshold: {results['best_threshold']:.4f}")
    print(f"  EER: {results['eer']:.4f}")
    print(f"  Avg Processing Time: {results['avg_processing_time']:.4f}s")

    return results


# ============================================================================
# 8. 시각화 함수
# ============================================================================

def plot_yunet_comparison(yunet_cpu_results: Dict, yunet_npu_results: Dict):
    """YuNet CPU vs NPU 검출 성능 비교 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('Detection Rate', 'detection_rate', axes[0, 0]),
        ('Avg Detection Time (s)', 'avg_detection_time', axes[0, 1]),
        ('Avg Faces Per Image', 'avg_faces_per_image', axes[1, 0]),
        ('Avg Confidence', 'avg_confidence', axes[1, 1])
    ]

    for metric_name, metric_key, ax in metrics:
        cpu_val = yunet_cpu_results[metric_key]
        npu_val = yunet_npu_results[metric_key]

        bars = ax.bar(['CPU', 'NPU'], [cpu_val, npu_val],
                      color=['blue', 'red'], alpha=0.7)

        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'YuNet {metric_name}', fontsize=12)

        # 값 레이블
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)

        # 차이 표시
        diff = npu_val - cpu_val
        diff_pct = (diff / cpu_val * 100) if cpu_val != 0 else 0
        ax.text(0.5, 0.95, f'Diff: {diff:.4f} ({diff_pct:+.2f}%)',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)

    plt.tight_layout()
    plt.savefig('yunet_cpu_vs_npu_detection.png', dpi=300, bbox_inches='tight')
    print("YuNet detection comparison saved to: yunet_cpu_vs_npu_detection.png")
    plt.close()


def plot_landmark_comparison(landmark_comparison: Dict):
    """Landmark 차이 시각화"""
    if not landmark_comparison or 'per_point_stats' not in landmark_comparison:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Per-point average distance
    names = [stat['name'] for stat in landmark_comparison['per_point_stats']]
    avgs = [stat['avg'] for stat in landmark_comparison['per_point_stats']]
    stds = [stat['std'] for stat in landmark_comparison['per_point_stats']]

    ax1.bar(names, avgs, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_ylabel('Avg Distance (pixels)', fontsize=11)
    ax1.set_title('Landmark Distance: CPU vs NPU (per point)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Overall distance distribution
    ax2.hist(landmark_comparison['landmark_distances'], bins=30, alpha=0.7, color='coral')
    ax2.axvline(landmark_comparison['avg_landmark_distance'], color='red',
                linestyle='--', linewidth=2, label=f"Mean: {landmark_comparison['avg_landmark_distance']:.2f}")
    ax2.set_xlabel('Total Landmark Distance (pixels)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Landmark Distances', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('yunet_landmark_comparison.png', dpi=300, bbox_inches='tight')
    print("Landmark comparison saved to: yunet_landmark_comparison.png")
    plt.close()


def plot_embedding_comparison(embedding_cpu: Dict, embedding_npu: Dict = None):
    """
    임베딩 비교 시각화

    Args:
        embedding_cpu: CPU YuNet으로 전처리한 임베딩 비교 결과
        embedding_npu: NPU YuNet으로 전처리한 임베딩 비교 결과 (optional)
    """
    if not embedding_cpu or 'cosine_similarities' not in embedding_cpu:
        return

    # Single detector case
    if embedding_npu is None or 'cosine_similarities' not in embedding_npu:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Cosine similarity distribution
        ax1.hist(embedding_cpu['cosine_similarities'], bins=30, alpha=0.7, color='green')
        ax1.axvline(embedding_cpu['avg_cosine_similarity'], color='red',
                    linestyle='--', linewidth=2,
                    label=f"Mean: {embedding_cpu['avg_cosine_similarity']:.4f}")
        ax1.set_xlabel('Cosine Similarity', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('PyTorch vs NPU Embedding Similarity', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # L2 distance distribution
        ax2.hist(embedding_cpu['l2_distances'], bins=30, alpha=0.7, color='orange')
        ax2.axvline(embedding_cpu['avg_l2_distance'], color='red',
                    linestyle='--', linewidth=2,
                    label=f"Mean: {embedding_cpu['avg_l2_distance']:.4f}")
        ax2.set_xlabel('L2 Distance', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('PyTorch vs NPU Embedding Distance', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('edgeface_pytorch_vs_npu_embeddings.png', dpi=300, bbox_inches='tight')
        print("Embedding comparison saved to: edgeface_pytorch_vs_npu_embeddings.png")
        plt.close()
        return

    # Comparison with both CPU and NPU detectors
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: CPU YuNet 전처리
    # Cosine similarity
    axes[0, 0].hist(embedding_cpu['cosine_similarities'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].axvline(embedding_cpu['avg_cosine_similarity'], color='red',
                       linestyle='--', linewidth=2,
                       label=f"Mean: {embedding_cpu['avg_cosine_similarity']:.4f}")
    axes[0, 0].set_xlabel('Cosine Similarity', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title(f'CPU YuNet: PyTorch vs NPU Similarity', fontsize=11)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # L2 distance
    axes[0, 1].hist(embedding_cpu['l2_distances'], bins=30, alpha=0.7, color='green')
    axes[0, 1].axvline(embedding_cpu['avg_l2_distance'], color='red',
                       linestyle='--', linewidth=2,
                       label=f"Mean: {embedding_cpu['avg_l2_distance']:.4f}")
    axes[0, 1].set_xlabel('L2 Distance', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title(f'CPU YuNet: PyTorch vs NPU Distance', fontsize=11)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Embedding distribution (first 3 dimensions)
    if len(embedding_cpu['pytorch_embeddings']) > 0:
        axes[0, 2].scatter(embedding_cpu['pytorch_embeddings'][:, 0],
                          embedding_cpu['pytorch_embeddings'][:, 1],
                          alpha=0.5, s=10, label='PyTorch', color='blue')
        axes[0, 2].scatter(embedding_cpu['npu_embeddings'][:, 0],
                          embedding_cpu['npu_embeddings'][:, 1],
                          alpha=0.5, s=10, label='NPU', color='red')
        axes[0, 2].set_xlabel('Dim 0', fontsize=10)
        axes[0, 2].set_ylabel('Dim 1', fontsize=10)
        axes[0, 2].set_title('CPU YuNet: Embedding Space (first 2D)', fontsize=11)
        axes[0, 2].legend(fontsize=9)
        axes[0, 2].grid(True, alpha=0.3)

    # Row 2: NPU YuNet 전처리
    # Cosine similarity
    axes[1, 0].hist(embedding_npu['cosine_similarities'], bins=30, alpha=0.7, color='blue')
    axes[1, 0].axvline(embedding_npu['avg_cosine_similarity'], color='red',
                       linestyle='--', linewidth=2,
                       label=f"Mean: {embedding_npu['avg_cosine_similarity']:.4f}")
    axes[1, 0].set_xlabel('Cosine Similarity', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title(f'NPU YuNet: PyTorch vs NPU Similarity', fontsize=11)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # L2 distance
    axes[1, 1].hist(embedding_npu['l2_distances'], bins=30, alpha=0.7, color='green')
    axes[1, 1].axvline(embedding_npu['avg_l2_distance'], color='red',
                       linestyle='--', linewidth=2,
                       label=f"Mean: {embedding_npu['avg_l2_distance']:.4f}")
    axes[1, 1].set_xlabel('L2 Distance', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].set_title(f'NPU YuNet: PyTorch vs NPU Distance', fontsize=11)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # Embedding distribution (first 2 dimensions)
    if len(embedding_npu['pytorch_embeddings']) > 0:
        axes[1, 2].scatter(embedding_npu['pytorch_embeddings'][:, 0],
                          embedding_npu['pytorch_embeddings'][:, 1],
                          alpha=0.5, s=10, label='PyTorch', color='blue')
        axes[1, 2].scatter(embedding_npu['npu_embeddings'][:, 0],
                          embedding_npu['npu_embeddings'][:, 1],
                          alpha=0.5, s=10, label='NPU', color='red')
        axes[1, 2].set_xlabel('Dim 0', fontsize=10)
        axes[1, 2].set_ylabel('Dim 1', fontsize=10)
        axes[1, 2].set_title('NPU YuNet: Embedding Space (first 2D)', fontsize=11)
        axes[1, 2].legend(fontsize=9)
        axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('EdgeFace Embedding Comparison: Impact of YuNet Detector',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('edgeface_pytorch_vs_npu_embeddings_comparison.png', dpi=300, bbox_inches='tight')
    print("Embedding comparison saved to: edgeface_pytorch_vs_npu_embeddings_comparison.png")
    plt.close()


def plot_roc_comparison(pipeline_results: Dict):
    """ROC 커브 비교"""
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange']
    linestyles = ['-', '-', '--', '--']

    for (name, result), color, linestyle in zip(pipeline_results.items(), colors, linestyles):
        if result is not None:
            plt.plot(result['fpr'], result['tpr'],
                    label=f"{result['pipeline']} (AUC = {result['roc_auc']:.4f})",
                    linewidth=2, color=color, linestyle=linestyle)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Pipeline Comparison', fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('pipeline_roc_comparison.png', dpi=300, bbox_inches='tight')
    print("ROC comparison saved to: pipeline_roc_comparison.png")
    plt.close()


def plot_metrics_comparison(pipeline_results: Dict):
    """성능 메트릭 비교 막대 그래프"""
    metrics_to_plot = ['roc_auc', 'best_accuracy', 'eer']
    metric_names = ['ROC AUC', 'Best Accuracy', 'EER']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, metric_name in zip(axes, metrics_to_plot, metric_names):
        pipelines = []
        values = []

        for name, result in pipeline_results.items():
            if result is not None:
                pipelines.append(name)
                values.append(result[metric])

        bars = ax.bar(pipelines, values, alpha=0.7)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Comparison', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # 값 레이블
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('pipeline_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Metrics comparison saved to: pipeline_metrics_comparison.png")
    plt.close()


# ============================================================================
# 9. 메인 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""

    # Pairs 로드
    if not os.path.exists(pairs_file):
        print(f"Error: Pairs file not found at {pairs_file}")
        return

    pairs = load_lfw_pairs(pairs_file, lfw_dir)
    print(f"\nLoaded {len(pairs)} pairs from LFW")

    positive_pairs = sum(1 for p in pairs if p[0])
    negative_pairs = len(pairs) - positive_pairs
    print(f"  Positive pairs (same person): {positive_pairs}")
    print(f"  Negative pairs (different person): {negative_pairs}")

    # 모델 초기화
    print("\n=== Initializing Models ===")

    # YuNet CPU 검출기
    print("\n=== Initializing YuNet CPU ===")
    yunet_cpu = YuNetDetector(YUNET_CPU_MODEL, device='cpu', crop_size=(112, 112))

    # YuNet NPU 검출기
    yunet_npu = None
    if YUNET_NPU_AVAILABLE and os.path.exists(YUNET_NPU_MODEL):
        print("\n=== Initializing YuNet NPU ===")
        yunet_npu = YuNetNPUDetector(YUNET_NPU_MODEL, device='npu', crop_size=(112, 112))
    else:
        print("⚠ YuNet NPU model not available")

    # EdgeFace PyTorch 모델
    print("\n=== Initializing EdgeFace PyTorch ===")
    model_name = 'edgeface_xs_gamma_06'
    edgeface_pytorch = get_model(model_name, fp16=False)
    edgeface_pytorch.load_state_dict(torch.load(EDGEFACE_PYTORCH_MODEL, map_location=device))
    edgeface_pytorch.to(device)
    edgeface_pytorch.eval()
    print(f"✓ EdgeFace PyTorch loaded on {device}")

    # EdgeFace NPU 모델
    edgeface_npu = None
    if EDGEFACE_NPU_AVAILABLE and os.path.exists(EDGEFACE_NPU_MODEL):
        print("\n=== Initializing EdgeFace NPU ===")
        edgeface_npu = EdgeFaceNPURecognizer(EDGEFACE_NPU_MODEL, model_name, device='npu')
    else:
        print("⚠ EdgeFace NPU model not available")

    print("\n✓ All available models initialized")

    # YuNet 검출 성능 평가
    print("\n" + "="*80)
    print("YuNet Detection Evaluation")
    print("="*80)

    yunet_cpu_results = evaluate_yunet_detection(pairs, yunet_cpu, "YuNet CPU", max_pairs=None)

    yunet_npu_results = None
    if yunet_npu is not None:
        yunet_npu_results = evaluate_yunet_detection(pairs, yunet_npu, "YuNet NPU", max_pairs=None)

        # 비교 시각화
        try:
            plot_yunet_comparison(yunet_cpu_results, yunet_npu_results)
        except Exception as e:
            print(f"⚠ Skipping YuNet plot generation due to matplotlib error: {e}")

    # Landmark 비교
    if yunet_npu is not None:
        print("\n" + "="*80)
        print("Landmark Comparison")
        print("="*80)

        landmark_comparison = compare_landmarks(pairs, yunet_cpu, yunet_npu, max_pairs=100)
        try:
            plot_landmark_comparison(landmark_comparison)
        except Exception as e:
            print(f"⚠ Skipping landmark plot generation due to matplotlib error: {e}")

    # 임베딩 비교
    embedding_cpu_comparison = None
    embedding_npu_comparison = None

    if edgeface_npu is not None:
        print("\n" + "="*80)
        print("Embedding Comparison: EdgeFace Performance with Different YuNet Detectors")
        print("="*80)

        # CPU YuNet으로 전처리 후 EdgeFace PyTorch vs NPU 비교
        embedding_cpu_comparison = compare_embeddings(
            pairs, yunet_cpu, edgeface_pytorch, edgeface_npu, device,
            detector_name="YuNet CPU", max_pairs=500
        )

        # NPU YuNet으로 전처리 후 EdgeFace PyTorch vs NPU 비교
        if yunet_npu is not None:
            embedding_npu_comparison = compare_embeddings(
                pairs, yunet_npu, edgeface_pytorch, edgeface_npu, device,
                detector_name="YuNet NPU", max_pairs=500
            )

        # 두 결과 비교 시각화
        try:
            plot_embedding_comparison(embedding_cpu_comparison, embedding_npu_comparison)
        except Exception as e:
            print(f"⚠ Skipping embedding plot generation due to matplotlib error: {e}")

    # End-to-End 파이프라인 평가
    print("\n" + "="*80)
    print("End-to-End Pipeline Evaluation")
    print("="*80)

    pipeline_results = {}

    # 1. YuNet CPU + EdgeFace PyTorch
    pipeline_results['CPU_PyTorch'] = evaluate_pipeline(
        pairs, yunet_cpu, edgeface_pytorch, "YuNet CPU + EdgeFace PyTorch",
        device=device, max_pairs=1000
    )

    # 2. YuNet NPU + EdgeFace NPU
    if yunet_npu is not None and edgeface_npu is not None:
        pipeline_results['NPU_NPU'] = evaluate_pipeline(
            pairs, yunet_npu, edgeface_npu, "YuNet NPU + EdgeFace NPU",
            max_pairs=1000
        )

    # 3. YuNet CPU + EdgeFace NPU (cross-compatibility)
    if edgeface_npu is not None:
        pipeline_results['CPU_NPU'] = evaluate_pipeline(
            pairs, yunet_cpu, edgeface_npu, "YuNet CPU + EdgeFace NPU",
            max_pairs=1000
        )

    # 4. YuNet NPU + EdgeFace PyTorch (cross-compatibility)
    if yunet_npu is not None:
        pipeline_results['NPU_PyTorch'] = evaluate_pipeline(
            pairs, yunet_npu, edgeface_pytorch, "YuNet NPU + EdgeFace PyTorch",
            device=device, max_pairs=1000
        )

    # 결과 비교 테이블
    print("\n" + "="*80)
    print("Pipeline Performance Comparison")
    print("="*80)

    if pipeline_results:
        comparison_data = []
        for name, result in pipeline_results.items():
            if result is not None:
                comparison_data.append({
                    'Pipeline': result['pipeline'],
                    'Success Rate': f"{result['success_rate']:.4f}",
                    'ROC AUC': f"{result['roc_auc']:.4f}",
                    'Best Accuracy': f"{result['best_accuracy']:.4f}",
                    'Best Threshold': f"{result['best_threshold']:.4f}",
                    'EER': f"{result['eer']:.4f}",
                    'Avg Time (s)': f"{result['avg_processing_time']:.4f}"
                })

        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))

        # CSV로 저장
        comparison_df.to_csv('pipeline_comparison.csv', index=False)
        print("\nComparison saved to: pipeline_comparison.csv")

        # 시각화
        try:
            plot_roc_comparison(pipeline_results)
            plot_metrics_comparison(pipeline_results)
        except Exception as e:
            print(f"⚠ Skipping ROC/Metrics plot generation due to matplotlib error: {e}")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
