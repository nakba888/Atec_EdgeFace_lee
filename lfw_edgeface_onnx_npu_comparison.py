"""
LFW 데이터셋으로 EdgeFace ONNX vs NPU 임베딩 비교

YuNet CPU로 얼굴 디텍션 후 EdgeFace ONNX와 NPU 임베딩을 비교합니다.
- 코사인 유사도
- 임베딩 최소값/최대값 비교
"""

import os
import sys
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# YuNet detector
sys.path.insert(0, 'face_alignment')
from face_alignment.yunet import YuNetDetector

# EdgeFace NPU
from edgeface_npu_recognizer import EdgeFaceNPURecognizer

# ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("onnxruntime이 설치되지 않았습니다. pip install onnxruntime")
    ONNX_AVAILABLE = False


# ============================================================================
# 설정
# ============================================================================

# LFW 데이터셋 경로
LFW_DIR = "/home/dxdemo/Downloads/lfw"

# 모델 경로
YUNET_CPU_MODEL = "face_alignment/models/face_detection_yunet_2023mar.onnx"
EDGEFACE_ONNX_MODEL = "checkpoints/edgeface_s_gamma_05.onnx"
EDGEFACE_NPU_MODEL = "checkpoints/edgeface_s_gamma_05.dxnn"

# 샘플 수
NUM_SAMPLES = 10


# ============================================================================
# EdgeFace ONNX 추론 클래스
# ============================================================================

class EdgeFaceONNXRecognizer:
    """EdgeFace ONNX 기반 얼굴 인식"""

    def __init__(self, model_path: str):
        """
        Initialize EdgeFace ONNX recognizer

        Args:
            model_path: Path to EdgeFace ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime이 설치되어 있지 않습니다.")

        self.model_path = model_path
        self.input_size = (112, 112)

        # ONNX Runtime session 생성
        print(f"EdgeFace ONNX: Loading model from {model_path}...")
        self.session = ort.InferenceSession(model_path)

        # 입력/출력 정보
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"EdgeFace ONNX: Model loaded successfully")
        print(f"EdgeFace ONNX: Input name: {self.input_name}")
        print(f"EdgeFace ONNX: Output name: {self.output_name}")

    def _preprocess_image(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for EdgeFace ONNX inference

        Args:
            face_img: Aligned face image (112x112x3) in BGR format

        Returns:
            preprocessed: Preprocessed image tensor
        """
        # Resize if needed
        if face_img.shape[:2] != self.input_size:
            face_img = cv2.resize(face_img, self.input_size)

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Convert to float and normalize
        img_float = rgb_img.astype(np.float32) / 255.0

        # Normalize with mean=0.5, std=0.5
        img_normalized = (img_float - 0.5) / 0.5

        # Transpose to CHW format
        chw_img = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        input_tensor = np.expand_dims(chw_img, axis=0)

        return input_tensor.astype(np.float32)

    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from aligned face image

        Args:
            face_img: Aligned face image (112x112x3) in BGR format

        Returns:
            Face embedding vector (512-d)
        """
        # Preprocess
        input_tensor = self._preprocess_image(face_img)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})

        # Extract embedding
        embedding = outputs[0].flatten()

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding


# ============================================================================
# 메인 함수
# ============================================================================

def get_random_lfw_images(lfw_dir: str, num_samples: int = 10):
    """
    LFW 데이터셋에서 랜덤으로 이미지 샘플링

    Args:
        lfw_dir: LFW 데이터셋 디렉토리
        num_samples: 샘플링할 이미지 수

    Returns:
        List of image paths
    """
    # LFW 디렉토리에서 모든 이미지 찾기
    all_images = []

    if not os.path.exists(lfw_dir):
        print(f"경고: LFW 디렉토리를 찾을 수 없습니다: {lfw_dir}")
        print(f"임시 디렉토리를 생성하고 샘플 이미지를 준비해주세요.")
        return []

    for person_dir in Path(lfw_dir).iterdir():
        if person_dir.is_dir():
            for img_path in person_dir.glob("*.jpg"):
                all_images.append(str(img_path))

    if len(all_images) == 0:
        print(f"경고: LFW 디렉토리에서 이미지를 찾을 수 없습니다.")
        return []

    # 랜덤 샘플링
    num_samples = min(num_samples, len(all_images))
    sampled_images = random.sample(all_images, num_samples)

    return sampled_images


def main():
    """메인 실행 함수"""

    print("=" * 80)
    print("LFW EdgeFace ONNX vs NPU 임베딩 비교")
    print("=" * 80)

    # 1. 모델 초기화
    print("\n=== 모델 초기화 ===")

    # YuNet CPU 검출기
    print("\n[1/3] YuNet CPU 검출기 로딩...")
    yunet_cpu = YuNetDetector(YUNET_CPU_MODEL, device='cpu', crop_size=(112, 112))
    print("YuNet CPU 검출기 로딩 완료")

    # EdgeFace ONNX
    print("\n[2/3] EdgeFace ONNX 모델 로딩...")
    edgeface_onnx = EdgeFaceONNXRecognizer(EDGEFACE_ONNX_MODEL)
    print("EdgeFace ONNX 모델 로딩 완료")

    # EdgeFace NPU
    print("\n[3/3] EdgeFace NPU 모델 로딩...")
    edgeface_npu = EdgeFaceNPURecognizer(EDGEFACE_NPU_MODEL, model_name='edgeface_xs_gamma_06', device='npu')
    print("EdgeFace NPU 모델 로딩 완료")

    # 2. LFW 이미지 샘플링
    print(f"\n=== LFW 데이터셋에서 {NUM_SAMPLES}개 이미지 샘플링 ===")
    print(f"LFW 디렉토리: {LFW_DIR}")

    sampled_images = get_random_lfw_images(LFW_DIR, NUM_SAMPLES)

    if len(sampled_images) == 0:
        print("\n이미지를 찾을 수 없습니다.")
        print(f"LFW_DIR 경로를 확인하거나, 코드 상단의 LFW_DIR 변수를 수정해주세요.")
        print(f"현재 설정: LFW_DIR = '{LFW_DIR}'")
        return

    print(f"\n샘플링된 이미지: {len(sampled_images)}개")

    # 3. 각 이미지에 대해 처리
    print("\n=== 이미지 처리 및 임베딩 비교 ===\n")

    results = []

    for idx, img_path in enumerate(sampled_images):
        print(f"[{idx+1}/{len(sampled_images)}] 처리 중: {os.path.basename(img_path)}")

        try:
            # 이미지 로드
            pil_img = Image.open(img_path).convert('RGB')

            # YuNet CPU로 얼굴 디텍션
            aligned_face = yunet_cpu.align(pil_img)

            if aligned_face is None:
                print(f"  얼굴 검출 실패")
                continue

            # BGR로 변환
            face_np = cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR)

            # EdgeFace ONNX 임베딩 추출
            emb_onnx = edgeface_onnx.extract_embedding(face_np)

            # EdgeFace NPU 임베딩 추출
            emb_npu = edgeface_npu.extract_embedding(face_np)
            

            # 코사인 유사도 계산
            cosine_sim = np.dot(emb_onnx, emb_npu)

            # L2 거리 계산
            l2_dist = np.linalg.norm(emb_onnx - emb_npu)

            # 임베딩 통계
            onnx_min = np.min(emb_onnx)
            onnx_max = np.max(emb_onnx)
            onnx_mean = np.mean(emb_onnx)
            onnx_std = np.std(emb_onnx)

            npu_min = np.min(emb_npu)
            npu_max = np.max(emb_npu)
            npu_mean = np.mean(emb_npu)
            npu_std = np.std(emb_npu)

            # 결과 저장
            result = {
                'image': os.path.basename(img_path),
                'cosine_similarity': cosine_sim,
                'l2_distance': l2_dist,
                'onnx_embedding': emb_onnx,
                'npu_embedding': emb_npu,
                'onnx_min': onnx_min,
                'onnx_max': onnx_max,
                'onnx_mean': onnx_mean,
                'onnx_std': onnx_std,
                'npu_min': npu_min,
                'npu_max': npu_max,
                'npu_mean': npu_mean,
                'npu_std': npu_std
            }

            results.append(result)

            # 결과 출력
            print(f"  코사인 유사도: {cosine_sim:.6f}")
            print(f"  L2 거리: {l2_dist:.6f}")
            print(f"  ONNX 임베딩 - Min: {onnx_min:.6f}, Max: {onnx_max:.6f}, Mean: {onnx_mean:.6f}, Std: {onnx_std:.6f}")
            print(f"  NPU 임베딩  - Min: {npu_min:.6f}, Max: {npu_max:.6f}, Mean: {npu_mean:.6f}, Std: {npu_std:.6f}")
            print()

        except Exception as e:
            print(f"  오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. 전체 결과 요약
    if len(results) > 0:
        print("\n" + "=" * 80)
        print("전체 결과 요약")
        print("=" * 80)

        cosine_sims = [r['cosine_similarity'] for r in results]
        l2_dists = [r['l2_distance'] for r in results]

        onnx_mins = [r['onnx_min'] for r in results]
        onnx_maxs = [r['onnx_max'] for r in results]
        onnx_means = [r['onnx_mean'] for r in results]

        npu_mins = [r['npu_min'] for r in results]
        npu_maxs = [r['npu_max'] for r in results]
        npu_means = [r['npu_mean'] for r in results]

        print(f"\n처리된 이미지 수: {len(results)}")

        print(f"\n[코사인 유사도]")
        print(f"  평균: {np.mean(cosine_sims):.6f}")
        print(f"  표준편차: {np.std(cosine_sims):.6f}")
        print(f"  최소: {np.min(cosine_sims):.6f}")
        print(f"  최대: {np.max(cosine_sims):.6f}")

        print(f"\n[L2 거리]")
        print(f"  평균: {np.mean(l2_dists):.6f}")
        print(f"  표준편차: {np.std(l2_dists):.6f}")
        print(f"  최소: {np.min(l2_dists):.6f}")
        print(f"  최대: {np.max(l2_dists):.6f}")

        print(f"\n[ONNX 임베딩 통계]")
        print(f"  Min 값 범위: {np.min(onnx_mins):.6f} ~ {np.max(onnx_mins):.6f} (평균: {np.mean(onnx_mins):.6f})")
        print(f"  Max 값 범위: {np.min(onnx_maxs):.6f} ~ {np.max(onnx_maxs):.6f} (평균: {np.mean(onnx_maxs):.6f})")
        print(f"  Mean 값 범위: {np.min(onnx_means):.6f} ~ {np.max(onnx_means):.6f} (평균: {np.mean(onnx_means):.6f})")

        print(f"\n[NPU 임베딩 통계]")
        print(f"  Min 값 범위: {np.min(npu_mins):.6f} ~ {np.max(npu_mins):.6f} (평균: {np.mean(npu_mins):.6f})")
        print(f"  Max 값 범위: {np.min(npu_maxs):.6f} ~ {np.max(npu_maxs):.6f} (평균: {np.mean(npu_maxs):.6f})")
        print(f"  Mean 값 범위: {np.min(npu_means):.6f} ~ {np.max(npu_means):.6f} (평균: {np.mean(npu_means):.6f})")

        print(f"\n[임베딩 값 차이]")
        print(f"  Min 값 차이 평균: {np.mean(np.array(onnx_mins) - np.array(npu_mins)):.6f}")
        print(f"  Max 값 차이 평균: {np.mean(np.array(onnx_maxs) - np.array(npu_maxs)):.6f}")
        print(f"  Mean 값 차이 평균: {np.mean(np.array(onnx_means) - np.array(npu_means)):.6f}")

        print("\n" + "=" * 80)
        print("완료!")
        print("=" * 80)
    else:
        print("\n처리된 이미지가 없습니다.")


if __name__ == "__main__":
    main()
