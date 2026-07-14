#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Step 3] 캘리브레이션 없이 PyTorch & ONNX 모델 직접 추론 및 검증 (No Calibration Baseline)

이 스크립트는 INT8 캘리브레이션 및 NPU 컴파일 과정을 거치지 않고, 
부동소수점(FP32) 상태의 PyTorch 원본 모델(`.pt`) 및 ONNX 모델(`.onnx`)을 
직접 구동하여 얼굴 임베딩을 추출하고 검증하는 실무 코드입니다.

[사용 목적]
1. 양자화(Calibration) 전에 ONNX 변환(`.onnx`)이 100% 오차 없이 완벽히 이루어졌는지 확인 (Baseline 검증)
2. NPU 보드 없이 PC/서버(CPU/GPU) 환경에서 즉각적인 안면 인식 추론 가동
3. 두 얼굴 이미지 간의 유사도(동일인 여부)를 캘리브레이션 없이 초고정밀 FP32로 비교
"""

import os
import sys
import cv2
import numpy as np
import torch

# onnxruntime 임포트 시도
try:
    import onnxruntime as ort
    has_ort = True
except ImportError:
    has_ort = False
    print("⚠️ onnxruntime 패키지가 없습니다. ONNX 검증을 위해 'pip install onnxruntime'을 권장합니다.")

# 모델 로더 패스 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    from backbones import get_model
except ImportError:
    print("❌ EdgeFace backbones 모듈을 찾을 수 없습니다. 경로를 확인해 주세요.")
    sys.exit(1)


def _preprocess_fp32(rgb_img: np.ndarray) -> np.ndarray:
    """
    부동소수점(FP32) PyTorch 및 ONNX 모델용 공통 전처리
    - 입력: (112, 112, 3) RGB uint8 이미지
    - 출력: (1, 3, 112, 112) float32 정규화 텐서 (CHW)
    
    ⚠️ 주의: NPU(.dxnn)는 'uint8 NHWC' 원본을 받지만,
            캘리브레이션 없는 순수 FP32 ONNX/PyTorch 모델은 'float32 CHW 정규화'를 받습니다!
    """
    float_img = rgb_img.astype(np.float32) / 255.0
    normalized = (float_img - 0.5) / 0.5
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def get_pytorch_embedding_fp32(pt_model, rgb_img, device='cpu'):
    """PyTorch FP32 모델에서 512차원 임베딩 추출"""
    input_tensor = torch.tensor(_preprocess_fp32(rgb_img)).to(device)
    with torch.no_grad():
        emb = pt_model(input_tensor).cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)


def get_onnx_embedding_fp32(ort_session, rgb_img):
    """ONNX FP32 모델에서 512차원 임베딩 추출"""
    input_name = ort_session.get_inputs()[0].name
    input_data = _preprocess_fp32(rgb_img)
    outputs = ort_session.run(None, {input_name: input_data})
    emb = outputs[0].flatten()
    return emb / np.linalg.norm(emb)


def main():
    print("=" * 75)
    print(" 🚀 [Step 3] 캘리브레이션 없는 순수 PyTorch vs ONNX 추론 및 정밀도 검증")
    print("=" * 75)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📌 실행 환경: {device.upper()}")

    # 1. 파일 경로 설정
    pt_path = os.path.join(script_dir, "models", "edgeface_xs_gamma_06.pt")
    if not os.path.exists(pt_path):
        pt_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.pt")
        
    onnx_path = os.path.join(script_dir, "models", "edgeface_xs_gamma_06.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.onnx")

    img_path_1 = os.path.join(script_dir, "test_images", "aligned_sample_1.jpg")
    img_path_2 = os.path.join(script_dir, "test_images", "aligned_sample_2.jpg")
    
    # 2. PyTorch FP32 모델 로드
    print(f"\n1️⃣ PyTorch 모델 로드: {os.path.basename(pt_path)}...")
    pt_model = get_model("edgeface_xs_gamma_06")
    checkpoint = torch.load(pt_path, map_location=device)
    if 'state_dict' in checkpoint:
        pt_model.load_state_dict(checkpoint['state_dict'])
    else:
        pt_model.load_state_dict(checkpoint)
    pt_model.to(device)
    pt_model.eval()
    print("   ✅ PyTorch 모델 로드 완료!")

    # 3. ONNX FP32 모델 로드 (onnxruntime)
    ort_session = None
    if has_ort and os.path.exists(onnx_path):
        print(f"2️⃣ ONNX 모델 로드: {os.path.basename(onnx_path)}...")
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print("   ✅ ONNX Runtime 세션 로드 완료!")
    else:
        print(f"⚠️ ONNX 모델을 로드하지 못했습니다 (경로: {onnx_path}, onnxruntime: {has_ort})")

    # 4. 테스트 이미지 로드 및 전처리
    print("\n3️⃣ 테스트 이미지 로드 및 전처리...")
    if not os.path.exists(img_path_1):
        print(f"❌ 테스트 이미지 1이 없습니다: {img_path_1}")
        return
        
    img1 = cv2.imread(img_path_1)
    rgb_img1 = cv2.cvtColor(cv2.resize(img1, (112, 112)), cv2.COLOR_BGR2RGB)
    print(f"   - 이미지 1 로드: {os.path.basename(img_path_1)} (Shape: {rgb_img1.shape})")

    # [검증 1] PyTorch FP32 vs ONNX FP32 모델 간 무차실(0% 오차) 일치 검증
    print("\n" + "=" * 75)
    print(" [검증 1] PyTorch FP32 vs ONNX FP32 단일 얼굴 임베딩 변환 정밀도 비교")
    print("=" * 75)
    
    emb_pt = get_pytorch_embedding_fp32(pt_model, rgb_img1, device)
    print(f" 🔹 PyTorch 임베딩 (첫 5개 값): {emb_pt[:5].round(6)}")
    
    if ort_session:
        emb_onnx = get_onnx_embedding_fp32(ort_session, rgb_img1)
        print(f" 🔹 ONNX    임베딩 (첫 5개 값): {emb_onnx[:5].round(6)}")
        
        # 코사인 유사도 및 오차 계산
        cosine_sim = np.dot(emb_pt, emb_onnx) / (np.linalg.norm(emb_pt) * np.linalg.norm(emb_onnx))
        mae = np.mean(np.abs(emb_pt - emb_onnx))
        max_err = np.max(np.abs(emb_pt - emb_onnx))
        
        print("-" * 75)
        print(f" 🎯 코사인 유사도 (Cosine Similarity) : {cosine_sim:.6f} ({cosine_sim*100:.4f}%)")
        print(f" 🎯 평균 절대 오차 (MAE)               : {mae:.8f}")
        print(f" 🎯 최대 절대 오차 (Max Error)         : {max_err:.8f}")
        
        if cosine_sim > 0.9999:
            print(" 🟢 [판정: 완벽 일치 (100.0%)] 양자화(Calibration) 전 부동소수점 모델 변환 무차실 증명 완료!")
        else:
            print(" 🟡 [판정: 미세 오차 존재] ONNX 변환 옵션을 확인해 주세요.")

    # [검증 2] 캘리브레이션 없는 두 얼굴 간의 안면 인식 유사도(동일인 여부) 판정
    if os.path.exists(img_path_2):
        print("\n" + "=" * 75)
        print(" [검증 2] 캘리브레이션 없는 순수 FP32 모델 기반 두 얼굴 간 유사도 비교")
        print("=" * 75)
        img2 = cv2.imread(img_path_2)
        rgb_img2 = cv2.cvtColor(cv2.resize(img2, (112, 112)), cv2.COLOR_BGR2RGB)
        print(f"   - 이미지 2 로드: {os.path.basename(img_path_2)}")
        
        # PyTorch로 비교
        emb_pt_1 = get_pytorch_embedding_fp32(pt_model, rgb_img1, device)
        emb_pt_2 = get_pytorch_embedding_fp32(pt_model, rgb_img2, device)
        face_sim_pt = np.dot(emb_pt_1, emb_pt_2)
        
        print(f"\n 👤 이미지 1 ({os.path.basename(img_path_1)}) vs 이미지 2 ({os.path.basename(img_path_2)})")
        print(f"    ▶ PyTorch FP32 안면 유사도: {face_sim_pt:.4f}")
        
        if ort_session:
            emb_onnx_1 = get_onnx_embedding_fp32(ort_session, rgb_img1)
            emb_onnx_2 = get_onnx_embedding_fp32(ort_session, rgb_img2)
            face_sim_onnx = np.dot(emb_onnx_1, emb_onnx_2)
            print(f"    ▶ ONNX    FP32 안면 유사도: {face_sim_onnx:.4f}")
            
        print("\n 💡 [참고] EdgeFace 안면 인식 판정 기준:")
        print("    - 유사도 > 0.50 : 동일 인물 (Same Person)")
        print("    - 유사도 < 0.50 : 서로 다른 인물 (Different Person)")
        print("=" * 75)

if __name__ == "__main__":
    main()
