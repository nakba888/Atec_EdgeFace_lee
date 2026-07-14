#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Step 6] PyTorch vs 캘리브레이션 DXNN vs 비캘리브레이션 DXNN 3자 정밀도 비교 검증

이 스크립트는 원본 PyTorch 모델(`.pt`), 정식 캘리브레이션을 거친 NPU 모델(`calibrated.dxnn`), 
그리고 캘리브레이션 없이(또는 더미/비정렬 이미지로 대충 컴파일한) NPU 모델(`uncalibrated.dxnn`) 
3개를 동시에 로드하여 1:1:1 정밀도를 비교하는 벤치마크 스크립트입니다.

[사용 목적]
"INT8 양자화 보정(Calibration)을 제대로 한 모델과 안 한 모델이 실제 NPU 추론 정밀도에서
얼마나 극심한 성능 차이를 보이는지" 실측 데이터로 명쾌하게 입증하기 위함입니다.
"""

import os
import sys
import cv2
import numpy as np
import torch

try:
    from dx_engine import InferenceEngine
    has_dx_engine = True
except ImportError:
    has_dx_engine = False

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    from backbones import get_model
except ImportError:
    print("❌ backbones 모듈을 찾을 수 없습니다.")
    sys.exit(1)


def get_pt_embedding(model, rgb_img, device):
    """PyTorch FP32 모델 임베딩 추출 (float32 CHW 정규화 입력)"""
    float_img = rgb_img.astype(np.float32) / 255.0
    normalized = (float_img - 0.5) / 0.5
    chw = np.transpose(normalized, (2, 0, 1))
    input_tensor = torch.tensor(chw).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(input_tensor).cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)


def get_dxnn_embedding(dxnn_path, rgb_img):
    """DeepX NPU DXNN 모델 임베딩 추출 (uint8 NHWC 원본 입력)"""
    if not has_dx_engine or not os.path.exists(dxnn_path):
        return None
    ie = InferenceEngine(dxnn_path)
    input_tensor = np.expand_dims(rgb_img, axis=0).astype(np.uint8)
    input_tensor = np.ascontiguousarray(input_tensor)
    ie.run(input_tensor)
    outputs = ie.get_all_task_outputs()
    
    emb = outputs[1] if len(outputs) >= 2 else outputs[0]
    if isinstance(emb, list):
        emb = emb[0]
    emb = emb.flatten()
    return emb / np.linalg.norm(emb)


def main():
    print("=" * 78)
    print(" 🚀 [Step 6] PyTorch(FP32) vs 캘리브레이션 DXNN vs 비캘리브레이션 DXNN 3자 대조")
    print("=" * 78)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 모델 경로 설정
    pt_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.pt")
    
    # 정식 캘리브레이션 완료 DXNN
    calib_dxnn_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.dxnn")
    
    # 캘리브레이션 미보정/더미 DXNN (사용자가 uncalibrated로 생성한 파일 경로 우선 참조)
    uncalib_dxnn_path = os.path.join(script_dir, "models", "edgeface_xs_uncalibrated.dxnn")
    if not os.path.exists(uncalib_dxnn_path):
        uncalib_dxnn_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_uncalibrated.dxnn")

    # 2. PyTorch 로드
    print(f"\n1️⃣ PyTorch FP32 원본 모델 로드: {os.path.basename(pt_path)}...")
    if not os.path.exists(pt_path):
        print(f"❌ PyTorch 모델이 없습니다: {pt_path}")
        return
    pt_model = get_model("edgeface_xs_gamma_06")
    checkpoint = torch.load(pt_path, map_location=device)
    pt_model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    pt_model.to(device)
    pt_model.eval()

    # 3. 테스트 이미지 로드
    img_path = os.path.join(script_dir, "test_images", "aligned_sample_1.jpg")
    if not os.path.exists(img_path):
        print(f"❌ 테스트 이미지가 없습니다: {img_path}")
        return
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(cv2.resize(img, (112, 112)), cv2.COLOR_BGR2RGB)
    print(f"2️⃣ 테스트 정렬 얼굴 이미지: {os.path.basename(img_path)}")

    # 4. 임베딩 추출
    print("\n3️⃣ 모델별 임베딩 벡터 추출 중...")
    emb_pt = get_pt_embedding(pt_model, rgb_img, device)
    
    emb_calib = get_dxnn_embedding(calib_dxnn_path, rgb_img)
    emb_uncalib = get_dxnn_embedding(uncalib_dxnn_path, rgb_img)

    # 5. 비교 테이블 출력
    print("\n" + "=" * 78)
    print(" 📊 [최종 벤치마크] 캘리브레이션 유무에 따른 NPU 모델 정합도 비교 결과")
    print("=" * 78)
    print(f" 🔹 PyTorch 원본 (FP32)                  : [기준선 Baseline]")
    
    if emb_calib is not None:
        sim_calib = np.dot(emb_pt, emb_calib)
        print(f" 🟢 캘리브레이션 완료 DXNN (`calibrated`)  : 코사인 유사도 {sim_calib:.6f} ({sim_calib*100:.2f}%)")
    else:
        print(f" ⚪ 캘리브레이션 완료 DXNN                 : [모델 없음 또는 NPU 환경 아님 - {os.path.basename(calib_dxnn_path)}]")
        
    if emb_uncalib is not None:
        sim_uncalib = np.dot(emb_pt, emb_uncalib)
        print(f" 🔴 비캘리브레이션 DXNN (`uncalibrated`) : 코사인 유사도 {sim_uncalib:.6f} ({sim_uncalib*100:.2f}%)")
    else:
        print(f" ⚪ 비캘리브레이션 DXNN                 : [모델 파일 없음 - {os.path.basename(uncalib_dxnn_path)}]")
        print("\n 👉 [비캘리브레이션 DXNN 생성 안내]")
        print("    캘리브레이션 이미지 없이 더미/비보정 DXNN 모델(`uncalibrated.dxnn`)을 만들어 비교하려면:")
        print("    1. config JSON에서 calibration_dataset 경로를 일반 비정렬 사진(lena.jpg) 단 1장으로 설정")
        print("    2. dx-com -m models/edgeface_xs_gamma_06.onnx -c uncalib_config.json -o models/edgeface_xs_uncalibrated.dxnn")
        print("    3. 본 스크립트를 실행하여 98.78% vs 왜곡된 정밀도 차이를 눈으로 직접 비교해 보세요!")
    print("=" * 78)

if __name__ == "__main__":
    main()
