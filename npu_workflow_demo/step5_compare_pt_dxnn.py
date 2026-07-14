import os
import sys
import cv2
import torch
import numpy as np

# Add project root to path to load backbone modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

try:
    from backbones import get_model
except ImportError:
    print("❌ 'backbones' 모듈을 불러올 수 없습니다. 경로를 확인하세요.")
    sys.exit(1)

try:
    from dx_engine import InferenceEngine
    has_dx_engine = True
except ImportError:
    has_dx_engine = False
    print("⚠️ [안내] dx_engine SDK를 찾을 수 없습니다. (라즈베리파이 NPU 환경에서 실행해야 NPU 비교가 가능합니다)")

def get_pytorch_embedding(pt_path, rgb_img):
    """PyTorch 모델(FP32)을 사용하여 얼굴 임베딩 벡터 추출"""
    print(f"1. PyTorch 원본 모델(FP32) 로드: {pt_path}...")
    model = get_model("edgeface_xs_gamma_06", fp16=False)
    model.load_state_dict(torch.load(pt_path, map_location='cpu'))
    model.eval()
    
    # PyTorch 전처리: (x - 127.5) / 127.5 (즉, [-1.0, 1.0] 범위로 정규화)
    img_float = rgb_img.astype(np.float32)
    normalized = (img_float - 127.5) / 127.5
    
    # HWC -> CHW 및 Tensor 변환 (1, 3, 112, 112)
    chw = np.transpose(normalized, (2, 0, 1))
    tensor_input = torch.tensor(chw, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor_input)
        
    embedding = output.cpu().numpy().flatten()
    embedding = embedding / np.linalg.norm(embedding)  # L2 Normalize
    return embedding

def get_npu_embedding(dxnn_path, rgb_img):
    """DeepX NPU 모델(DXNN)을 사용하여 얼굴 임베딩 벡터 추출"""
    if not has_dx_engine:
        return None
        
    print(f"2. DeepX NPU 모델(DXNN) 로드: {dxnn_path}...")
    ie = InferenceEngine(dxnn_path)
    
    # 기존 checkpoints/edgeface_xs_gamma_06.dxnn 컴파일 설정과 완벽 일치하는 float32 정규화 입력 방식
    float_img = rgb_img.astype(np.float32) / 255.0
    normalized = (float_img - 0.5) / 0.5
    chw = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)
    input_tensor = np.ascontiguousarray(input_tensor)
    
    ie.run(input_tensor)
    outputs = ie.get_all_task_outputs()
    
    # EdgeFace NPU의 경우 outputs[1]이 512차원 특징 임베딩 텐서입니다.
    if len(outputs) >= 2:
        embedding = outputs[1]
    else:
        embedding = outputs[0]
        
    if isinstance(embedding, list):
        embedding = embedding[0]
        
    embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)  # L2 Normalize
    return embedding

def compare_embeddings(pt_emb, npu_emb):
    """두 임베딩 벡터 간의 유사도(Cosine Similarity) 및 오차 비교"""
    print("\n" + "=" * 75)
    print(" 📊 PyTorch (FP32) vs DeepX NPU (INT8) 임베딩 벡터 비교 결과")
    print("=" * 75)
    
    # 1. 코사인 유사도 (Cosine Similarity) 계산
    # 두 벡터가 모두 L2 정규화되어 있으므로 내적(Dot product)이 곧 코사인 유사도입니다.
    cosine_sim = np.dot(pt_emb, npu_emb)
    
    # 2. 평균 절대 오차 (MAE: Mean Absolute Error) 및 최대 오차 (Max Error)
    mae = np.mean(np.abs(pt_emb - npu_emb))
    max_err = np.max(np.abs(pt_emb - npu_emb))
    
    print(f"🔹 1. 첫 10개 임베딩 값 비교:")
    print(f"   [PyTorch FP32] : {pt_emb[:10].round(6)}")
    print(f"   [DeepX   NPU] : {npu_emb[:10].round(6)}")
    print("-" * 75)
    print(f"🔹 2. 코사인 유사도 (Cosine Similarity) : {cosine_sim:.6f}  ({cosine_sim * 100:.2f}%)")
    print(f"🔹 3. 평균 절대 오차 (MAE)                 : {mae:.6f}")
    print(f"🔹 4. 최대 절대 오차 (Max Error)           : {max_err:.6f}")
    print("=" * 75)
    
    # 3. 평가 판정 안내
    if cosine_sim >= 0.98:
        print("🟢 [판정: 최우수 (Excellent)]")
        print("   -> 양자화(INT8) 후에도 원본 모델(FP32)과 거의 완벽하게 동일한 특성을 유지하고 있습니다.")
        print("   -> 안면 인식 정합도 손실 없이 즉시 현장 배포가 가능한 훌륭한 수준입니다!")
    elif cosine_sim >= 0.95:
        print("🔵 [판정: 우수 (Good)]")
        print("   -> 양자화에 따른 미세한 오차가 있지만 안면 인식 실무에 충분히 활용 가능한 수준입니다.")
    else:
        print("🟡 [판정: 보정 권장 (Needs Tuning)]")
        print("   -> 양자화 손실이 다소 존재합니다. step2의 Calibration 데이터셋 이미지 수량을 늘리거나")
        print("      실제 현장 카메라 환경과 유사한 이미지로 재보정하는 것을 권장합니다.")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    # 파일 경로 설정 (npu_workflow_demo 하위 경로 우선 참조)
    pt_path = os.path.join(script_dir, "models", "edgeface_xs_gamma_06.pt")
    if not os.path.exists(pt_path):
        pt_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.pt")
        
    dxnn_path = os.path.join(script_dir, "models", "edgeface_xs_gamma_06.dxnn")
    if not os.path.exists(dxnn_path):
        dxnn_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.dxnn")
        
    img_path = os.path.join(script_dir, "test_images", "lena.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(project_dir, "test01", "lena.jpg")
        
    if not os.path.exists(pt_path):
        print(f"❌ PyTorch 모델을 찾을 수 없습니다: {pt_path}")
        sys.exit(1)
    if not os.path.exists(dxnn_path) and has_dx_engine:
        print(f"❌ DXNN 모델을 찾을 수 없습니다: {dxnn_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"❌ 테스트 이미지를 찾을 수 없습니다: {img_path}")
        sys.exit(1)
        
    print(f"\n🚀 [Step 5] 원본 PyTorch 모델 vs DeepX NPU 모델 정밀도 비교 검증 시작")
    print(f"   테스트 이미지: {img_path}\n")
    
    # 공통 이미지 읽기 및 전처리 (112x112 리사이즈 + RGB 변환)
    img = cv2.imread(img_path)
    resized = cv2.resize(img, (112, 112))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 1. PyTorch 임베딩 추출 (PyTorch는 float32 정규화 RGB 이미지 입력)
    pt_emb = get_pytorch_embedding(pt_path, rgb_img)
    
    # 2. NPU 임베딩 추출 및 비교 (기존 dxnn 호환 float32 정규화 RGB 이미지 입력)
    if has_dx_engine:
        npu_emb = get_npu_embedding(dxnn_path, rgb_img)
        if npu_emb is not None:
            compare_embeddings(pt_emb, npu_emb)
    else:
        print("\n⚠️ NPU 환경이 아니어서 PyTorch 추출 결과만 확인했습니다.")
        print(f"   [PyTorch FP32 첫 10개 값]: {pt_emb[:10].round(6)}")
        print("   👉 NPU 비교를 완료하려면 라즈베리파이에서 이 스크립트를 실행해 주세요!\n")
