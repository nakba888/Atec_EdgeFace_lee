import os
import sys
import torch

# Add face_alignment to path to load the backbone modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, 'face_alignment'))

from backbones import get_model

def convert_pytorch_to_onnx(pytorch_path, onnx_path, model_name="edgeface_xs_gamma_06"):
    if not os.path.exists(pytorch_path):
        print(f"❌ PyTorch 모델 파일을 찾을 수 없습니다: {pytorch_path}")
        return False
        
    print(f"1. PyTorch 모델 로드 중: {model_name}...")
    try:
        model = get_model(model_name, fp16=False)
        model.load_state_dict(torch.load(pytorch_path, map_location='cpu'))
        model.eval()
        print("✅ 모델 구조 및 가중치 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return False
        
    print("2. 더미 입력 데이터 생성 중 (Shape: 1 x 3 x 112 x 112)...")
    dummy_input = torch.randn(1, 3, 112, 112)
    
    print(f"3. ONNX 포맷 변환 및 익스포트 중: {onnx_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,        # 모델 가중치를 ONNX 파일 내부로 포함
            opset_version=11,          # DeepX NPU 호환 opset 버전
            do_constant_folding=True,  # 상수 폴딩 최적화
            input_names=['input.1'],   # 입력 노드 이름 정의
            output_names=['output'],   # 출력 노드 이름 정의
            dynamic_axes={             # 배치 크기 가변화 설정
                'input.1': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ ONNX 변환 성공! 생성 위치: {onnx_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        return False

if __name__ == "__main__":
    pt_model = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.pt")
    out_onnx = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.onnx")
    
    # checkpoints 폴더 생성 (없을 경우)
    os.makedirs(os.path.dirname(out_onnx), exist_ok=True)
    
    convert_pytorch_to_onnx(pt_model, out_onnx)
