import os
import sys
import cv2
import numpy as np

# Try to load the DeepX dx_engine SDK
try:
    from dx_engine import InferenceEngine
    print("✅ dx_engine SDK 로드 성공")
except ImportError:
    print("❌ dx_engine SDK를 임포트할 수 없습니다. DeepX NPU SDK 설치를 확인하세요.")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

def run_inference(model_path, image_path):
    if not os.path.exists(model_path):
        print(f"❌ DXNN 모델 파일을 찾을 수 없습니다: {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"❌ 테스트 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    print(f"\n1. NPU DXNN 모델 로딩: {model_path}...")
    try:
        ie = InferenceEngine(model_path)
        print("✅ 엔진 로드 완료")
        if hasattr(ie, 'get_input_size'):
            print(f"   예상 입력 차원 (HWC/CHW): {ie.get_input_size()}")
        else:
            print(f"   예상 입력 차원 (HWC/CHW): {ie.input_size()}")
            
        if hasattr(ie, 'get_output_tensors_info'):
            print(f"   출력 텐서 정보: {ie.get_output_tensors_info()}")
        else:
            print(f"   출력 정밀도 타입: {ie.output_dtype()}")
    except Exception as e:
        print(f"❌ 엔진 로드 실패: {e}")
        return

    # 2. 이미지 읽기 및 크기 리사이징 (112x112)
    print(f"\n2. 테스트 이미지 전처리: {image_path}...")
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (112, 112))
    
    # BGR -> RGB 변환 (기존 edgeface_npu_recognizer.py 호환 규격)
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 3. NHWC (HWC) 포맷 및 uint8 주입 (DeepX NPU 컴파일 모델의 기본 기대 형상)
    # NPU 컴파일 시 정규화(mean/std)가 내부 하드웨어 레이어로 삽입되었으며, shape은 (1, 112, 112, 3)을 기대합니다.
    input_tensor = np.expand_dims(rgb_img, axis=0).astype(np.uint8)
    
    # NPU 연산 가속 및 안전성을 위해 연속 메모리(C-contiguous) 보정
    input_tensor = np.ascontiguousarray(input_tensor)

    # 4. NPU 추론 실행
    print("\n3. 🚀 NPU 추론 구동 중...")
    try:
        ie.run(input_tensor)
        print("✅ 추론 연산 성공")
    except Exception as e:
        print(f"❌ 추론 구동 중 에러 발생: {e}")
        return

    # 5. 결과 텐서 수집 및 특징 벡터 후처리
    outputs = ie.get_all_task_outputs()
    print(f"   반환된 NPU 출력 텐서 개수: {len(outputs)}")

    # EdgeFace NPU의 경우 outputs[1]이 특징 임베딩 텐서(1, 512)입니다.
    if len(outputs) >= 2:
        embedding = outputs[1]
    else:
        embedding = outputs[0]

    if isinstance(embedding, list):
        embedding = embedding[0]

    # 1D 차원으로 평탄화
    embedding = embedding.flatten()
    
    # 정규화 유사도 측정을 위해 L2 Normalize 수행
    embedding = embedding / np.linalg.norm(embedding)
    
    print("\n🎉 특징 벡터(Embedding) 추출 완료!")
    print(f"   임베딩 벡터 차원: {embedding.size}-d")
    print(f"   특징 데이터 일부값 (첫 10개 요소):")
    print(f"   {embedding[:10]}")

if __name__ == "__main__":
    # 1. DXNN 모델 경로: npu_workflow_demo/models 내부를 우선 참조
    dxnn_path = os.path.join(script_dir, "models", "edgeface_xs_gamma_06.dxnn")
    if not os.path.exists(dxnn_path):
        dxnn_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.dxnn")
        
    # 2. 테스트 이미지 경로: npu_workflow_demo/test_images 내부를 우선 참조
    test_img = os.path.join(script_dir, "test_images", "lena.jpg")
    if not os.path.exists(test_img):
        test_img = os.path.join(project_dir, "test01", "lena.jpg")
    
    run_inference(dxnn_path, test_img)
