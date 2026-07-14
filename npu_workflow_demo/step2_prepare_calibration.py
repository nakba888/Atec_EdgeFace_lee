import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

def generate_config():
    # NPU 컴파일 단계에서 사용할 양자화 및 입력 전처리 옵션 구성
    config = {
      "model_info": {
        "input_name": "input.1",
        "input_shape": [1, 3, 112, 112],
        "input_dtype": "float32"
      },
      "calibration_info": {
        "calibration_dataset_dir": "npu_workflow_demo/calibration_dataset",
        "calibration_num": 100,
        "calibration_method": "ema",
        "preprocessing": {
          "mean": [127.5, 127.5, 127.5],
          "std": [127.5, 127.5, 127.5],
          "swap_rb": True,  # BGR -> RGB 채널 스왑을 NPU 레이어 레벨에서 처리
          "scale": 1.0
        }
      }
    }
    
    # 설정 파일(JSON)을 npu_workflow_demo/configs 폴더 안에 저장
    output_path = os.path.join(script_dir, "configs", "calibration_config_edgeface_xs_gamma_06.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    print(f"✅ Calibration Config 생성 성공! 생성 위치: {output_path}")
    print("\n[안내] 호스트 PC 또는 컴파일 서버에서 아래 명령을 실행하여 .dxnn 파일을 컴파일할 수 있습니다:")
    print("=" * 85)
    print(f"dx_compiler \\")
    print(f"    --model npu_workflow_demo/models/edgeface_xs_gamma_06.onnx \\")
    print(f"    --config npu_workflow_demo/configs/calibration_config_edgeface_xs_gamma_06.json \\")
    print(f"    --output npu_workflow_demo/models/edgeface_xs_gamma_06.dxnn \\")
    print(f"    --target npu \\")
    print(f"    --optimize")
    print("=" * 85)

if __name__ == "__main__":
    generate_config()
