#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Step 2-Uncalib] 캘리브레이션 비교 실험용 '비보정/비정렬(Uncalibrated)' Config 생성기

이 스크립트는 정렬된 얼굴 데이터셋(`aligned_sample_*.jpg`) 대신, 
배경과 모자가 섞인 엉뚱한 일반 비정렬 사진(`lena.jpg`) 단 1장만을 
캘리브레이션 데이터셋으로 지정하는 설정 파일(`uncalib_config.json`)을 생성합니다.

[사용 목적]
이렇게 만든 설정 파일로 `dx-com`을 돌려서 `uncalibrated.dxnn`을 만든 뒤,
`step6_compare_all_models.py`를 가동하면 "캘리브레이션을 안 하거나 엉뚱한 사진으로 대충 했을 때
얼마나 텐서가 파손되고 안면 인식률이 폭락하는지" 극명하게 비교 검증할 수 있습니다!
"""

import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
config_dir = os.path.join(script_dir, "configs")
os.makedirs(config_dir, exist_ok=True)

def main():
    print("=" * 75)
    print(" 🛠️ [Step 2-Uncalib] 캘리브레이션 비교 실험용 비보정 Config JSON 생성")
    print("=" * 75)
    
    # 엉뚱한 일반 사진(lena.jpg) 1장만을 담은 디렉토리를 지정하여 가짜/부실 캘리브레이션 유도
    lena_dir = os.path.join(script_dir, "test_images")
    output_json = os.path.join(config_dir, "uncalib_config.json")
    
    # DeepX NPU 양자화 컴파일용 비보정 설정 구조
    config = {
        "model_type": "ONNX",
        "model_path": "models/edgeface_xs_gamma_06.onnx",
        "output_path": "models/edgeface_xs_uncalibrated.dxnn",
        "input_names": ["input.1"],
        "output_names": ["output"],
        "preprocessing": {
            "mean": [127.5, 127.5, 127.5],
            "std": [127.5, 127.5, 127.5],
            "swap_rb": True,
            "scale": 1.0
        },
        "calibration": {
            "calibration_dataset": lena_dir,  # ⚠️ 정렬 얼굴이 아닌 lena.jpg 폴더 지정!
            "calibration_num": 1,             # ⚠️ 단 1장만 대충 보정!
            "quantization_type": "INT8"
        }
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    print(f" ✅ 비캘리브레이션 설정 JSON 생성 완료: {output_json}")
    print("\n 👉 [다음 단계: uncalibrated.dxnn 컴파일 명령어]")
    print(f"    dx-com -m checkpoints/edgeface_xs_gamma_06.onnx -c npu_workflow_demo/configs/uncalib_config.json -o npu_workflow_demo/models/edgeface_xs_uncalibrated.dxnn")
    print("\n 👉 [그 후 3자 비교 검증 다시 실행]")
    print(f"    python3 npu_workflow_demo/step6_compare_all_models.py")
    print("=" * 75)

if __name__ == "__main__":
    main()
