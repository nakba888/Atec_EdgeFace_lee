#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Step 3] 캘리브레이션 있는 버전 vs 없는 버전 DXNN 컴파일 마스터 스크립트

이 스크립트는 Step 1의 ONNX 모델(`.onnx`)과 Step 2의 JSON 설정 파일들을 결합하여,
1. 정식 안면 정렬 이미지로 보정된 '캘리브레이션 완료 DXNN' (`edgeface_xs_gamma_06.dxnn`)
2. 일반 비정렬 이미지로 대충/비보정된 '비캘리브레이션 DXNN' (`edgeface_xs_uncalibrated.dxnn`)
두 가지 버전의 NPU 모델을 자동으로 컴파일하거나 정확한 CLI 실행 명령어를 안내합니다.

과거 `convert_edgeface_s_to_dxnn.py` 의 컴파일 템플릿을 현 워크플로우에 맞게 100% 계승하였습니다.
"""

import os
import sys
import shutil
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
models_dir = os.path.join(script_dir, "models")
configs_dir = os.path.join(script_dir, "configs")
os.makedirs(models_dir, exist_ok=True)


def check_compiler_exists():
    """DeepX NPU 컴파일러(dx_compiler 또는 dxcom/dx-com) 존재 여부 체크"""
    # 1. PATH에서 먼저 검색
    for cmd in ["dxcom", "dx_compiler", "dx-com", "dx-compiler"]:
        path = shutil.which(cmd)
        if path:
            return cmd, path
            
    # 2. 라즈베리파이 dxdemo dx-all-suite 및 사용자 로컬/가상환경 경로 자동 탐색
    custom_paths = [
        "/home/jarvis/.virtualenvs/venv-dx-compiler/bin/dxcom",
        "/home/jarvis/.virtualenvs/venv-dx-compiler/bin/dx_compiler",
        "/home/dxdemo/dx-all-suite/dx-compiler/bin/dx_compiler",
        "/home/dxdemo/dx-all-suite/dx-compiler/bin/dx-com",
        "/home/dxdemo/dx-all-suite/bin/dx-com",
        "/home/dxdemo/dx-all-suite/dx-compiler/dx_compiler",
        "/home/dxdemo/dx-all-suite/dx-compiler/dx-com",
        "/home/dxdemo/.local/bin/dxcom",
        "/home/dxdemo/.local/bin/dx-com",
        "/home/dxdemo/.local/bin/dx_compiler",
        "/usr/local/bin/dxcom",
        "/usr/local/bin/dx-com",
        "/usr/local/bin/dx_compiler"
    ]
    for abs_path in custom_paths:
        if os.path.exists(abs_path) and os.access(abs_path, os.X_OK):
            cmd_name = "dx_compiler" if "dx_compiler" in abs_path else "dxcom"
            return cmd_name, abs_path

    # 3. /home/dxdemo/dx-all-suite 디렉토리 내부 재귀적 자동 탐색 (폴더가 존재할 경우)
    suite_dir = "/home/dxdemo/dx-all-suite"
    if os.path.exists(suite_dir):
        for root, dirs, files in os.walk(suite_dir):
            for file_name in files:
                if file_name in ["dxcom", "dx_compiler", "dx-com"]:
                    full_path = os.path.join(root, file_name)
                    if os.access(full_path, os.X_OK):
                        return file_name, full_path
            
    return None, None


def run_or_print_compile(compiler_cmd, compiler_path, onnx_path, config_path, dxnn_output, version_name):
    """컴파일러가 있으면 즉시 실행, 없으면 과거 convert_edgeface처럼 수동 명령어 출력"""
    print("\n" + "-" * 78)
    print(f" 🔹 [{version_name}] DXNN 컴파일 작업 정보")
    print("-" * 78)
    print(f"   - 입력 ONNX   : {os.path.relpath(onnx_path, project_dir)}")
    print(f"   - 설정 JSON   : {os.path.relpath(config_path, project_dir)}")
    print(f"   - 출력 DXNN   : {os.path.relpath(dxnn_output, project_dir)}")

    if not os.path.exists(onnx_path):
        print(f"   ❌ 오류: 입력 ONNX 모델이 없습니다 -> {onnx_path}")
        print(f"      먼저 `python3 npu_workflow_demo/step1_pytorch_to_onnx.py`를 실행하세요.")
        return False

    if not os.path.exists(config_path):
        print(f"   ❌ 오류: 설정 JSON 파일이 없습니다 -> {config_path}")
        print(f"      먼저 `step2_prepare_calibration.py` 또는 `step2_prepare_uncalibrated_config.py`를 실행하세요.")
        return False

    if compiler_cmd and compiler_path:
        print(f"\n 🚀 NPU 컴파일러({compiler_cmd} @ {compiler_path}) 감지됨! 자동 컴파일을 시작합니다...")
        if compiler_cmd == "dx_compiler":
            cmd_list = [
                compiler_path,
                "--model", onnx_path,
                "--config", config_path,
                "--output", dxnn_output,
                "--target", "npu",
                "--optimize"
            ]
        else: # dx-com
            cmd_list = [
                compiler_path,
                "-m", onnx_path,
                "-c", config_path,
                "-o", dxnn_output
            ]
        try:
            subprocess.run(cmd_list, check=True)
            print(f" ✅ [{version_name}] DXNN 컴파일 성공: {dxnn_output}")
            return True
        except subprocess.CalledProcessError as e:
            print(f" ❌ 컴파일 실패 (Exit Code {e.returncode})")
            return False
    else:
        # 컴파일러가 없는 런타임 환경(예: 라즈베리파이)일 경우 명령어 안내
        print("\n ⚠️ [수동 컴파일 안내] 현재 환경(라즈베리파이 보드 등)에는 DeepX 컴파일러가 없습니다.")
        print("    아래 명령어를 DeepX 컴파일러(`dx_compiler` 또는 `dx-com`)가 설치된 PC/도커에서 실행하세요:\n")
        print(" [명령어 옵션 1: dx_compiler CLI 사용 시]")
        print(f" dx_compiler \\\n    --model {onnx_path} \\\n    --config {config_path} \\\n    --output {dxnn_output} \\\n    --target npu \\\n    --optimize")
        print("\n [명령어 옵션 2: dx-com CLI 사용 시]")
        print(f" dx-com -m {onnx_path} -c {config_path} -o {dxnn_output}")
        return False


def main():
    print("=" * 78)
    print(" 🛠️ [Step 3] 캘리브레이션 버전 vs 비캘리브레이션 버전 DXNN 올인원 컴파일러")
    print("=" * 78)

    compiler_cmd, compiler_path = check_compiler_exists()
    if compiler_cmd and compiler_path:
        print(f" ✅ 시스템 컴파일러 감지: {compiler_cmd} ({compiler_path})")
    else:
        print(" ℹ️ DeepX NPU 컴파일러 미감지 (수동 명령어 출력 모드로 작동합니다)")

    onnx_path = os.path.join(models_dir, "edgeface_xs_gamma_06.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(project_dir, "checkpoints", "edgeface_xs_gamma_06.onnx")

    # 1. 정식 캘리브레이션 완료 모델 (Calibrated DXNN)
    calib_config = os.path.join(configs_dir, "calibration_config.json")
    if not os.path.exists(calib_config):
        historical_config = os.path.join(project_dir, "npu_calibration", "calibration_config_edgeface.json")
        if os.path.exists(historical_config):
            calib_config = historical_config
            print(f"\n ℹ️ 원본 캘리브레이션 JSON 설정 감지 및 연동: {calib_config}")
        else:
            print("\n ℹ️ `calibration_config.json`이 없어 자동 생성을 시도합니다...")
            try:
                import step2_prepare_calibration as calib_script
                calib_script.main()
            except Exception:
                pass
    calib_dxnn = os.path.join(models_dir, "edgeface_xs_gamma_06.dxnn")
    run_or_print_compile(
        compiler_cmd, compiler_path, onnx_path, calib_config, calib_dxnn, 
        version_name="🟢 정식 캘리브레이션 버전 (Calibrated)"
    )

    # 2. 비캘리브레이션/더미 보정 모델 (Uncalibrated DXNN)
    uncalib_config = os.path.join(configs_dir, "uncalib_config.json")
    uncalib_dxnn = os.path.join(models_dir, "edgeface_xs_uncalibrated.dxnn")
    
    # 만약 uncalib_config.json이 없으면 자동 생성 시도
    if not os.path.exists(uncalib_config):
        print("\n ℹ️ `uncalib_config.json`이 없어 자동 생성을 시도합니다...")
        try:
            import step2_prepare_uncalibrated_config as uncalib_script
            uncalib_script.main()
        except Exception:
            pass

    run_or_print_compile(
        compiler_cmd, compiler_path, onnx_path, uncalib_config, uncalib_dxnn, 
        version_name="🔴 비캘리브레이션 버전 (Uncalibrated)"
    )

    print("\n" + "=" * 78)
    print(" 🎯 컴파일 완료 후 다음 단계 안내:")
    print("    두 모델(`calibrated.dxnn` vs `uncalibrated.dxnn`) 생성 완료 후,")
    print("    라즈베리파이에서 다음 스크립트를 실행하여 정합도(98.96% vs 폭락)를 비교하세요:")
    print("    👉 python3 npu_workflow_demo/step6_compare_all_models.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
