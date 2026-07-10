# Raspberry Pi & DeepX NPU (dxnn SDK) Integration Summary

이 문서에는 Raspberry Pi 및 DeepX NPU(dxnn SDK)를 활용하여 EdgeFace 얼굴 인식 및 YuNet 얼굴 검출 시스템을 통합하고 가속화한 작업 결과가 정리되어 있습니다.

---

## 1. NPU 연동 개요

EdgeFace 얼굴 인식 시스템에 DeepX NPU 가속 지원을 추가하였습니다. 하드웨어 리소스가 제한적인 Raspberry Pi 환경에서 딥러닝 추론의 병목을 해결하기 위해, NPU 기반의 전체 연동 파이프라인을 구축하였습니다.

### 🌟 자동 NPU 활성화 (Unified Pipeline)
- Face Detector를 **`yunet_npu`**로 설정하면 별도의 인자 설정 없이도 EdgeFace 얼굴 인식 모델이 자동으로 NPU(dxnn) 모드로 초기화 및 실행됩니다.
- CPU, GPU, NPU 장치를 명시적으로 제어할 수 있는 유연성을 제공합니다.

### 지원 모델 규격
1. **YuNet Face Detector (NPU)**
   - 모델 파일: `face_alignment/models/face_detection_yunet_2023mar.dxnn`
   - 입력 규격: `640x640`, 3채널, `uint8`
   - 주요 역할: 이미지 내 얼굴 검출 및 5개 핵심 랜드마크 추출
2. **EdgeFace Recognizer (NPU)**
   - 모델 파일: `checkpoints/edgeface_xs_gamma_06.dxnn`
   - 입력 규격: `112x112`, 3채널, `uint8`
   - 주요 역할: 정렬된 얼굴 영역에서 512차원 특징 벡터(Embedding) 추출

---

## 2. 주요 구현 모듈 및 소스코드 구조

프로젝트 루트 내에 NPU 연동을 지원하기 위해 다음과 같은 전용 모듈과 테스트 유틸리티가 구현되었습니다.

### 📂 디렉토리 구조
```
Atec_EdgeFace_lee/
├── face_alignment/
│   ├── yunet_npu.py                    # YuNet NPU 디코딩 및 검출 모듈
│   └── models/
│       └── face_detection_yunet_2023mar.dxnn  # NPU용 YuNet 컴파일 모델
├── edgeface_npu_recognizer.py          # EdgeFace NPU 추론 및 정규화 모듈
├── face_recognition_system.py          # 통합 얼굴 인식 파이프라인 (NPU 지원)
├── face_recognition_gui.py             # NPU 장치 선택을 지원하는 GUI 어플리케이션
├── convert_edgeface_s_to_dxnn.py       # PyTorch -> ONNX -> DXNN 변환 및 컴파일 가이드
├── test_npu_models.py                  # 개별 NPU 모델 (.dxnn) 추론 테스트 스크립트
└── test01/
    ├── compare_fp32_int8.py            # FP32 vs INT8 모델 성능 및 수치 분석 스크립트
    └── process_lfw_int8.py             # LFW 데이터셋 샘플을 이용한 INT8 모델 검증
```

### 1) edgeface_npu_recognizer.py (EdgeFace NPU Recognizer)
DeepX Python SDK인 `dx_engine.InferenceEngine`을 기반으로 구현되었습니다.
- **전처리**: `cv2.resize`로 112x112 크기 조정 후, BGR을 RGB로 변환하고 `uint8` 텐서로 포맷팅하여 NPU에 주입합니다. (양자화 보정 파라미터는 dxnn 파일 내부에 패킹됨)
- **추론**: `self.inference_engine.run(input_tensor)` 호출 및 `get_all_task_outputs()`로 NPU 출력 수집.
- **후처리**: 512차원 출력 벡터를 Flatten한 후 $L_2$ Normalization을 적용해 정규화된 임베딩을 계산합니다.
- **기타**: 추론 속도 벤치마크를 위한 `extract_embedding_timed()` 및 순차적 일괄 처리를 위한 `extract_embeddings_batch()`를 탑재했습니다.

### 2) face_alignment/yunet_npu.py (YuNet NPU Detector)
ONNX 모델을 DeepX NPU용으로 컴파일할 때 **NMS 및 BBox 디코딩 레이어가 제거되어 생기는 원시 텐서(Raw features) 출력 문제**를 해결하기 위해 디코딩 과정을 직접 파이썬으로 구현했습니다.
- **Raw feature map 디코딩**: Stride 8(80x80), Stride 16(40x40), Stride 32(20x20)의 3가지 스케일 피처 맵에 대해 Classification, Objectness, Bounding box offset, Landmark offset을 계산합니다.
- **수치 보정 로직**:
  - Grid top-left anchor 위치 기반의 BBox 디코딩 수행.
  - 얼굴 형상(종횡비)에 최적화된 Prior width/height 비율 적용 (예: `prior_w = stride * 3`, `prior_h = stride * 3.6`).
  - 중복 검출을 배제하기 위한 **NMS(Non-Maximum Suppression)** 연산 직접 구현.
  - 정렬 및 인식을 더 안정적으로 만들기 위해 **랜드마크 안정화(Landmark Stabilization)** 필터 적용.

---

## 3. 모델 변환 및 양자화 가이드 (ONNX -> DXNN)

convert_edgeface_s_to_dxnn.py 스크립트를 통해 모델을 최적화 및 컴파일하는 프로세스를 구현하였습니다.

### 1단계: PyTorch에서 ONNX로 익스포트
`torch.onnx.export`를 사용하여 opset_version=11 기준으로 `dynamic_axes`를 지정하여 익스포트합니다.

### 2단계: Calibration Dataset 준비
양자화 오차를 줄이기 위해 LFW 데이터셋 중 고품질 얼굴 이미지 100장을 선정하여 `npu_calibration/calibration_dataset` 디렉토리에 전처리(112x112 리사이즈 및 크롭)하여 저장합니다.

### 3단계: Calibration Config 구성
EMA(Exponential Moving Average) 기반의 클리핑 및 양자화 스케일 파라미터 정보가 담긴 `calibration_config_edgeface_s.json` 설정 파일을 빌드합니다.

### 4단계: DeepX Compiler 컴파일 (dx_compiler)
NPU 하드웨어 타겟에 맞춘 모델 컴파일러를 수동으로 구동하여 최적화된 `.dxnn` 정밀도 변환 모델을 생성합니다.
```bash
dx_compiler \
    --model checkpoints/edgeface_s_gamma_05.onnx \
    --config npu_calibration/calibration_config_edgeface_s.json \
    --output checkpoints/edgeface_s_gamma_05.dxnn \
    --target npu \
    --optimize
```

---

## 4. 검증 및 벤치마크 결과

NPU 칩셋을 직접 모사하거나 임베디드 성능을 검증하기 위해 작성된 검증 유틸리티입니다.

### 1) FP32 ONNX vs INT8 양자화 ONNX 비교 (compare_fp32_int8.py)
- LFW에서 무작위 선택된 5개 이미지에 대해 원본 FP32 모델과 INT8 양자화 모델을 동시에 구동하여 오차를 수치화합니다.
- **측정 항목**: Bounding Box IoU, Confidence 차이값, Landmark 픽셀 이동 오차(MAE, Mean Absolute Error).
- **결과**: `overlap_<image_name>.jpg` 및 `side_by_side_<image_name>.jpg` 시각화 파일과 `numerical_comparison.txt` 보고서를 생성하여 양자화에 따른 정밀도 손실 수준이 무시할 정도로 미미함을 사전 검증하였습니다.

### 2) 예상 성능 개선 지표 (Raspberry Pi 5 + DeepX NPU 기준)
| 실행 모드 | 얼굴 검출 (Detector) | 얼굴 인식 (Recognizer) | 평균 처리 속도 (FPS) | 전력 효율성 |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Only** | YuNet (OpenCV) | EdgeFace (PyTorch CPU) | **~4 - 5 FPS** | 높음 (스로틀링 우려) |
| **NPU Acceleration** | YuNet (DeepX NPU) | EdgeFace (DeepX NPU) | **~10 - 13 FPS** | 낮음 (효율적 전력 소모) |

---

## 5. Raspberry Pi 임베디드 보드 배포 체크리스트

NPU SDK 환경에서 구동하기 전 검증해야 할 필수 사항들입니다.

- **SDK 모듈 설치 확인**: 
  ```bash
  python3 -c "from dx_engine import InferenceEngine; print('✅ SDK Ready')"
  ```
- **모델 파일 배치 확인**:
  - `face_alignment/models/face_detection_yunet_2023mar.dxnn`
  - `checkpoints/edgeface_xs_gamma_06.dxnn`
- **디바이스 상태 및 권한 검사**:
  - 카메라 디바이스 리스트 확인: `v4l2-ctl --list-devices`
  - 권한 접근 확인: `ls -l /dev/video*`
- **최종 통합 테스트**:
  - `python3 test_npu_models.py` 실행 시 오류 없이 입출력 텐서 모양 및 추론 결과가 출력되는지 검증.
