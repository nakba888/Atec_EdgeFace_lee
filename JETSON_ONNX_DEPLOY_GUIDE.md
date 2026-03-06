# Jetson Orin Nano: ONNX 변환부터 배포까지 완전 가이드

> EdgeFace 얼굴 인식 모델을 Jetson Orin Nano에서 운용하기 위한 **End-to-End 가이드**입니다.  
> PyTorch 모델 → ONNX 변환 → TensorRT 엔진 생성 → 실시간 얼굴 인식 시스템 실행까지의 전체 흐름을 설명합니다.

---

## 목차

1. [전체 파이프라인 개요](#1-전체-파이프라인-개요)
2. [환경 준비](#2-환경-준비)
3. [Step 1: PyTorch → ONNX 변환](#3-step-1-pytorch--onnx-변환)
4. [Step 2: ONNX 모델 검증](#4-step-2-onnx-모델-검증)
5. [Step 3: ONNX → TensorRT 엔진 변환](#5-step-3-onnx--tensorrt-엔진-변환)
6. [Step 4: TensorRT 엔진 검증](#6-step-4-tensorrt-엔진-검증)
7. [Step 5: 얼굴 인식 시스템 실행](#7-step-5-얼굴-인식-시스템-실행)
8. [Step 6: 성능 벤치마크](#8-step-6-성능-벤치마크)
9. [모델 종류 및 선택 가이드](#9-모델-종류-및-선택-가이드)
10. [전체 코드 구조](#10-전체-코드-구조)
11. [자주 발생하는 문제 및 해결](#11-자주-발생하는-문제-및-해결)
12. [참고 자료](#12-참고-자료)

---

## 1. 전체 파이프라인 개요

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  PyTorch     │     │    ONNX      │     │  TensorRT    │     │  실시간 얼굴     │
│  모델 (.pt)  │ ──▶ │  모델 (.onnx)│ ──▶ │  엔진 (.trt) │ ──▶ │  인식 시스템     │
│              │     │              │     │              │     │                  │
│ torch2onnx.py│     │  검증/시각화 │     │onnx_to_      │     │face_recognition_ │
│              │     │              │     │tensorrt.py   │     │jetson_system.py  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────────┘
     (PC)               (PC/Jetson)           (Jetson)               (Jetson)
```

| 단계 | 스크립트 | 실행 환경 | 설명 |
|------|----------|-----------|------|
| 1 | `torch2onnx.py` | PC (GPU) | PyTorch → ONNX 변환 |
| 2 | `onnxruntime` | PC / Jetson | ONNX 모델 검증 |
| 3 | `onnx_to_tensorrt.py` | **Jetson** | ONNX → TensorRT 엔진 |
| 4 | `onnx_to_tensorrt.py --verify` | **Jetson** | 엔진 무결성 검증 |
| 5 | `face_recognition_jetson_system.py` | **Jetson** | 실시간 얼굴 인식 |
| 6 | `jetson_pytorch_comparison.py` | **Jetson** | 성능 비교 벤치마크 |

> ⚠️ **TensorRT 엔진 (.trt)은 생성한 디바이스에서만 동작합니다.** PC에서 생성한 엔진은 Jetson에서 사용할 수 없습니다.

---

## 2. 환경 준비

### 2.1 하드웨어

- **Jetson Orin Nano** (8GB 권장)
- USB 카메라 또는 CSI 카메라
- 안정적인 전원 공급 (5V 4A / PD 규격)

### 2.2 Jetson 소프트웨어 (JetPack)

```bash
# JetPack 버전 확인
cat /etc/nv_tegra_release

# TensorRT 버전 확인
python3 -c "import tensorrt; print(tensorrt.__version__)"

# CUDA 버전 확인
nvcc --version
```

- **JetPack 5.x** 이상 필요 (TensorRT 8.5+ 포함)
- Python 3.8+

### 2.3 Python 패키지 설치

```bash
# 가상환경 생성 (권장)
python3 -m venv ~/edgeface_env
source ~/edgeface_env/bin/activate

# 필수 패키지 설치
pip install numpy opencv-python pillow timm onnx onnxruntime

# PyTorch (Jetson용 - NVIDIA에서 제공하는 whl 사용)
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# 예시 (JetPack 5.x, Python 3.8):
# pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

> 💡 **Jetson에서 PyTorch를 pip으로 직접 설치하면 안 됩니다.** NVIDIA 공식 포럼에서 JetPack 버전에 맞는 whl 파일을 다운로드하여 설치하세요.

### 2.4 전력 모드 설정 (성능 최적화)

```bash
# 현재 전력 모드 확인
sudo nvpmodel -q

# 최대 성능 모드 (MAXN) - 변환/추론 시 권장
sudo nvpmodel -m 0
sudo jetson_clocks

# 15W 절전 모드
sudo nvpmodel -m 1

# 사용 가능한 모드 목록
sudo nvpmodel -p
```

### 2.5 디스크 용량 확인

```bash
# 설치 전 반드시 여유 공간 확인 (최소 3~5GB 이상 권장)
df -h
```

---

## 3. Step 1: PyTorch → ONNX 변환

### 3.1 사용 가능한 모델

| 모델 이름 | 파일 | 크기 | 특징 |
|-----------|------|------|------|
| `edgeface_xs_gamma_06` | `edgeface_xs_gamma_06.pt` | ~7MB | **권장** - 경량, Jetson 최적화 |
| `edgeface_s_gamma_05` | `edgeface_s_gamma_05.pt` | ~14MB | 중간 크기, 더 높은 정확도 |
| `edgeface_xxs` | `edgeface_xxs.pt` | ~5MB | 초경량 |
| `edgeface_base` | `edgeface_base.pt` | ~73MB | 최고 정확도, 무거움 |

### 3.2 변환 명령어

```bash
# 기본 변환 (edgeface_xs_gamma_06 - 권장)
python torch2onnx.py checkpoints/edgeface_xs_gamma_06.pt \
    --network edgeface_xs_gamma_06 \
    --output checkpoints/edgeface_xs_gamma_06.onnx

# edgeface_s_gamma_05 변환
python torch2onnx.py checkpoints/edgeface_s_gamma_05.pt \
    --network edgeface_s_gamma_05 \
    --output checkpoints/edgeface_s_gamma_05.onnx

# ONNX 단순화 적용 (선택)
python torch2onnx.py checkpoints/edgeface_xs_gamma_06.pt \
    --network edgeface_xs_gamma_06 \
    --output checkpoints/edgeface_xs_gamma_06.onnx \
    --simplify True
```

### 3.3 변환 코드 핵심 설명

`torch2onnx.py`의 핵심 동작:

```python
# 1. 더미 입력 생성 (112x112x3 이미지)
img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
img = img.astype(np.float)
img = (img / 255. - 0.5) / 0.5   # [-1, 1] 정규화 (torch style)
img = img.transpose((2, 0, 1))    # HWC → CHW
img = torch.from_numpy(img).unsqueeze(0).float()  # (1, 3, 112, 112)

# 2. 모델 로드 및 변환
net.load_state_dict(weight, strict=True)
net.eval()
torch.onnx.export(net, img, output,
                  input_names=["data"],
                  opset_version=11)

# 3. 배치 크기를 동적으로 설정
graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
```

**입력 사양**:
- 크기: `112 x 112` (BGR or RGB)
- 정규화: `(pixel / 255.0 - 0.5) / 0.5` → [-1, 1] 범위
- 형식: `(Batch, 3, 112, 112)` - NCHW

**출력**: 512차원 임베딩 벡터

---

## 4. Step 2: ONNX 모델 검증

### 4.1 기본 검증

```python
import onnx

# ONNX 모델 로드 및 검증
model = onnx.load("checkpoints/edgeface_xs_gamma_06.onnx")
onnx.checker.check_model(model)
print("✅ ONNX 모델 검증 통과!")

# 모델 정보 출력
print(f"입력: {[i.name for i in model.graph.input]}")
print(f"출력: {[o.name for o in model.graph.output]}")
```

### 4.2 ONNXRuntime으로 추론 테스트

```python
import onnxruntime as ort
import numpy as np
import cv2

# 세션 생성
session = ort.InferenceSession("checkpoints/edgeface_xs_gamma_06.onnx")

# 테스트 이미지 준비
img = cv2.imread("test_face.jpg")
img = cv2.resize(img, (112, 112))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = (img - 0.5) / 0.5
img = np.transpose(img, (2, 0, 1))           # HWC → CHW
img = np.expand_dims(img, axis=0)              # (1, 3, 112, 112)

# 추론 실행
result = session.run(None, {"data": img})
embedding = result[0]
print(f"임베딩 shape: {embedding.shape}")       # (1, 512)
print(f"임베딩 norm: {np.linalg.norm(embedding):.4f}")
```

---

## 5. Step 3: ONNX → TensorRT 엔진 변환

> ⚠️ **이 단계는 반드시 Jetson Orin Nano에서 실행해야 합니다.**

### 5.1 기본 변환 (FP16 - 권장)

```bash
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06.trt \
    --fp16
```

### 5.2 정밀도 옵션

```bash
# FP16 모드 (권장 - 최고 속도, 정확도 거의 동일)
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06_fp16.trt \
    --fp16

# FP32 모드 (최대 정확도, 느림)
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06_fp32.trt \
    --no-fp16

# INT8 모드 (최고 속도, 캘리브레이션 데이터 필요)
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06_int8.trt \
    --int8 \
    --calibration-dir ./calibration_images
```

### 5.3 고급 옵션

```bash
# 워크스페이스 크기 늘리기 (메모리 부족 시)
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06.trt \
    --fp16 \
    --workspace 2048

# 상세 로그 출력
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06.trt \
    --fp16 \
    --verbose

# 동적 배치 크기 지원
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06.trt \
    --fp16 \
    --dynamic \
    --max-batch-size 4
```

### 5.4 정밀도 모드 비교

| 모드 | 속도 | 정확도 | 메모리 | 비고 |
|------|------|--------|--------|------|
| **FP16** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **권장** - Jetson에 최적 |
| FP32 | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 정확도 기준 테스트용 |
| INT8 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 캘리브레이션 필요 |

> 💡 FP16은 Jetson Orin Nano의 GPU가 하드웨어 레벨에서 지원하므로 속도 손실 없이 메모리를 절약할 수 있습니다.

---

## 6. Step 4: TensorRT 엔진 검증

```bash
# 엔진 로드 및 I/O 텐서 정보 확인
python onnx_to_tensorrt.py --verify checkpoints/edgeface_xs_gamma_06.trt
```

**기대 출력 예시:**
```
Engine loaded successfully!
  Num bindings: 2
  Binding 0: data, shape: (1, 3, 112, 112), dtype: DataType.FLOAT, mode: INPUT
  Binding 1: output, shape: (1, 512), dtype: DataType.FLOAT, mode: OUTPUT
✅ Engine verification passed!
```

---

## 7. Step 5: 얼굴 인식 시스템 실행

### 7.1 커맨드라인 시스템 (CLI)

```bash
# TensorRT 모드로 실행 (기본값)
python face_recognition_jetson_system.py --camera 0

# PyTorch 모드로 실행 (비교용)
python face_recognition_jetson_system.py --camera 0 --no-tensorrt

# 특정 모델 경로 지정
python face_recognition_jetson_system.py \
    --camera 0 \
    --model checkpoints/edgeface_xs_gamma_06.trt \
    --detector yunet
```

### 7.2 GUI 시스템

```bash
# Jetson 전용 GUI 실행
python face_recognition_jetson_gui.py
```

**GUI 기능:**

| 기능 | 설명 |
|------|------|
| 🚀 TensorRT 체크박스 | TensorRT / PyTorch 모드 전환 |
| ⚡ FP16 체크박스 | 반정밀도 연산 활성화 |
| 📊 Run Benchmark | 현재 설정으로 성능 측정 |
| 실시간 성능 표시 | FPS, Detection/Recognition 시간, GPU 온도 |

**키보드 단축키:**

| 키 | 동작 |
|-----|------|
| `q` | 종료 |
| `t` | TensorRT / PyTorch 전환 |
| `b` | 벤치마크 실행 |
| `c` | 얼굴 캡처 (레퍼런스 등록) |

### 7.3 Python 코드에서 직접 사용

```python
from edgeface_jetson_recognizer import EdgeFaceJetsonRecognizer
import cv2
import numpy as np

# 1. 인식기 초기화
recognizer = EdgeFaceJetsonRecognizer(
    model_path="checkpoints/edgeface_xs_gamma_06.trt",
    model_name="edgeface_xs_gamma_06",
    device="jetson"
)

# 2. 얼굴 이미지에서 임베딩 추출
face_img = cv2.imread("aligned_face.jpg")
face_img = cv2.resize(face_img, (112, 112))
embedding = recognizer.extract_embedding(face_img)
print(f"임베딩 shape: {embedding.shape}")    # (512,)

# 3. 두 얼굴 비교
face_img2 = cv2.imread("aligned_face2.jpg")
face_img2 = cv2.resize(face_img2, (112, 112))
embedding2 = recognizer.extract_embedding(face_img2)

# 코사인 유사도 계산
similarity = EdgeFaceJetsonRecognizer.cosine_similarity(embedding, embedding2)
print(f"유사도: {similarity:.4f}")

if similarity > 0.5:
    print("✅ 동일 인물")
else:
    print("❌ 다른 인물")

# 4. 추론 시간 확인
inference_time = recognizer.get_inference_time()
print(f"추론 시간: {inference_time:.2f}ms")
```

### 7.4 전체 시스템 (감지 + 인식) 코드

```python
from face_recognition_jetson_system import FaceRecognitionJetsonSystem

# 시스템 초기화
system = FaceRecognitionJetsonSystem(
    detector_method='yunet',                                    # 얼굴 감지 방법
    edgeface_model_path='checkpoints/edgeface_xs_gamma_06.trt', # TensorRT 엔진
    edgeface_model_name='edgeface_xs_gamma_06',
    device='jetson',
    similarity_threshold=0.5,
    use_tensorrt=True,
    fp16=True
)

# 레퍼런스 얼굴 등록
system.add_reference_from_image("reference_face.jpg", "person_name")

# 카메라로 실시간 인식 실행
system.run_camera(camera_id=0)
```

---

## 8. Step 6: 성능 벤치마크

### 8.1 빠른 벤치마크

```bash
python jetson_pytorch_comparison.py
```

### 8.2 상세 벤치마크

```bash
python jetson_pytorch_comparison.py \
    --iterations 200 \
    --accuracy-samples 100 \
    --output benchmark_results
```

### 8.3 예상 성능 (Jetson Orin Nano, MAXN 모드)

| 항목 | PyTorch (FP32) | TensorRT (FP16) | 향상 비율 |
|------|----------------|-----------------|-----------|
| 추론 시간 | ~25ms | ~8ms | **~3x** |
| FPS | ~40 | ~120 | **~3x** |
| GPU 메모리 | ~1.5GB | ~0.8GB | **~2x** |

> 실제 성능은 JetPack 버전, 전력 모드, 시스템 부하에 따라 달라질 수 있습니다.

### 8.4 벤치마크 결과 파일

`benchmark_results/` 폴더에 생성됩니다:

| 파일 | 설명 |
|------|------|
| `benchmark_results.json` | 수치 결과 (JSON) |
| `latency_comparison.png` | 추론 속도 비교 차트 |
| `accuracy_comparison.png` | 정확도 비교 차트 |

---

## 9. 모델 종류 및 선택 가이드

### 9.1 사용 가능한 모델 전체 목록

| 모델 | 파라미터 | 정확도 | 속도 | Jetson 권장 |
|------|----------|--------|------|-------------|
| `edgeface_xxs` | 최소 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ 초저전력 |
| `edgeface_xs_gamma_06` | 소 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ **최적 균형** |
| `edgeface_s_gamma_05` | 중 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ 고정확도 필요 시 |
| `edgeface_base` | 대 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⚠️ 메모리 주의 |

### 9.2 선택 기준

- **실시간 처리 (≥30FPS 필요)**: `edgeface_xs_gamma_06` + TensorRT FP16
- **최고 정확도 필요**: `edgeface_s_gamma_05` + TensorRT FP16
- **극도로 제한된 환경**: `edgeface_xxs` + TensorRT FP16
- **학습/평가 목적**: `edgeface_base` (PyTorch)

---

## 10. 전체 코드 구조

```
Atec_EdgeFace_lee/
├── 📦 모델 변환
│   ├── torch2onnx.py                        # PyTorch → ONNX 변환
│   ├── onnx_to_tensorrt.py                  # ONNX → TensorRT 변환 (Jetson 전용)
│   └── onnx_helper.py                       # ONNX 유틸리티 (ArcFaceORT)
│
├── 🧠 인식 모듈
│   ├── edgeface_jetson_recognizer.py        # TensorRT 추론 엔진 래퍼
│   ├── face_recognition_jetson_system.py    # 실시간 인식 시스템 (CLI)
│   └── face_recognition_jetson_gui.py       # 실시간 인식 시스템 (GUI)
│
├── 📊 평가/벤치마크
│   ├── jetson_pytorch_comparison.py         # PyTorch vs TensorRT 성능 비교
│   ├── lfw_pytorch_tensorrt_comparison.py   # LFW 데이터셋 정확도 비교
│   └── lfw_evaluation.py                   # LFW 평가
│
├── 💾 체크포인트
│   ├── checkpoints/
│   │   ├── edgeface_xs_gamma_06.pt          # PyTorch 원본
│   │   ├── edgeface_xs_gamma_06.onnx        # ONNX 변환 결과
│   │   └── edgeface_xs_gamma_06.trt         # TensorRT 엔진 (Jetson에서 생성)
│
├── 👤 얼굴 감지 (Face Detection)
│   └── face_alignment/
│       ├── models/
│       │   ├── face_detection_yunet_2023mar.onnx  # YuNet (권장)
│       │   ├── yolov8n-face.onnx                  # YOLOv8 Face
│       │   └── yolov5_face.onnx                   # YOLOv5 Face
│       └── ...
│
├── 📝 문서
│   ├── JETSON_ONNX_DEPLOY_GUIDE.md          # 이 문서
│   ├── JETSON_TENSORRT_GUIDE.md             # TensorRT 가속화 가이드
│   ├── JETSON_SAFETY_GUIDE.md               # 시스템 안전 가이드
│   └── README.md                            # 프로젝트 전체 소개
│
└── 🛠️ 학습
    ├── backbones/                            # 모델 아키텍처 정의
    ├── train_v2.py                           # 학습 스크립트
    └── configs/                              # 학습 설정 파일
```

---

## 11. 자주 발생하는 문제 및 해결

### 11.1 ONNX 변환 실패

**증상**: `torch.onnx.export()` 에러

**해결 방법**:
```bash
# 1. opset 버전 변경
# torch2onnx.py 내부의 opset_version을 12 또는 13으로 변경

# 2. ONNX 단순화 적용
pip install onnxsim
python -m onnxsim model.onnx model_simplified.onnx
```

### 11.2 TensorRT 엔진 생성 실패

**증상**: `Failed to build TensorRT engine`

**해결 방법**:
```bash
# 1. 워크스페이스 크기 늘리기
python onnx_to_tensorrt.py --input model.onnx --output model.trt --workspace 2048

# 2. ONNX 단순화 후 재변환
pip install onnxsim
python -m onnxsim model.onnx model_simplified.onnx
python onnx_to_tensorrt.py --input model_simplified.onnx --output model.trt --fp16
```

### 11.3 PyCUDA / CUDA 오류

**증상**: `cuInit failed: no CUDA-capable device is detected`

**해결 방법**:
```bash
# CUDA 환경 변수 설정
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 재부팅
sudo reboot
```

### 11.4 메모리 부족 (OOM)

**증상**: `CUDA out of memory` 또는 시스템 느려짐

**해결 방법**:
```bash
# 1. 스왑 메모리 추가 (8GB 권장)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 적용
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 2. 불필요한 프로세스 종료
sudo systemctl stop gdm    # GUI 끄기 (메모리 절약)
```

### 11.5 TensorRT 버전 불일치

**증상**: `engine was built for different version`

**해결 방법**:
```bash
# 현재 Jetson에서 엔진을 다시 생성해야 합니다
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06.trt \
    --fp16
```

### 11.6 카메라 열기 실패

**증상**: `Cannot open camera`

**해결 방법**:
```bash
# 카메라 장치 확인
v4l2-ctl --list-devices

# 권한 설정
sudo chmod 666 /dev/video0

# 다른 카메라 ID로 실행
python face_recognition_jetson_system.py --camera 1
```

---

## 12. 참고 자료

| 항목 | 링크 |
|------|------|
| TensorRT Documentation | https://docs.nvidia.com/deeplearning/tensorrt/ |
| Jetson Orin Nano Developer Guide | https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit |
| EdgeFace Paper (arXiv) | https://arxiv.org/abs/2307.01838 |
| PyTorch for Jetson | https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 |
| ONNX Documentation | https://onnx.ai/onnx/intro/ |
| JetPack SDK | https://developer.nvidia.com/embedded/jetpack |

---

## Quick Start (빠른 시작)

Jetson에서 가장 빠르게 실행하는 방법:

```bash
# 1. ONNX가 이미 있다면, TensorRT 엔진 생성 (최초 1회)
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06.trt \
    --fp16

# 2. 엔진 검증
python onnx_to_tensorrt.py --verify checkpoints/edgeface_xs_gamma_06.trt

# 3. 카메라로 실시간 얼굴 인식 실행
python face_recognition_jetson_system.py --camera 0
```
