# Jetson Orin Nano TensorRT 가속화 가이드

Jetson Orin Nano에서 EdgeFace 모델을 TensorRT로 가속화하여 실시간 얼굴 인식 성능을 최적화하는 가이드입니다.

## 목차

1. [환경 요구사항](#환경-요구사항)
2. [TensorRT 엔진 생성](#tensorrt-엔진-생성)
3. [GUI 실행](#gui-실행)
4. [성능 비교](#성능-비교)
5. [트러블슈팅](#트러블슈팅)

---

## 환경 요구사항

### 하드웨어
- **Jetson Orin Nano** (8GB 권장)
- USB 카메라 또는 CSI 카메라

### 소프트웨어
- **JetPack 5.x** 이상 (TensorRT 8.5+ 포함)
- Python 3.8+
- OpenCV 4.x

### Python 패키지 설치

```bash
# 기본 패키지 (Jetson에 이미 설치되어 있음)
# - tensorrt
# - pycuda

# 추가 패키지
pip install numpy opencv-python pillow timm onnx

# GUI용 (Jetson에 기본 설치됨)
# - tkinter
```

### JetPack 버전 확인

```bash
# JetPack 버전 확인
cat /etc/nv_tegra_release

# TensorRT 버전 확인
python3 -c "import tensorrt; print(tensorrt.__version__)"

# CUDA 버전 확인
nvcc --version
```

---

## TensorRT 엔진 생성

### 1단계: ONNX 모델 확인

먼저 ONNX 모델이 있는지 확인합니다:

```bash
ls -la checkpoints/edgeface_xs_gamma_06.onnx
```

ONNX 모델이 없으면 PyTorch 모델에서 변환합니다:

```bash
python torch2onnx.py checkpoints/edgeface_xs_gamma_06.pt \
    --network edgeface_xs_gamma_06 \
    --output checkpoints/edgeface_xs_gamma_06.onnx
```

### 2단계: TensorRT 엔진 생성

```bash
# FP16 모드 (권장 - 가장 빠르고 정확도 유지)
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06.trt \
    --fp16

# FP32 모드 (최대 정확도, 느림)
python onnx_to_tensorrt.py \
    --input checkpoints/edgeface_xs_gamma_06.onnx \
    --output checkpoints/edgeface_xs_gamma_06_fp32.trt \
    --no-fp16
```

### 3단계: 엔진 검증

```bash
python onnx_to_tensorrt.py --verify checkpoints/edgeface_xs_gamma_06.trt
```

> ⚠️ **중요**: TensorRT 엔진은 **생성된 디바이스에서만** 사용할 수 있습니다. 다른 Jetson 디바이스에서 사용하려면 해당 디바이스에서 다시 생성해야 합니다.

---

## GUI 실행

### Jetson 전용 GUI 실행

```bash
python face_recognition_jetson_gui.py
```

### GUI 기능

| 기능 | 설명 |
|------|------|
| 🚀 TensorRT 체크박스 | TensorRT/PyTorch 모드 전환 |
| ⚡ FP16 체크박스 | 반정밀도 연산 사용 |
| 📊 Run Benchmark | 현재 설정으로 성능 측정 |
| 실시간 성능 표시 | FPS, Detection/Recognition 시간, 온도 |

### 키보드 단축키 (카메라 뷰)

- `q`: 종료
- `t`: TensorRT/PyTorch 전환
- `b`: 벤치마크 실행
- `c`: 얼굴 캡처

---

## 성능 비교

### 빠른 비교 실행

```bash
python jetson_pytorch_comparison.py
```

### 상세 비교 실행

```bash
python jetson_pytorch_comparison.py \
    --iterations 200 \
    --accuracy-samples 100 \
    --output benchmark_results
```

### 예상 결과 (Jetson Orin Nano)

| 항목 | PyTorch (FP32) | TensorRT (FP16) | 향상 |
|------|----------------|-----------------|------|
| 추론 시간 | ~25ms | ~8ms | ~3x |
| FPS | ~40 | ~120 | ~3x |
| 메모리 | ~1.5GB | ~0.8GB | ~2x |

> 실제 성능은 JetPack 버전, 전력 모드, 시스템 상태에 따라 다를 수 있습니다.

### 결과 파일

벤치마크 실행 후 `benchmark_results/` 폴더에 다음 파일이 생성됩니다:

- `benchmark_results.json`: 수치 결과
- `latency_comparison.png`: 속도 비교 차트
- `accuracy_comparison.png`: 정확도 비교 차트

---

## 전력 모드 설정

Jetson의 전력 모드에 따라 성능이 달라집니다:

```bash
# 현재 전력 모드 확인
sudo nvpmodel -q

# 최대 성능 모드 (MAXN)
sudo nvpmodel -m 0
sudo jetson_clocks

# 15W 모드
sudo nvpmodel -m 1

# 모드 목록 확인
sudo nvpmodel -p
```

---

## 트러블슈팅

### TensorRT 엔진 생성 실패

**증상**: `Failed to build TensorRT engine`

**해결 방법**:
1. 워크스페이스 크기 늘리기:
   ```bash
   python onnx_to_tensorrt.py --input model.onnx --output model.trt --workspace 2048
   ```

2. ONNX 모델 단순화:
   ```bash
   pip install onnxsim
   python -m onnxsim model.onnx model_simplified.onnx
   ```

### PyCUDA 오류

**증상**: `pycuda._driver.LogicError: cuInit failed: no CUDA-capable device is detected`

**해결 방법**:
```bash
# CUDA 환경 변수 설정
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 재부팅
sudo reboot
```

### 메모리 부족

**증상**: `CUDA out of memory` 또는 시스템 느려짐

**해결 방법**:
1. 스왑 메모리 늘리기:
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. 사용하지 않는 프로세스 종료

### 카메라 열기 실패

**증상**: `Cannot open camera`

**해결 방법**:
```bash
# 카메라 장치 확인
v4l2-ctl --list-devices

# 권한 확인
sudo chmod 666 /dev/video0

# 다른 카메라 ID 시도
python face_recognition_jetson_gui.py  # Camera ID: 1로 변경
```

### TensorRT 버전 불일치

**증상**: `engine was built for different version`

**해결 방법**:
해당 Jetson에서 TensorRT 엔진을 다시 생성하세요:
```bash
python onnx_to_tensorrt.py --input checkpoints/edgeface_xs_gamma_06.onnx --output checkpoints/edgeface_xs_gamma_06.trt --fp16
```

---

## 파일 구조

```
Atec_EdgeFace_lee/
├── checkpoints/
│   ├── edgeface_xs_gamma_06.pt      # PyTorch 모델
│   ├── edgeface_xs_gamma_06.onnx    # ONNX 모델
│   └── edgeface_xs_gamma_06.trt     # TensorRT 엔진 (Jetson에서 생성)
├── onnx_to_tensorrt.py              # ONNX → TensorRT 변환
├── edgeface_jetson_recognizer.py    # TensorRT 인퍼런스 클래스
├── face_recognition_jetson_system.py # Jetson 시스템
├── face_recognition_jetson_gui.py   # Jetson GUI
├── jetson_pytorch_comparison.py     # 성능 비교
└── JETSON_TENSORRT_GUIDE.md         # 이 문서
```

---

## 참고 자료

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Jetson Orin Nano Developer Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
- [EdgeFace Paper](https://arxiv.org/abs/2307.01838)
