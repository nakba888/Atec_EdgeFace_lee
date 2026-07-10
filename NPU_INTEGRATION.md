# DeepX NPU Integration Guide

EdgeFace 얼굴 인식 시스템에 DeepX NPU 지원이 추가되었습니다.

## 개요

DeepX NPU를 사용하여 YuNet 얼굴 검출과 EdgeFace 얼굴 인식을 가속화할 수 있습니다.

### ✨ 자동 NPU 활성화 (New!)

**`detector_method='yunet_npu'` 선택 시, EdgeFace 인식 모델도 자동으로 NPU로 실행됩니다!**

더 이상 `device='npu'` 또는 `use_npu=True`를 따로 설정할 필요가 없습니다.

### 지원하는 모델

1. **YuNet Face Detector** (NPU)
   - 모델 파일: `face_alignment/models/face_detection_yunet_2023mar.dxnn`
   - 입력 크기: 640x640
   - 기능: 얼굴 검출 및 랜드마크 추출 (5-point landmarks)

2. **EdgeFace Recognizer** (NPU)
   - 모델 파일: `checkpoints/edgeface_xs_gamma_06.dxnn`
   - 입력 크기: 112x112
   - 기능: 얼굴 임베딩 추출 (512-d vector)

## 파일 구조

```
EdgeFace/
├── face_alignment/
│   ├── yunet_npu.py                    # YuNet NPU detector wrapper
│   ├── unified_detector.py             # Updated with yunet_npu support
│   └── models/
│       └── face_detection_yunet_2023mar.dxnn  # YuNet NPU model
├── edgeface_npu_recognizer.py          # EdgeFace NPU recognizer
├── face_recognition_system.py          # Updated with NPU support
├── face_recognition_gui.py             # Updated GUI with NPU options
├── checkpoints/
│   ├── edgeface_xs_gamma_06.pt         # PyTorch model
│   └── edgeface_xs_gamma_06.dxnn       # NPU model
└── test_npu_inference.py               # NPU model testing script
```

## 사전 요구사항

### DeepX NPU SDK 설치

NPU를 사용하려면 DeepX NPU SDK가 설치되어 있어야 합니다.

```bash
# DeepX SDK 설치 (DeepX 문서 참조)
cd /your-dxrt-directory/python_package
pip install .
```

설치 확인:
```bash
python -c "from dx_engine import InferenceEngine; print('dx_engine available')"
```

### 모델 파일 준비

NPU 모델 파일(.dxnn)을 지정된 위치에 배치:

1. **YuNet NPU 모델**
   ```bash
   cp face_detection_yunet_2023mar.dxnn face_alignment/models/
   ```

2. **EdgeFace NPU 모델**
   ```bash
   cp edgeface_xs_gamma_06.dxnn checkpoints/
   ```

## 사용 방법

### 1. GUI에서 사용

GUI를 실행하고 NPU 옵션을 선택:

```bash
python face_recognition_gui.py
```

**설정 방법:**
1. **Face Detector** 드롭다운에서 `yunet_npu` 선택
2. **Device** 드롭다운에서 `npu` 선택
3. **Start Camera** 버튼 클릭

또는:
- **Device**를 `npu`로 설정하면 자동으로 `yunet`이 `yunet_npu`로 전환됩니다.

### 2. Python 코드에서 사용

#### 방법 1: 자동 NPU 활성화 (권장)

```python
from face_recognition_system import FaceRecognitionSystem

# yunet_npu만 선택하면 EdgeFace도 자동으로 NPU로 실행됩니다!
system = FaceRecognitionSystem(
    detector_method='yunet_npu',  # 이것만으로 전체 NPU 활성화!
    edgeface_model_path='checkpoints/edgeface_xs_gamma_06.dxnn',
    edgeface_model_name='edgeface_xs_gamma_06',
    similarity_threshold=0.5
)

# 카메라 실행
system.run_camera(camera_id=0)
```

#### 방법 2: 명시적 NPU 설정 (선택적)

```python
from face_recognition_system import FaceRecognitionSystem

# 명시적으로 NPU 설정 (동일한 결과)
system = FaceRecognitionSystem(
    detector_method='yunet_npu',
    edgeface_model_path='checkpoints/edgeface_xs_gamma_06.dxnn',
    edgeface_model_name='edgeface_xs_gamma_06',
    device='npu',        # 명시적 설정
    use_npu=True,        # 명시적 설정
    similarity_threshold=0.5
)

# 카메라 실행
system.run_camera(camera_id=0)
```

### 3. 명령줄에서 NPU 사용

#### 간단한 방법 (자동 NPU 활성화)

```bash
# yunet_npu 선택만으로 전체 NPU 파이프라인 실행
python3 face_recognition_system.py \
    --detector yunet_npu \
    --model checkpoints/edgeface_xs_gamma_06.dxnn \
    --model-name edgeface_xs_gamma_06
```

#### 명시적 방법

```bash
# device=npu 추가 (선택적)
python3 face_recognition_system.py \
    --detector yunet_npu \
    --model checkpoints/edgeface_xs_gamma_06.dxnn \
    --model-name edgeface_xs_gamma_06 \
    --device npu
```

### 4. NPU 모델 테스트

#### 전체 시스템 테스트

```bash
# YuNet NPU + EdgeFace NPU 통합 테스트
python3 test_npu_full_system.py
```

#### 개별 모델 테스트

```bash
# YuNet NPU 모델만 테스트
python test_npu_inference.py
```

이 스크립트는:
- YuNet NPU 모델의 입출력 형식 확인
- EdgeFace NPU 모델의 입출력 형식 확인
- 테스트 이미지로 추론 실행

## 구현 세부사항

### 자동 NPU 활성화 로직

`FaceRecognitionSystem.__init__()`에서 자동으로 NPU를 감지합니다:

```python
# face_recognition_system.py:383
# Determine if we should use NPU
# Automatically enable NPU for EdgeFace when yunet_npu is used
self.use_npu = use_npu or device == 'npu' or detector_method == 'yunet_npu'
```

이 한 줄로:
1. `detector_method='yunet_npu'` 선택 시 자동으로 `use_npu=True` 설정
2. EdgeFace 초기화 시 NPU 모드로 자동 전환
3. 별도의 설정 없이 전체 파이프라인이 NPU에서 실행

```python
# face_recognition_system.py:397-410
if self.use_npu:
    if not NPU_AVAILABLE:
        raise ImportError("NPU recognizer not available. Install DeepX NPU SDK.")
    # Use NPU recognizer (자동 선택!)
    self.recognizer = EdgeFaceNPURecognizer(
        edgeface_model_path,
        edgeface_model_name,
        device='npu'
    )
else:
    # Use PyTorch recognizer
    self.recognizer = EdgeFaceRecognizer(
        edgeface_model_path,
        edgeface_model_name,
        self.device
    )
```

### YuNet NPU Detector

**파일:** `face_alignment/yunet_npu.py`

**전처리:**
1. 이미지를 640x640으로 리사이즈
2. BGR → RGB 변환
3. HWC → CHW 전치
4. 배치 차원 추가 (1, 3, 640, 640)
5. uint8 타입으로 변환

**후처리:**
- YuNet 출력을 얼굴 검출 결과로 디코딩
- 각 얼굴: [x, y, w, h, x1, y1, ..., x5, y5, confidence]
- 좌표를 원본 이미지 크기로 스케일 변환

### EdgeFace NPU Recognizer

**파일:** `edgeface_npu_recognizer.py`

**전처리:**
1. 얼굴 이미지를 112x112로 리사이즈
2. BGR → RGB 변환
3. [0, 255] → [0, 1] 정규화 (÷ 255.0)
4. 표준화: (x - 0.5) / 0.5
5. HWC → CHW 전치
6. 배치 차원 추가 (1, 3, 112, 112)
7. float32 타입으로 변환

**후처리:**
- 출력 텐서를 512-d 벡터로 평탄화
- L2 정규화

## 성능 비교

| 모드 | Detector | Recognizer | 예상 속도 |
|------|----------|------------|-----------|
| CPU  | YuNet (OpenCV) | EdgeFace (PyTorch) | ~10-15 FPS |
| GPU  | YuNet (OpenCV) | EdgeFace (PyTorch CUDA) | ~30-50 FPS |
| NPU  | YuNet (DeepX) | EdgeFace (DeepX) | ~40-60 FPS* |

\* 실제 성능은 NPU 하드웨어에 따라 다름

## 문제 해결

### 1. dx_engine import 실패

```
ImportError: cannot import name 'InferenceEngine' from 'dx_engine'
```

**해결:** DeepX SDK를 설치하세요.
```bash
cd /your-dxrt-directory/python_package
pip install .
```

### 2. 모델 파일을 찾을 수 없음

```
RuntimeError: YuNet NPU detector 초기화 실패
```

**해결:** 모델 파일이 올바른 위치에 있는지 확인:
- `face_alignment/models/face_detection_yunet_2023mar.dxnn`
- `checkpoints/edgeface_xs_gamma_06.dxnn`

### 3. YuNet 출력 형식 확인

YuNet NPU의 출력 디코딩은 다음과 같은 형식을 가정합니다:

**예상 출력 형식:**
- **Shape**: `(N, 15)` 또는 `(1, N, 15)`
  - N: 검출된 얼굴 개수
  - 15: [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, confidence]

**디버깅 방법:**

임베디드 보드에서 실행:
```bash
python test_npu_inference.py
```

출력 예시:
```
Testing YuNet NPU Model
============================================================
Loading model: face_alignment/models/face_detection_yunet_2023mar.dxnn
✅ Model loaded successfully
📊 Input size: [1, 3, 640, 640]
📊 Output dtype: ['FLOAT32']

Loading test image: test.jpg
Image shape: (480, 640, 3)
Input tensor shape: (1, 3, 640, 640), dtype: uint8

🚀 Running inference...

📊 Number of outputs: 1
  Output 0:
    - Shape: (5, 15)           # 5개 얼굴 검출
    - Dtype: float32
    - Min: -12.3456, Max: 456.7890
    - Mean: 123.4567, Std: 89.0123
    - Flattened shape: (75,)
    - First 10 values: [120.5, 85.3, 60.2, 75.8, 115.2, ...]
    - ✅ Detected YuNet format: (N=5, 15)
    - Confidence scores (column 14): [0.95, 0.88, 0.82, 0.76, 0.65]

🔍 Testing decoder...
✅ Decoded 5 faces
  Face 0: confidence=0.950, bbox=[120.5, 85.3, 60.2, 75.8]
  Face 1: confidence=0.880, bbox=[250.1, 120.4, 58.9, 72.3]
  ...
```

**만약 출력 형식이 다른 경우:**

출력이 다른 형태(예: 여러 텐서, 다른 차원)라면 `yunet_npu.py`의 `_decode_outputs()` 함수를 수정해야 합니다.

예시 수정 위치 (face_alignment/yunet_npu.py:124):
```python
def _decode_outputs(self, outputs, scale_x, scale_y):
    # 출력 형식에 맞게 수정
    # ...
```

### 4. 성능 및 메모리 이슈

**느린 추론 속도:**
- NPU 드라이버가 올바르게 설치되었는지 확인
- `test_npu_inference.py`로 추론 시간 측정
- calibration 설정 확인

**메모리 부족:**
```python
# 배치 크기 조정 (현재는 항상 1)
# EdgeFace recognizer의 extract_embeddings_batch() 사용 시 주의
```

### 5. 임베디드 보드 배포 체크리스트

1. ✅ DeepX SDK 설치 확인
   ```bash
   python -c "from dx_engine import InferenceEngine; print('OK')"
   ```

2. ✅ 모델 파일 복사
   ```bash
   ls face_alignment/models/face_detection_yunet_2023mar.dxnn
   ls checkpoints/edgeface_xs_gamma_06.dxnn
   ```

3. ✅ 테스트 스크립트 실행
   ```bash
   python test_npu_inference.py
   ```

4. ✅ GUI 테스트
   ```bash
   python face_recognition_gui.py
   # Device: npu 선택
   # Detector: yunet_npu 선택
   ```

5. ✅ 카메라 권한 확인
   ```bash
   v4l2-ctl --list-devices
   ls -l /dev/video*
   ```

## 모델 변환 (참고)

ONNX 모델을 DXNN으로 변환하는 방법 (DeepX 문서 참조):

```bash
# YuNet 변환 예시
dx-com -m face_detection_yunet_2023mar.onnx \
       -c calibration_config_yunet.json \
       -o face_detection_yunet_2023mar.dxnn

# EdgeFace 변환 예시
dx-com -m edgeface_xs_gamma_06.onnx \
       -c calibration_config_edgeface.json \
       -o edgeface_xs_gamma_06.dxnn
```

calibration config 파일은 `npu_calibration/` 디렉토리를 참조하세요.

## 추가 정보

- DeepX SDK 문서: `npu_calibration/deepX_document/`
- Python 예제: `npu_calibration/deepX_document/07_Python_Examples.md`
- YuNet calibration config: `npu_calibration/calibration_output/calibration_config_yunet.json`
- EdgeFace calibration config: `npu_calibration/edgeface_calibration_output/calibration_config_edgeface.json`

## 🔧 YuNet NPU 상세 가이드

### YuNet NPU 출력 구조

YuNet ONNX 모델을 DeepX NPU로 컴파일하면 **출력 구조가 변경**됩니다:

#### 원본 ONNX 모델 출력 (12개)
- Feature Pyramid Network (FPN) 기반 3-scale detection
- 각 scale당 4가지 출력: cls, obj, bbox, kps
- 출력 이름: `cls_8`, `cls_16`, `cls_32`, `obj_8`, `obj_16`, `obj_32`, `bbox_8`, `bbox_16`, `bbox_32`, `kps_8`, `kps_16`, `kps_32`
- **Post-processing 포함**: NMS, bbox decoding 등이 ONNX 모델 내부에 구현됨

#### NPU 컴파일된 모델 출력 (13개)
- DeepX NPU 컴파일러가 **출력 순서를 재배열**
- **Post-processing 제거**: NMS, bbox decoding이 제거되어 raw feature map 출력
- 13개 텐서 출력 (spatial feature map 1개 + FPN outputs 12개)

```python
# NPU 출력 매핑 예시 (user's compiled version)
Output 0:  (1, 1, 80, 80)    # Spatial feature map (unused)
Output 1:  (1, 6400, 1)      # cls_8 (stride 8, 80x80 = 6400 anchors)
Output 2:  (1, 1600, 1)      # cls_16 (stride 16, 40x40 = 1600 anchors)
Output 3:  (1, 6400, 4)      # bbox_8
Output 4:  (1, 1600, 4)      # bbox_16
Output 5:  (1, 1600, 1)      # obj_16
Output 6:  (1, 400, 10)      # kps_32 (stride 32, 20x20 = 400 anchors)
Output 7:  (1, 6400, 10)     # kps_8
Output 8:  (1, 1600, 10)     # kps_16
Output 9:  (1, 400, 1)       # obj_32
Output 10: (1, 400, 4)       # bbox_32
Output 11: (1, 6400, 1)      # obj_8
Output 12: (1, 400, 1)       # cls_32
```

**⚠️ 중요**: NPU 컴파일러 버전이나 설정에 따라 출력 순서가 달라질 수 있습니다!

### YuNet NPU 디코딩 구현

NPU 모델의 raw 출력을 bbox/landmark로 변환하는 과정:

#### 1. Score 계산
```python
# YuNet은 cls × obj를 final confidence로 사용
score = cls_score * obj_score
```

#### 2. Anchor 기반 Bbox 디코딩
```python
# Anchor 위치 계산 (grid 좌상단 기준, 0.5 offset 없음)
feat_size = input_size // stride  # 640/32 = 20 for stride 32
anchor_y = idx // feat_size
anchor_x = idx % feat_size

# Center 디코딩 (offset scaling)
cx = (anchor_x + bbox[0] * 0.5) * stride
cy = (anchor_y + bbox[1] * 0.5) * stride

# Size 디코딩 (linear scaling with prior)
prior_size = stride * 3  # 또는 stride * 4
w = bbox[2] * prior_size
h = bbox[3] * prior_size

# Top-left corner 형식으로 변환
x = cx - w / 2
y = cy - h / 2
```

#### 3. Landmark 디코딩
```python
# Landmarks도 anchor 기준 offset
for i in range(5):  # 5 keypoints
    lm_x = (anchor_x + lms[i*2] * 1.0) * stride
    lm_y = (anchor_y + lms[i*2 + 1] * 1.0) * stride
```

#### 4. Multi-scale Detection
```python
# 3개 scale의 detection을 모두 수집
detections_stride8 = process_scale(..., stride=8)   # 80x80 feature map
detections_stride16 = process_scale(..., stride=16) # 40x40 feature map
detections_stride32 = process_scale(..., stride=32) # 20x20 feature map

# 모든 detection 합치기
all_detections = detections_stride8 + detections_stride16 + detections_stride32
```

#### 5. NMS (Non-Maximum Suppression)
```python
# 중복 detection 제거
final_detections = apply_nms(all_detections, iou_threshold=0.3)
```

### Bbox/Landmark 조정 방법

Detection 결과가 정확하지 않을 때 조정하는 방법:

#### 파일 위치
`face_alignment/yunet_npu.py`의 `_process_scale()` 메서드 (라인 ~295-325)

#### 조정 파라미터

1. **Bbox 크기 조정** (`prior_size`)
   ```python
   prior_size = stride * 3  # 이 값을 조정
   # 크게: stride * 4, stride * 5
   # 작게: stride * 2, stride * 1.5
   ```

2. **Bbox width/height 비율 조정**
   ```python
   # 정사각형이 아닌 얼굴 비율 적용
   prior_w = stride * 3    # width
   prior_h = stride * 4    # height (더 길게)
   w = bbox[2] * prior_w
   h = bbox[3] * prior_h
   ```

3. **Bbox center offset 조정**
   ```python
   # 위치 shift 조정 (현재 0.5)
   cx = (anchor_x + bbox[0] * 0.5) * stride  # 0.5를 0.3~1.0 사이로 조정
   cy = (anchor_y + bbox[1] * 0.5) * stride

   # 우하단으로 치우치면: 계수를 줄임 (0.3, 0.4)
   # 좌상단으로 치우치면: 계수를 늘림 (0.7, 0.8)
   ```

4. **Landmark offset 조정**
   ```python
   # 현재 1.0 계수 사용
   lm_x = (anchor_x + lms[i*2] * 1.0) * stride  # 1.0을 조정
   lm_y = (anchor_y + lms[i*2 + 1] * 1.0) * stride

   # Landmark가 bbox center와 함께 움직이지 않으면 bbox center 기준으로:
   lm_x = cx + lms[i*2] * w  # bbox 크기 기준
   lm_y = cy + lms[i*2 + 1] * h
   ```

5. **Anchor 0.5 offset**
   ```python
   # 현재: anchor는 grid 좌상단 (0.5 offset 없음)
   anchor_y = (idx // feat_size)
   anchor_x = (idx % feat_size)

   # Grid center를 사용하려면:
   anchor_y = (idx // feat_size) + 0.5
   anchor_x = (idx % feat_size) + 0.5
   ```

#### Debug 출력 활용

코드 실행 시 다음과 같은 debug 정보가 출력됩니다:

```
[DEBUG] Scale 3 (stride 32): score range [0.000000, 0.810580], mean=0.015924
[DEBUG]   Stride 32: 7/400 above threshold 0.6
[DEBUG]     First valid score: 0.7965272068977356
[DEBUG]     First valid bbox raw: [0.9806061 1.3796387 1.3396912 1.7766724]
[DEBUG]     First valid landmark raw: [-0.03892517  0.46404266  1.4368286 ...]
```

이 값들을 보면서:
- `bbox raw` 값이 크면 (>2.0) → offset scaling 줄이기
- `bbox[2]`, `bbox[3]` (width/height scale) → 1.0~2.0 사이면 linear, >3.0이면 exponential 고려
- Landmark 값의 범위 → bbox와 비슷한 스케일이면 같은 방식 적용

### NPU 컴파일 시 추가 주의사항

1. **출력 순서 확인 필수**
   - 다른 버전으로 컴파일하면 출력 순서가 다를 수 있음
   - `test_npu_inference.py` 실행해서 shape 확인
   - `_decode_outputs()` 메서드에서 output 매핑 수정

2. **Post-processing 제거됨**
   - ONNX 모델의 NMS, bbox decoding이 제거됨
   - Python에서 직접 구현 필요

3. **Calibration 데이터 중요**
   - YuNet은 다양한 얼굴 크기/각도 이미지로 calibration
   - `npu_calibration/` 시스템 활용 권장

4. **입력 전처리**
   - YuNet: RGB, HWC format, uint8
   - 640x640 입력 크기

## 라이선스

이 통합 코드는 EdgeFace 프로젝트의 라이선스를 따릅니다.
DeepX NPU SDK는 별도의 라이선스가 적용됩니다.
