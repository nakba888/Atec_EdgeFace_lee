# NPU 빠른 시작 가이드

DeepX NPU를 사용하여 EdgeFace를 실행하는 빠른 가이드입니다.

## 🚀 빠른 시작 (임베디드 보드에서)

### 1단계: 모델 파일 준비

NPU 모델 파일(.dxnn)을 지정된 위치에 복사:

```bash
# YuNet detector 모델
cp face_detection_yunet_2023mar.dxnn face_alignment/models/

# EdgeFace recognizer 모델
cp edgeface_xs_gamma_06.dxnn checkpoints/
```

### 2단계: 의존성 확인

```bash
# DeepX SDK 확인
python -c "from dx_engine import InferenceEngine; print('✅ DeepX SDK OK')"

# 기타 라이브러리
pip install opencv-python numpy pillow torch
```

### 3단계: 테스트 실행

```bash
# NPU 모델 테스트
python test_npu_inference.py
```

### 4단계: GUI 실행

```bash
python face_recognition_gui.py
```

GUI에서:
1. **Device**: `npu` 선택
2. **Face Detector**: `yunet_npu` 선택
3. **Start Camera** 클릭

## 📋 체크리스트

임베디드 보드에 배포하기 전 확인사항:

- [ ] DeepX NPU SDK 설치됨
- [ ] `face_detection_yunet_2023mar.dxnn` 파일 존재
- [ ] `edgeface_xs_gamma_06.dxnn` 파일 존재
- [ ] `test_npu_inference.py` 정상 실행
- [ ] 카메라 접근 권한 확인 (`/dev/video*`)

## 🔍 문제 발생 시

### dx_engine import 오류
```bash
cd /your-dxrt-directory/python_package
pip install .
```

### 모델 파일 없음
```bash
# 모델 파일 위치 확인
ls -lh face_alignment/models/face_detection_yunet_2023mar.dxnn
ls -lh checkpoints/edgeface_xs_gamma_06.dxnn
```

### 카메라 오류
```bash
# 사용 가능한 카메라 확인
v4l2-ctl --list-devices

# 카메라 권한 확인
ls -l /dev/video*

# GUI에서 Camera ID 변경 (0, 1, 2, ...)
```

## 🎯 예상 동작

### test_npu_inference.py 성공 예시:

```
Testing YuNet NPU Model
============================================================
✅ Model loaded successfully
📊 Input size: [1, 3, 640, 640]
✅ Decoded 3 faces

Testing EdgeFace NPU Model
============================================================
✅ Model loaded successfully
📊 Input size: [1, 3, 112, 112]
✅ Extracted embedding: shape=(512,), norm=1.0000
```

### GUI 동작:

1. 카메라 영상이 표시됨
2. 얼굴이 검출되면 초록색 박스 표시
3. 등록된 사람은 이름과 유사도 표시
4. 미등록 사람은 "Unknown" 표시
5. FPS가 화면 좌측 상단에 표시

## 📚 추가 문서

- [전체 통합 가이드](NPU_INTEGRATION.md) - 상세 설명 및 구조
- [DeepX 문서](https://github.com/DEEPX-AI/dx-all-suite.git) - NPU SDK 공식 문서
- [calibration 설정](npu_calibration/) - 모델 변환 참고

## ⚡ 성능 팁

### CPU vs NPU 비교
Raspberry Pi 5 기준
| 모드 | 예상 FPS | 전력 소비 |
|------|----------|-----------|
| CPU  | ~4-5   | 높음      |
| NPU  | ~10-13   | 낮음      |

### 최적화 방법

1. **해상도 조정**: 카메라 해상도를 640x480으로 설정
2. **Threshold 조정**: Similarity threshold를 0.5-0.6으로 설정
3. **Multi-face**: 많은 얼굴 검출 시 성능 저하 가능

## 🐛 디버깅 모드

출력 형식 확인이 필요한 경우:

```python
# test_npu_inference.py 수정
# 더 상세한 출력 정보 확인
```

YuNet 출력이 예상과 다른 경우:
- `face_alignment/yunet_npu.py`의 `_decode_outputs()` 함수 수정
- 출력 텐서 shape 확인
- confidence threshold 조정

## 📞 지원

문제가 해결되지 않으면:
1. `test_npu_inference.py` 출력 결과 확인
2. 모델 파일 크기/무결성 확인
3. NPU 드라이버 로그 확인
4. DeepX 공식 문서 참조
