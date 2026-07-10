import os
import cv2
import numpy as np
import onnxruntime as ort

def process_scale(cls_scores, obj_scores, bboxes, landmarks, stride, input_size, score_threshold=0.5):
    detections = []
    cls_scores = cls_scores.flatten()
    obj_scores = obj_scores.flatten()
    
    # YuNet confidence score = cls * obj
    scores = cls_scores * obj_scores
    
    feat_size = input_size // stride
    valid_indices = np.where(scores >= score_threshold)[0]
    
    for idx in valid_indices:
        score = float(scores[idx])
        bbox = bboxes[idx]
        
        anchor_y = idx // feat_size
        anchor_x = idx % feat_size
        
        # Decode center (cx, cy)
        cx = (anchor_x + bbox[0]) * stride
        cy = (anchor_y + bbox[1]) * stride
        
        # Decode size (w, h)
        prior_w = stride * 3
        prior_h = stride * 3.6
        w = bbox[2] * prior_w
        h = bbox[3] * prior_h
        
        # Convert to top-left format
        x = cx - w / 2
        y = cy - h / 2
        
        # Decode landmarks (5 points * 2 coordinates)
        lms = landmarks[idx]
        decoded_lms = []
        for i in range(5):
            lm_x = (anchor_x + lms[i*2]) * stride
            lm_y = (anchor_y + lms[i*2+1]) * stride
            decoded_lms.extend([lm_x, lm_y])
            
        detections.append([x, y, w, h] + decoded_lms + [score])
        
    return detections

def apply_nms(detections, nms_threshold=0.3):
    if not detections:
        return []
        
    faces_array = np.array(detections)
    x = faces_array[:, 0]
    y = faces_array[:, 1]
    w = faces_array[:, 2]
    h = faces_array[:, 3]
    scores = faces_array[:, 14]
    
    x2 = x + w
    y2 = y + h
    areas = w * h
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
            
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= nms_threshold)[0]
        order = order[inds + 1]
        
    return [detections[i] for i in keep]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'yunet_fp32.onnx')
    image_path = os.path.join(script_dir, 'lena.jpg')
    output_dir = os.path.join(script_dir, 'lfw_outputs')
    
    txt_path = os.path.join(output_dir, 'nms_demo_results.txt')
    
    session = ort.InferenceSession(model_path)
    
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    
    input_size = 640
    resized = cv2.resize(img, (input_size, input_size))
    blob = resized.astype(np.float32)
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    
    # 모델 실행
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    
    # ONNX 출력 텐서 매핑
    cls_8, cls_16, cls_32 = outputs[0][0], outputs[1][0], outputs[2][0]
    obj_8, obj_16, obj_32 = outputs[3][0], outputs[4][0], outputs[5][0]
    bbox_8, bbox_16, bbox_32 = outputs[6][0], outputs[7][0], outputs[8][0]
    kps_8, kps_16, kps_32 = outputs[9][0], outputs[10][0], outputs[11][0]
    
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    
    # 각 스케일별 디코딩 (NMS 전 후보군 추출)
    candidates = []
    candidates.extend(process_scale(cls_8, obj_8, bbox_8, kps_8, 8, input_size, score_threshold=0.5))
    candidates.extend(process_scale(cls_16, obj_16, bbox_16, kps_16, 16, input_size, score_threshold=0.5))
    candidates.extend(process_scale(cls_32, obj_32, bbox_32, kps_32, 32, input_size, score_threshold=0.5))
    
    # 원본 이미지 크기로 좌표 복원
    for c in candidates:
        c[0] *= scale_x
        c[1] *= scale_y
        c[2] *= scale_x
        c[3] *= scale_y
        for i in range(4, 14, 2):
            c[i] *= scale_x
            c[i+1] *= scale_y
            
    # NMS 적용 (임계치 0.3)
    final_faces = apply_nms(candidates, nms_threshold=0.3)
    
    # 결과 파일 쓰기
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("======================================================================\n")
        f.write("YuNet Face Detection - NMS (Non-Maximum Suppression) Demo Results\n")
        f.write("Target Image: lena.jpg (Original Size: {}x{})\n".format(orig_w, orig_h))
        f.write("======================================================================\n\n")
        
        f.write("--- BEFORE NMS (Raw decoded candidates with confidence >= 0.5) ---\n")
        f.write(f"Total raw candidates found: {len(candidates)}\n\n")
        
        # 신뢰도 내림차순 정렬
        sorted_candidates = sorted(candidates, key=lambda x: x[14], reverse=True)
        for i, c in enumerate(sorted_candidates):
            f.write(f"Candidate {i+1}:\n")
            f.write(f"  Bounding Box (x, y, w, h): {[round(val, 2) for val in c[0:4]]}\n")
            f.write(f"  Confidence Score: {c[14]:.6f}\n")
            f.write(f"  Landmarks: {[ [round(c[4+2*j], 2), round(c[4+2*j+1], 2)] for j in range(5) ]}\n\n")
            
        f.write("\n" + "-"*70 + "\n\n")
        
        f.write("--- AFTER NMS (Deduplicated final detections with NMS IoU threshold = 0.3) ---\n")
        f.write(f"Total final faces found: {len(final_faces)}\n\n")
        for i, c in enumerate(final_faces):
            f.write(f"Face {i+1}:\n")
            f.write(f"  Bounding Box (x, y, w, h): {[round(val, 2) for val in c[0:4]]}\n")
            f.write(f"  Confidence Score: {c[14]:.6f}\n")
            f.write(f"  Landmarks: {[ [round(c[4+2*j], 2), round(c[4+2*j+1], 2)] for j in range(5) ]}\n\n")
            
        f.write("\n======================================================================\n")
        f.write("End of NMS Demo.\n")
        
    print(f"NMS Demo results saved to {txt_path}")

if __name__ == '__main__':
    main()
