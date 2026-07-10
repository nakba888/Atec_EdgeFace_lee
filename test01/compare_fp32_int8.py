import os
import cv2
import numpy as np

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-10)
    return iou

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    lfw_outputs_dir = os.path.join(script_dir, 'lfw_outputs')
    model_fp32 = os.path.join(script_dir, 'yunet_fp32.onnx')
    model_int8 = os.path.join(script_dir, 'yunet_int8.onnx')
    
    comparison_dir = os.path.join(script_dir, 'comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
        
    report_path = os.path.join(comparison_dir, 'numerical_comparison.txt')
    
    # lfw_outputs에 백업해 둔 original 이미지 5개 리스트 획득
    orig_images = sorted([f for f in os.listdir(lfw_outputs_dir) if f.startswith('original_') and f.endswith('.jpg')])
    
    with open(report_path, 'w', encoding='utf-8') as f_rep:
        f_rep.write("==================================================\n")
        f_rep.write("YuNet FP32 vs INT8 Face Detection Comparison Report\n")
        f_rep.write("==================================================\n\n")
        
        for idx, orig_name in enumerate(orig_images):
            img_path = os.path.join(lfw_outputs_dir, orig_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w, _ = img.shape
            base_name = orig_name.replace('original_', '')
            f_rep.write(f"[{idx+1}/5] Image: {base_name}\n")
            print(f"Comparing {base_name}...")
            
            # FP32 검출 실행
            det_fp32 = cv2.FaceDetectorYN.create(model=model_fp32, config="", input_size=(w, h), score_threshold=0.5)
            ret_f, faces_f = det_fp32.detect(img)
            
            # INT8 검출 실행
            det_int8 = cv2.FaceDetectorYN.create(model=model_int8, config="", input_size=(w, h), score_threshold=0.5)
            ret_i, faces_i = det_int8.detect(img)
            
            img_overlap = img.copy()
            img_fp32_only = img.copy()
            img_int8_only = img.copy()
            
            has_fp32 = ret_f and faces_f is not None and len(faces_f) > 0
            has_int8 = ret_i and faces_i is not None and len(faces_i) > 0
            
            if has_fp32 and has_int8:
                face_f = faces_f[0]
                face_i = faces_i[0]
                
                box_f = face_f[0:4]
                lms_f = face_f[4:14].reshape(5, 2)
                score_f = face_f[14]
                
                box_i = face_i[0:4]
                lms_i = face_i[4:14].reshape(5, 2)
                score_i = face_i[14]
                
                # 차이값 연산
                iou = calculate_iou(box_f, box_i)
                box_diff = np.abs(box_f - box_i)
                lms_diff = np.abs(lms_f - lms_i)
                score_diff = abs(score_f - score_i)
                
                f_rep.write(f"  FP32 Confidence: {score_f:.6f} | INT8 Confidence: {score_i:.6f} (Diff: {score_diff:.6f})\n")
                f_rep.write(f"  Bounding Box IoU (Overlap): {iou:.6f}\n")
                f_rep.write(f"  Bounding Box Difference (pixels):\n")
                f_rep.write(f"    x diff: {box_diff[0]:.2f}, y diff: {box_diff[1]:.2f}, w diff: {box_diff[2]:.2f}, h diff: {box_diff[3]:.2f}\n")
                f_rep.write(f"    Mean Absolute Error (MAE): {np.mean(box_diff):.4f} px\n")
                f_rep.write(f"  Landmarks Difference (pixels):\n")
                for j in range(5):
                    f_rep.write(f"    Pt {j+1} diff: [{lms_diff[j][0]:.2f}, {lms_diff[j][1]:.2f}] | Distance: {np.linalg.norm(lms_f[j] - lms_i[j]):.4f} px\n")
                f_rep.write(f"    Mean Absolute Error (MAE): {np.mean(lms_diff):.4f} px\n\n")
                
                # FP32 정보 그리기 (Green)
                cv2.rectangle(img_fp32_only, (int(box_f[0]), int(box_f[1])), (int(box_f[0]+box_f[2]), int(box_f[1]+box_f[3])), (0, 255, 0), 2)
                cv2.rectangle(img_overlap, (int(box_f[0]), int(box_f[1])), (int(box_f[0]+box_f[2]), int(box_f[1]+box_f[3])), (0, 255, 0), 2)
                for pt in lms_f.astype(int):
                    cv2.circle(img_fp32_only, tuple(pt), 3, (0, 255, 0), -1)
                    cv2.circle(img_overlap, tuple(pt), 3, (0, 255, 0), -1)
                cv2.putText(img_fp32_only, f"FP32 Conf: {score_f:.4f}", (int(box_f[0]), int(box_f[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img_overlap, f"FP32", (int(box_f[0]), int(box_f[1] - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                
                # INT8 정보 그리기 (Red)
                cv2.rectangle(img_int8_only, (int(box_i[0]), int(box_i[1])), (int(box_i[0]+box_i[2]), int(box_i[1]+box_i[3])), (0, 0, 255), 2)
                cv2.rectangle(img_overlap, (int(box_i[0]), int(box_i[1])), (int(box_i[0]+box_i[2]), int(box_i[1]+box_i[3])), (0, 0, 255), 2)
                for pt in lms_i.astype(int):
                    cv2.circle(img_int8_only, tuple(pt), 3, (0, 0, 255), -1)
                    cv2.circle(img_overlap, tuple(pt), 3, (0, 0, 255), -1)
                cv2.putText(img_int8_only, f"INT8 Conf: {score_i:.4f}", (int(box_i[0]), int(box_i[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(img_overlap, f"INT8", (int(box_i[0]), int(box_i[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                
            else:
                f_rep.write("  Error: Could not detect face in both versions.\n\n")
                
            # 중첩 이미지 저장
            cv2.imwrite(os.path.join(comparison_dir, f"overlap_{base_name}"), img_overlap)
            
            # 좌우 결합 이미지 생성 및 저장
            cv2.putText(img_fp32_only, "FP32 (Green)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img_int8_only, "INT8 (Red)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            side_by_side = cv2.hconcat([img_fp32_only, img_int8_only])
            cv2.imwrite(os.path.join(comparison_dir, f"side_by_side_{base_name}"), side_by_side)
            
        f_rep.write("==================================================\n")
        f_rep.write("End of Report.\n")
        
    print("Comparison done!")

if __name__ == '__main__':
    main()
