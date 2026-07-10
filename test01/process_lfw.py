import os
import cv2
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    lfw_dir = os.path.join(project_dir, 'lfw_dataset', 'lfw-deepfunneled')
    model_path = os.path.join(script_dir, 'yunet_fp32.onnx')
    output_dir = os.path.join(script_dir, 'lfw_outputs')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    txt_path = os.path.join(output_dir, 'lfw_detection_results.txt')
    
    # LFW 데이터셋에서 첫 5개의 이미지 경로 수집
    image_paths = []
    people = sorted(os.listdir(lfw_dir))
    for person in people:
        person_dir = os.path.join(lfw_dir, person)
        if os.path.isdir(person_dir):
            images = sorted([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
            for img in images:
                image_paths.append((person, os.path.join(person_dir, img)))
                if len(image_paths) == 5:
                    break
        if len(image_paths) == 5:
            break
            
    print(f"Selected 5 LFW images: {[os.path.basename(p[1]) for p in image_paths]}")
    
    with open(txt_path, 'w', encoding='utf-8') as f_out:
        f_out.write("==================================================\n")
        f_out.write("LFW Face Detection Results using YuNet FP32\n")
        f_out.write("==================================================\n\n")
        
        for idx, (person, img_path) in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            f_out.write(f"[{idx+1}/5] Image: {img_name} (Person: {person})\n")
            print(f"Processing {img_name}...")
            
            img = cv2.imread(img_path)
            if img is None:
                f_out.write(f"  Error: Cannot read image at {img_path}\n\n")
                continue
                
            h, w, _ = img.shape
            f_out.write(f"  Image Size: {w}x{h}\n")
            
            try:
                detector = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(w, h),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000
                )
                retval, faces = detector.detect(img)
                
                if retval and faces is not None:
                    f_out.write(f"  Detected {len(faces)} face(s):\n")
                    for i, face in enumerate(faces):
                        bbox = face[0:4].astype(int)
                        landmarks = face[4:14].reshape(5, 2).astype(int)
                        score = face[14]
                        
                        f_out.write(f"    - Face {i+1}:\n")
                        f_out.write(f"      Bounding Box (x, y, w, h): {bbox.tolist()}\n")
                        f_out.write(f"      Landmarks: {landmarks.tolist()}\n")
                        f_out.write(f"      Confidence Score: {score:.6f}\n")
                        
                        # 바운딩 박스(초록색) 및 랜드마크(빨간색) 그리기
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                        for pt in landmarks:
                            cv2.circle(img, tuple(pt), 3, (0, 0, 255), -1)
                        cv2.putText(img, f"Conf: {score:.4f}", (bbox[0], bbox[1] - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    f_out.write("  No face detected.\n")
            except Exception as e:
                f_out.write(f"  Error running detector: {str(e)}\n")
                
            out_img_path = os.path.join(output_dir, f"result_{img_name}")
            cv2.imwrite(out_img_path, img)
            f_out.write(f"  Saved visualized output to: result_{img_name}\n\n")
            
        f_out.write("==================================================\n")
        f_out.write("End of results.\n")
        
    print(f"Results saved successfully to {txt_path}")

if __name__ == '__main__':
    main()
