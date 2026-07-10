import cv2
import numpy as np
import onnxruntime as ort
import os

def run_opencv_yunet(model_path, image_path):
    print(f"\n--- OpenCV FaceDetectorYN: {os.path.basename(model_path)} ---")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image at {image_path}")
        return None
        
    h, w, _ = img.shape
    print(f"Image shape: {w}x{h}")
    
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
            print(f"Detected {len(faces)} face(s):")
            for i, face in enumerate(faces):
                bbox = face[0:4].astype(int)
                landmarks = face[4:14].reshape(5, 2).astype(int)
                score = face[14]
                print(f"  Face {i+1}:")
                print(f"    Bounding Box (x, y, w, h): {bbox.tolist()}")
                print(f"    Landmarks: {landmarks.tolist()}")
                print(f"    Confidence Score: {score:.6f}")
            return faces
        else:
            print("No face detected.")
            return None
    except Exception as e:
        print(f"Error running FaceDetectorYN: {e}")
        return None

def run_ort_raw(model_path, image_path):
    print(f"\n--- ONNX Runtime: {os.path.basename(model_path)} ---")
    
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    input_name = inputs[0].name
    input_shape = inputs[0].shape
    
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    target_w, target_h = 640, 640
    resized = cv2.resize(img, (target_w, target_h))
    
    if len(input_shape) == 4:
        if input_shape[1] == 3 or isinstance(input_shape[1], str):
            blob = resized.astype(np.float32)
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)
        else:
            blob = np.expand_dims(resized.astype(np.float32), axis=0)
    else:
        blob = np.expand_dims(resized.astype(np.float32), axis=0)
        
    ort_outputs = session.run([out.name for out in outputs], {input_name: blob})
    return {out.name: ort_outputs[i] for i, out in enumerate(outputs)}

def draw_visualizations(image_path, faces_fp32, faces_int8, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return
        
    # Draw FP32 in Green
    if faces_fp32 is not None:
        for face in faces_fp32:
            bbox = face[0:4].astype(int)
            landmarks = face[4:14].reshape(5, 2).astype(int)
            score = face[14]
            # Draw bbox
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            # Draw landmarks
            for pt in landmarks:
                cv2.circle(img, tuple(pt), 3, (0, 255, 0), -1)
            # Put text
            cv2.putText(img, f"FP32: {score:.4f}", (bbox[0], bbox[1] - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw INT8 in Red (with slight offset so we can see both if they overlap)
    if faces_int8 is not None:
        for face in faces_int8:
            bbox = face[0:4].astype(int)
            landmarks = face[4:14].reshape(5, 2).astype(int)
            score = face[14]
            # Draw bbox
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
            # Draw landmarks
            for pt in landmarks:
                cv2.circle(img, tuple(pt), 2, (0, 0, 255), -1)
            # Put text
            cv2.putText(img, f"INT8: {score:.4f}", (bbox[0], bbox[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"Visualization saved to {output_path}")

if __name__ == '__main__':
    img_path = '/home/jarvis/jarvis/Atec_EdgeFace_lee/test01/lena.jpg'
    fp32_model = '/home/jarvis/jarvis/Atec_EdgeFace_lee/test01/yunet_fp32.onnx'
    int8_model = '/home/jarvis/jarvis/Atec_EdgeFace_lee/test01/yunet_int8.onnx'
    out_img = '/home/jarvis/jarvis/Atec_EdgeFace_lee/test01/lena_detected.jpg'
    
    # OpenCV Face Detection
    faces_fp32 = run_opencv_yunet(fp32_model, img_path)
    faces_int8 = run_opencv_yunet(int8_model, img_path)
    
    # Draw visualizations
    draw_visualizations(img_path, faces_fp32, faces_int8, out_img)
    
    # Raw ONNX Runtime Inference
    outputs_fp32 = run_ort_raw(fp32_model, img_path)
    outputs_int8 = run_ort_raw(int8_model, img_path)
    
    # Calculate difference between FP32 and INT8 output tensors
    if outputs_fp32 and outputs_int8:
        print("\n=== Comparing Raw Tensors (FP32 vs INT8) ===")
        for name in outputs_fp32.keys():
            if name in outputs_int8:
                t1 = outputs_fp32[name]
                t2 = outputs_int8[name]
                if t1.shape == t2.shape:
                    mae = np.mean(np.abs(t1 - t2))
                    mse = np.mean((t1 - t2) ** 2)
                    max_diff = np.max(np.abs(t1 - t2))
                    cos_sim = np.dot(t1.flatten(), t2.flatten()) / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-10)
                    print(f"Tensor '{name}':")
                    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
                    print(f"  Mean Squared Error (MSE): {mse:.6f}")
                    print(f"  Max Absolute Difference: {max_diff:.6f}")
                    print(f"  Cosine Similarity: {cos_sim:.6f}")
