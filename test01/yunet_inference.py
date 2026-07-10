import cv2
import sys
import os

def detect_face(model_path, image_path, output_path="result.jpg"):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image at '{image_path}'")
        return False

    h, w, _ = img.shape
    print(f"Loaded image: {image_path} ({w}x{h})")

    # 2. Create FaceDetectorYN
    try:
        detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(w, h),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000
        )
    except Exception as e:
        print(f"Error initializing detector with '{model_path}': {e}")
        return False

    # 3. Detect
    retval, faces = detector.detect(img)

    # 4. Process results
    if retval and faces is not None:
        print(f"\nSuccessfully detected {len(faces)} face(s):")
        for i, face in enumerate(faces):
            bbox = face[0:4].astype(int)
            landmarks = face[4:14].reshape(5, 2).astype(int)
            score = face[14]
            print(f"  [Face {i+1}]")
            print(f"    Bounding Box (x, y, w, h): {bbox.tolist()}")
            print(f"    Landmarks: {landmarks.tolist()}")
            print(f"    Confidence: {score:.6f}")

            # Draw bbox (Green) and landmarks (Red)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            for pt in landmarks:
                cv2.circle(img, tuple(pt), 3, (0, 0, 255), -1)
            cv2.putText(img, f"Conf: {score:.4f}", (bbox[0], bbox[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imwrite(output_path, img)
        print(f"\nSaved visualization to: {output_path}")
        return True
    else:
        print("No face detected.")
        return False

if __name__ == '__main__':
    # Default paths
    default_image = "lena.jpg"
    default_model = "yunet_fp32.onnx"

    # Usage: python yunet_inference.py [model_path] [image_path]
    model_arg = sys.argv[1] if len(sys.argv) > 1 else default_model
    img_arg = sys.argv[2] if len(sys.argv) > 2 else default_image

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    img_path = img_arg if os.path.isabs(img_arg) else os.path.join(script_dir, img_arg)
    model_path = model_arg if os.path.isabs(model_arg) else os.path.join(script_dir, model_arg)

    out_name = f"result_{os.path.basename(model_path).replace('.onnx', '')}_{os.path.basename(img_path)}"
    out_path = os.path.join(script_dir, out_name)

    detect_face(model_path, img_path, out_path)
