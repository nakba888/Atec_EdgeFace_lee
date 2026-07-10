import sys
import cv2
import numpy as np
import os

try:
    from dx_engine import InferenceEngine
    print("✅ dx_engine imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dx_engine: {e}")
    sys.exit(1)

print("=" * 60)
print("NPU Inference Test (YuNet + EdgeFace)")
print("=" * 60)


def test_yunet():
    """Test YuNet face detection"""
    print("\n" + "=" * 60)
    print("2. Testing YuNet Face Detection")
    print("=" * 60)

    model_path = "face_alignment/models/face_detection_yunet_2023mar.dxnn"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    # Load model
    print(f"\nLoading model: {model_path}")
    try:
        ie = InferenceEngine(model_path)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Find test image
    test_image_paths = [
        "/home/dxdemo/Atec_EdgeFace/npu_calibration/calibration_output/calibration_dataset/calib_0000.jpg",
        "npu_calibration/calibration_output/calibration_dataset/calib_0000.jpg",
        "test01/lena.jpg",
    ]

    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break

    if test_image_path is None:
        print("❌ No test image found")
        return False

    print(f"\nLoading test image: {test_image_path}")
    img = cv2.imread(test_image_path)
    print(f"Image shape: {img.shape}")

    # Preprocess (640x640, RGB, HWC, batch)
    resized = cv2.resize(img, (640, 640))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # chw_img = np.transpose(rgb_img, (2, 0, 1))
    input_tensor = np.expand_dims(rgb_img, axis=0).astype(np.uint8)

    # Make contiguous to avoid warning
    input_tensor = np.ascontiguousarray(input_tensor)

    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"Input tensor contiguous: {input_tensor.flags['C_CONTIGUOUS']}")

    # Run inference
    print("\n🚀 Running inference...")
    print("Calling ie.run()...")
    sys.stdout.flush()

    try:
        ie.run(input_tensor)
        print(f"✅ Inference completed!")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get outputs
    print("\n📥 Getting outputs...")
    try:
        outputs = ie.get_all_task_outputs()
        print(f"✅ Got outputs!")
    except Exception as e:
        print(f"❌ Failed to get outputs: {e}")
        import traceback
        traceback.print_exc()
        return False

# Print output info
    print(f"\nNumber of outputs: {len(outputs)}")
    print("\nOutput Details:")
    print("-" * 60)

    try:
        for i in range(len(outputs)):
            out = outputs[i]
            print(f"--- Output {i} ---")
            print(f"  Type: {type(out)}")

            # 1. If the output is a standard NumPy array
            if isinstance(out, np.ndarray):
                print(f"  Shape: {out.shape}")
                print(f"  Dtype: {out.dtype}")
                flat = out.flatten()

                # Special check for EdgeFace 512-d embedding
                if flat.size == 512:
                    l2_norm = np.linalg.norm(flat)
                    print(f"  L2 norm (Embedding?): {l2_norm:.4f}")

                if flat.size <= 20:
                    print(f"  Values: {flat}")
                else:
                    print(f"  Min/Max: [{flat.min():.4f}, {flat.max():.4f}]")

            # 2. If the output is a list (common from NPU)
            elif isinstance(out, list):
                print(f"  List Length: {len(out)}")
                if len(out) == 0:
                    print("  List is empty.")
                    continue

                print(f"  Type of first element: {type(out[0])}")

                # Check if this is a list of *one* tensor (common wrapper pattern)
                if len(out) == 1 and isinstance(out[0], np.ndarray):
                    out_tensor = out[0]
                    print(f"  Unwrapped Tensor Shape: {out_tensor.shape}")
                    print(f"  Unwrapped Tensor Dtype: {out_tensor.dtype}")
                    flat = out_tensor.flatten()

                    # Special check for EdgeFace 512-d embedding
                    if flat.size == 512:
                        l2_norm = np.linalg.norm(flat)
                        print(f"  L2 norm (Embedding?): {l2_norm:.4f}")

                    if flat.size <= 20:
                        print(f"  Unwrapped Values: {flat}")
                    else:
                        print(f"  Unwrapped Min/Max: [{flat.min():.4f}, {flat.max():.4f}]")
                else:
                    # It's a complex list (like YuNet's Output 0)
                    print_limit = min(len(out), 5)
                    print(f"  Inspecting first {print_limit} list elements:")
                    for j in range(print_limit):
                        list_element = out[j]
                        if isinstance(list_element, np.ndarray):
                            # Print shape, not the full array
                            print(f"    Element {j} [np.ndarray]: shape={list_element.shape}, dtype={list_element.dtype}")
                        else:
                            print(f"    Element {j} [{type(list_element)}]: {list_element}")

            # 3. For any other data type
            else:
                print(f"  Value: {out}")

    except Exception as e:
        print(f"Error printing output details: {e}")
        import traceback
        traceback.print_exc()
        return False
    print("\n✅ YuNet test completed")
    return True


def test_edgeface():
    """Test EdgeFace face recognition"""
    print("\n" + "=" * 60)
    print("1. Testing EdgeFace Face Recognition")
    print("=" * 60)

    model_path = "checkpoints/edgeface_xs_gamma_06.dxnn"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    # Load model
    print(f"\nLoading model: {model_path}")
    try:
        ie = InferenceEngine(model_path)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Find test image
    test_image_paths = [
        "/home/dxdemo/Atec_EdgeFace/npu_calibration/edgeface_calibration_output/aligned_faces/calib_0000.jpg",
        "npu_calibration/edgeface_calibration_output/aligned_faces/calib_0000.jpg",
        "npu_calibration/calibration_output/calibration_dataset/calib_0000.jpg",
        "test01/lena.jpg",
    ]

    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break

    if test_image_path is None:
        print("❌ No test image found")
        return False

    print(f"\nLoading test image: {test_image_path}")
    img = cv2.imread(test_image_path)
    print(f"Image shape: {img.shape}")

    # Preprocess (112x112, RGB, normalize, CHW, batch)
    resized = cv2.resize(img, (112, 112))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    float_img = rgb_img.astype(np.float32) / 255.0
    normalized = (float_img - 0.5) / 0.5
    chw_img = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(chw_img, axis=0).astype(np.float32)

    # Make contiguous to avoid warning
    input_tensor = np.ascontiguousarray(input_tensor)

    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"Input tensor contiguous: {input_tensor.flags['C_CONTIGUOUS']}")
    print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    # Run inference
    print("\n🚀 Running inference...")
    print("Calling ie.run()...")
    sys.stdout.flush()

    try:
        ie.run(input_tensor)
        print(f"✅ Inference completed!")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get outputs
    print("\n📥 Getting outputs...")
    try:
        outputs = ie.get_all_task_outputs()
        print(f"✅ Got outputs!")
    except Exception as e:
        print(f"❌ Failed to get outputs: {e}")
        import traceback
        traceback.print_exc()
        return False

# Print output info
    print(f"\nNumber of outputs: {len(outputs)}")
    print("\nOutput Details:")
    print("-" * 60)

    try:
        for i in range(len(outputs)):
            out = outputs[i]
            print(f"--- Output {i} ---")
            print(f"  Type: {type(out)}")

            # 1. If the output is a standard NumPy array
            if isinstance(out, np.ndarray):
                print(f"  Shape: {out.shape}")
                print(f"  Dtype: {out.dtype}")
                flat = out.flatten()

                # Special check for EdgeFace 512-d embedding
                if flat.size == 512:
                    l2_norm = np.linalg.norm(flat)
                    print(f"  L2 norm (Embedding?): {l2_norm:.4f}")

                if flat.size <= 20:
                    print(f"  Values: {flat}")
                else:
                    print(f"  Min/Max: [{flat.min():.4f}, {flat.max():.4f}]")

            # 2. If the output is a list (common from NPU)
            elif isinstance(out, list):
                print(f"  List Length: {len(out)}")
                if len(out) == 0:
                    print("  List is empty.")
                    continue

                print(f"  Type of first element: {type(out[0])}")

                # Check if this is a list of *one* tensor (common wrapper pattern)
                if len(out) == 1 and isinstance(out[0], np.ndarray):
                    out_tensor = out[0]
                    print(f"  Unwrapped Tensor Shape: {out_tensor.shape}")
                    print(f"  Unwrapped Tensor Dtype: {out_tensor.dtype}")
                    flat = out_tensor.flatten()

                    # Special check for EdgeFace 512-d embedding
                    if flat.size == 512:
                        l2_norm = np.linalg.norm(flat)
                        print(f"  L2 norm (Embedding?): {l2_norm:.4f}")

                    if flat.size <= 20:
                        print(f"  Unwrapped Values: {flat}")
                    else:
                        print(f"  Unwrapped Min/Max: [{flat.min():.4f}, {flat.max():.4f}]")
                else:
                    # It's a complex list (like YuNet's Output 0)
                    print_limit = min(len(out), 5)
                    print(f"  Inspecting first {print_limit} list elements:")
                    for j in range(print_limit):
                        list_element = out[j]
                        if isinstance(list_element, np.ndarray):
                            # Print shape, not the full array
                            print(f"    Element {j} [np.ndarray]: shape={list_element.shape}, dtype={list_element.dtype}")
                        else:
                            print(f"    Element {j} [{type(list_element)}]: {list_element}")

            # 3. For any other data type
            else:
                print(f"  Value: {out}")

    except Exception as e:
        print(f"Error printing output details: {e}")
        import traceback
        traceback.print_exc()
        return False
    print("\n✅ EdgeFace test completed")
    return True


def main():
    """Run all tests"""

    # Test EdgeFace first (known to work)
    edgeface_success = test_edgeface()

    # Test YuNet
    yunet_success = test_yunet()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"EdgeFace: {'✅ PASS' if edgeface_success else '❌ FAIL'}")
    print(f"YuNet:    {'✅ PASS' if yunet_success else '❌ FAIL'}")
    print("=" * 60)

    if yunet_success and edgeface_success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
