#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX to TensorRT Conversion Script for Jetson Orin Nano
Jetson Orin Nano용 ONNX → TensorRT 변환 스크립트

This script converts ONNX models to TensorRT engines optimized for Jetson.
Supports FP16 and INT8 precision modes.

Usage:
    python onnx_to_tensorrt.py --input model.onnx --output model.trt --fp16
    
Requirements:
    - TensorRT (included in JetPack)
    - pycuda (optional, for INT8 calibration)
"""

import os
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_VERSION = trt.__version__
except ImportError:
    TRT_AVAILABLE = False
    TRT_VERSION = None
    logger.warning("TensorRT not available. Please install TensorRT (included in JetPack for Jetson).")


class TensorRTConverter:
    """Convert ONNX models to TensorRT engines"""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize TensorRT converter
        
        Args:
            verbose: Enable verbose logging
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install JetPack on Jetson.")
        
        self.verbose = verbose
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        
    def convert(self, 
                onnx_path: str, 
                engine_path: str,
                fp16: bool = True,
                int8: bool = False,
                max_batch_size: int = 1,
                max_workspace_size: int = 1 << 30,  # 1GB
                calibration_data_dir: str = None,
                dynamic_shapes: bool = False) -> bool:
        """
        Convert ONNX model to TensorRT engine
        
        Args:
            onnx_path: Path to input ONNX model
            engine_path: Path to save TensorRT engine
            fp16: Enable FP16 precision (default: True for Jetson)
            int8: Enable INT8 precision (requires calibration)
            max_batch_size: Maximum batch size
            max_workspace_size: Maximum workspace size in bytes
            calibration_data_dir: Directory with calibration images (for INT8)
            dynamic_shapes: Enable dynamic input shapes
            
        Returns:
            True if conversion successful, False otherwise
        """
        if not os.path.exists(onnx_path):
            logger.error(f"ONNX file not found: {onnx_path}")
            return False
        
        logger.info(f"Converting ONNX model: {onnx_path}")
        logger.info(f"TensorRT version: {TRT_VERSION}")
        logger.info(f"FP16: {fp16}, INT8: {int8}")
        
        try:
            # Create builder and network
            builder = trt.Builder(self.logger)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            parser = trt.OnnxParser(network, self.logger)
            
            # Parse ONNX model
            logger.info("Parsing ONNX model...")
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"ONNX Parse Error: {parser.get_error(i)}")
                    return False
            
            logger.info(f"Network inputs: {network.num_inputs}")
            logger.info(f"Network outputs: {network.num_outputs}")
            
            # Print input/output info
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                logger.info(f"  Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            
            for i in range(network.num_outputs):
                output_tensor = network.get_output(i)
                logger.info(f"  Output {i}: {output_tensor.name}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
            
            # Create builder config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
            
            # Set precision flags
            if fp16 and builder.platform_has_fast_fp16:
                logger.info("Enabling FP16 precision")
                config.set_flag(trt.BuilderFlag.FP16)
            elif fp16:
                logger.warning("FP16 requested but not supported on this platform")
            
            if int8 and builder.platform_has_fast_int8:
                logger.info("Enabling INT8 precision")
                config.set_flag(trt.BuilderFlag.INT8)
                
                # INT8 calibration
                if calibration_data_dir:
                    calibrator = self._create_int8_calibrator(
                        network, calibration_data_dir, max_batch_size
                    )
                    config.int8_calibrator = calibrator
                else:
                    logger.warning("INT8 enabled but no calibration data provided. Using FP16 fallback.")
            elif int8:
                logger.warning("INT8 requested but not supported on this platform")
            
            # Set optimization profile for dynamic shapes
            # Automatically detect dynamic shapes if not explicitly requested
            is_dynamic = dynamic_shapes
            if not is_dynamic:
                for i in range(network.num_inputs):
                    if network.get_input(i).shape[0] == -1:
                        is_dynamic = True
                        logger.info(f"Dynamic shape detected on input {i}, enabling optimization profile automatically.")
                        break

            if is_dynamic:
                profile = builder.create_optimization_profile()
                for i in range(network.num_inputs):
                    input_tensor = network.get_input(i)
                    input_name = input_tensor.name
                    
                    # Get the shape, replacing dynamic -1 with reasonable defaults
                    shape = list(input_tensor.shape)
                    
                    # EdgeFace input: (batch, 3, 112, 112)
                    min_shape = [1 if s == -1 else s for s in shape]
                    opt_shape = [1 if s == -1 else s for s in shape]
                    max_shape = [max_batch_size if s == -1 else s for s in shape]
                    
                    profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
                    logger.info(f"Profile for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                
                config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("Building TensorRT engine... (this may take a few minutes)")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            logger.info(f"Saving TensorRT engine to: {engine_path}")
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            # Verify file size
            engine_size = os.path.getsize(engine_path)
            logger.info(f"Engine saved successfully! Size: {engine_size / (1024*1024):.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_int8_calibrator(self, network, calibration_data_dir: str, batch_size: int):
        """
        Create INT8 calibrator for quantization
        
        Args:
            network: TensorRT network
            calibration_data_dir: Directory containing calibration images
            batch_size: Batch size for calibration
            
        Returns:
            Calibrator object
        """
        try:
            import numpy as np
            import cv2
            from glob import glob
            
            class EdgeFaceCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, data_dir, batch_size, input_shape=(3, 112, 112)):
                    super().__init__()
                    self.batch_size = batch_size
                    self.input_shape = input_shape
                    self.cache_file = "edgeface_calibration.cache"
                    
                    # Load calibration images
                    image_paths = glob(os.path.join(data_dir, "*.jpg")) + \
                                  glob(os.path.join(data_dir, "*.png"))
                    self.image_paths = image_paths[:500]  # Use max 500 images
                    self.current_idx = 0
                    
                    # Allocate device memory
                    import pycuda.driver as cuda
                    import pycuda.autoinit
                    
                    self.device_input = cuda.mem_alloc(
                        batch_size * np.prod(input_shape) * np.float32().nbytes
                    )
                    
                    logger.info(f"INT8 Calibrator: {len(self.image_paths)} calibration images")
                
                def get_batch_size(self):
                    return self.batch_size
                
                def get_batch(self, names):
                    if self.current_idx >= len(self.image_paths):
                        return None
                    
                    batch_data = []
                    for i in range(self.batch_size):
                        if self.current_idx >= len(self.image_paths):
                            break
                        
                        img_path = self.image_paths[self.current_idx]
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (112, 112))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32) / 255.0
                        img = (img - 0.5) / 0.5
                        img = np.transpose(img, (2, 0, 1))
                        batch_data.append(img)
                        self.current_idx += 1
                    
                    if not batch_data:
                        return None
                    
                    batch_array = np.ascontiguousarray(np.array(batch_data))
                    
                    import pycuda.driver as cuda
                    cuda.memcpy_htod(self.device_input, batch_array)
                    
                    return [int(self.device_input)]
                
                def read_calibration_cache(self):
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, 'rb') as f:
                            return f.read()
                    return None
                
                def write_calibration_cache(self, cache):
                    with open(self.cache_file, 'wb') as f:
                        f.write(cache)
            
            return EdgeFaceCalibrator(calibration_data_dir, batch_size)
            
        except ImportError as e:
            logger.warning(f"pycuda not available for INT8 calibration: {e}")
            return None


def verify_engine(engine_path: str) -> bool:
    """
    Verify TensorRT engine can be loaded
    
    Args:
        engine_path: Path to TensorRT engine
        
    Returns:
        True if engine is valid
    """
    if not TRT_AVAILABLE:
        logger.error("TensorRT not available")
        return False
    
    try:
        # User standard python logger for messages, and trt.Logger for TRT internal
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            logger.error("Failed to load engine")
            return False
        
        logger.info(f"Engine loaded successfully!")
        
        # In TRT 10+, use get_tensor_name
        num_io = engine.num_io_tensors
        logger.info(f"  Num bindings: {num_io}")
        
        for i in range(num_io):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            logger.info(f"  Binding {i}: {name}, shape: {shape}, dtype: {dtype}, mode: {mode}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify engine: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to TensorRT engine for Jetson Orin Nano',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic FP16 conversion (recommended for Jetson)
  python onnx_to_tensorrt.py --input model.onnx --output model.trt --fp16
  
  # INT8 conversion with calibration data
  python onnx_to_tensorrt.py --input model.onnx --output model.trt --int8 --calibration-dir ./calibration_images
  
  # Verify existing engine
  python onnx_to_tensorrt.py --verify model.trt
        """
    )
    
    parser.add_argument('--input', '-i', type=str, help='Input ONNX model path')
    parser.add_argument('--output', '-o', type=str, help='Output TensorRT engine path')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable FP16 precision (default: True)')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 precision')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 precision (requires calibration)')
    parser.add_argument('--calibration-dir', type=str, help='Directory with calibration images for INT8')
    parser.add_argument('--max-batch-size', type=int, default=1, help='Maximum batch size')
    parser.add_argument('--workspace', type=int, default=1024, help='Max workspace size in MB')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic input shapes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--verify', type=str, help='Verify existing TensorRT engine')
    
    args = parser.parse_args()
    
    # Check TensorRT availability
    if not TRT_AVAILABLE:
        logger.error("TensorRT is not installed. Please install JetPack on Jetson Orin Nano.")
        logger.info("Installation guide: https://developer.nvidia.com/embedded/jetpack")
        return 1
    
    logger.info(f"TensorRT version: {TRT_VERSION}")
    
    # Verify mode
    if args.verify:
        logger.info(f"Verifying TensorRT engine: {args.verify}")
        if verify_engine(args.verify):
            logger.info("✅ Engine verification passed!")
            return 0
        else:
            logger.error("❌ Engine verification failed!")
            return 1
    
    # Conversion mode
    if not args.input:
        parser.error("--input is required for conversion")
    
    if not args.output:
        # Generate output path
        args.output = args.input.replace('.onnx', '.trt')
        logger.info(f"Output path not specified, using: {args.output}")
    
    # Handle FP16 flag
    fp16 = args.fp16 and not args.no_fp16
    
    # Convert
    converter = TensorRTConverter(verbose=args.verbose)
    success = converter.convert(
        onnx_path=args.input,
        engine_path=args.output,
        fp16=fp16,
        int8=args.int8,
        max_batch_size=args.max_batch_size,
        max_workspace_size=args.workspace * (1 << 20),  # Convert MB to bytes
        calibration_data_dir=args.calibration_dir,
        dynamic_shapes=args.dynamic
    )
    
    if success:
        logger.info("✅ Conversion completed successfully!")
        
        # Verify the generated engine
        logger.info("Verifying generated engine...")
        if verify_engine(args.output):
            logger.info("✅ Engine verification passed!")
        else:
            logger.warning("⚠️ Engine verification had issues, but file was created")
        
        return 0
    else:
        logger.error("❌ Conversion failed!")
        return 1


if __name__ == '__main__':
    exit(main())
