#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Face Recognition System with GUI for Jetson Orin Nano
Jetson Orin Nano용 실시간 얼굴 인식 시스템 (GUI 버전)

Features:
- Tkinter-based GUI interface
- TensorRT acceleration toggle
- Real-time performance monitoring (FPS, latency, temperature)
- PyTorch/TensorRT comparison
- Reference image management (Add/Remove)
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time
from typing import Optional

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

# Import Jetson system
from face_recognition_jetson_system import FaceRecognitionJetsonSystem, JETSON_AVAILABLE
from face_recognition_system import FaceAngleCalculator


class FaceRecognitionJetsonGUI:
    """GUI for Jetson Face Recognition System"""

    def __init__(self, root: tk.Tk):
        """
        Initialize GUI
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Face Recognition System (Jetson Orin Nano)")
        self.root.geometry("1400x850")

        # System variables
        self.system: Optional[FaceRecognitionJetsonSystem] = None
        self.camera_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame = None
        self.current_landmarks = None
        self.processing_frame = False
        self.last_process_time = 0

        # Multi-angle capture state
        self.capture_mode = False
        self.capture_person_id = None
        self.required_angles = ['front', 'left', 'right', 'up', 'down']
        self.captured_angles = set()
        self.angle_slots = {}

        # Configuration
        self.detector_var = tk.StringVar(value='yunet')
        self.device_var = tk.StringVar(value='jetson')
        self.use_tensorrt_var = tk.BooleanVar(value=True)
        self.use_fp16_var = tk.BooleanVar(value=True)
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.camera_id_var = tk.IntVar(value=0)

        # Model paths
        self.model_path_trt = 'checkpoints/edgeface_xs_gamma_06.trt'
        self.model_path_pt = 'checkpoints/edgeface_xs_gamma_06.pt'
        self.model_name = 'edgeface_xs_gamma_06'

        # Performance tracking
        self.perf_stats = {
            'fps': 0.0,
            'detection_ms': 0.0,
            'recognition_ms': 0.0,
            'temperature_c': 0.0
        }

        # Build UI
        self.build_ui()

        # Initialize system
        self.initialize_system()

    def build_ui(self):
        """Build user interface"""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ===== Left Panel: Controls =====
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        row = 0

        # TensorRT Section
        ttk.Label(control_frame, text="Acceleration:", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # TensorRT Checkbox
        self.tensorrt_checkbox = ttk.Checkbutton(
            control_frame, text="🚀 Use TensorRT",
            variable=self.use_tensorrt_var,
            command=self.on_tensorrt_changed)
        self.tensorrt_checkbox.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        # FP16 Checkbox
        self.fp16_checkbox = ttk.Checkbutton(
            control_frame, text="⚡ FP16 Precision",
            variable=self.use_fp16_var)
        self.fp16_checkbox.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        # TensorRT status label
        self.trt_status_label = ttk.Label(
            control_frame, 
            text="TensorRT: Available" if JETSON_AVAILABLE else "TensorRT: Not Available",
            foreground='green' if JETSON_AVAILABLE else 'red')
        self.trt_status_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Detector Selection
        ttk.Label(control_frame, text="Face Detector:").grid(row=row, column=0, sticky=tk.W, pady=5)
        detector_combo = ttk.Combobox(control_frame, textvariable=self.detector_var, state='readonly', width=15)
        detector_combo['values'] = ('yunet', 'mtcnn', 'yolov5_face', 'yolov8')
        detector_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        detector_combo.bind('<<ComboboxSelected>>', self.on_detector_changed)
        row += 1

        # Similarity Threshold
        ttk.Label(control_frame, text="Threshold:").grid(row=row, column=0, sticky=tk.W, pady=5)
        threshold_spinbox = ttk.Spinbox(control_frame, from_=0.0, to=1.0, increment=0.05,
                                       textvariable=self.threshold_var, width=15)
        threshold_spinbox.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Camera ID
        ttk.Label(control_frame, text="Camera ID:").grid(row=row, column=0, sticky=tk.W, pady=5)
        camera_spinbox = ttk.Spinbox(control_frame, from_=0, to=5, increment=1,
                                     textvariable=self.camera_id_var, width=15)
        camera_spinbox.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Camera Control
        ttk.Label(control_frame, text="Camera Control:", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        self.start_btn = ttk.Button(control_frame, text="▶ Start Camera", command=self.start_camera)
        self.start_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.stop_btn = ttk.Button(control_frame, text="⬛ Stop Camera", command=self.stop_camera, state='disabled')
        self.stop_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Benchmark button
        self.benchmark_btn = ttk.Button(control_frame, text="📊 Run Benchmark", command=self.run_benchmark)
        self.benchmark_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Reference Management
        ttk.Label(control_frame, text="Reference Management:", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        self.capture_btn = ttk.Button(control_frame, text="📸 Capture Multi-Angle",
                                       command=self.start_multi_angle_capture, state='disabled')
        self.capture_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.cancel_capture_btn = ttk.Button(control_frame, text="❌ Cancel Capture",
                                              command=self.cancel_capture, state='disabled')
        self.cancel_capture_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        ttk.Button(control_frame, text="➕ Add from File", command=self.add_reference).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        ttk.Button(control_frame, text="➖ Remove Person", command=self.remove_reference).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Reference List
        ttk.Label(control_frame, text="Registered Persons:", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1

        self.ref_listbox = tk.Listbox(control_frame, height=8, width=25)
        self.ref_listbox.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.ref_listbox.bind('<<ListboxSelect>>', self.on_person_selected)
        row += 1

        # Angles label
        self.angles_label = ttk.Label(control_frame, text="Captured angles: -", font=('', 9))
        self.angles_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # Configure control frame
        control_frame.columnconfigure(1, weight=1)

        # ===== Right Panel: Video Display =====
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        # Video Frame
        video_frame = ttk.LabelFrame(right_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ===== Performance Panel =====
        perf_frame = ttk.LabelFrame(right_frame, text="Performance Monitor", padding="10")
        perf_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # Performance labels with grid layout
        perf_labels_frame = ttk.Frame(perf_frame)
        perf_labels_frame.pack(fill=tk.X)

        # Mode indicator
        self.mode_label = ttk.Label(perf_labels_frame, text="Mode: TensorRT", font=('', 12, 'bold'),
                                   foreground='cyan')
        self.mode_label.grid(row=0, column=0, padx=20, pady=5)

        # FPS
        self.fps_label = ttk.Label(perf_labels_frame, text="FPS: 0.0", font=('', 12, 'bold'),
                                  foreground='green')
        self.fps_label.grid(row=0, column=1, padx=20, pady=5)

        # Detection time
        self.detect_label = ttk.Label(perf_labels_frame, text="Detect: 0.0ms", font=('', 11))
        self.detect_label.grid(row=0, column=2, padx=20, pady=5)

        # Recognition time
        self.recog_label = ttk.Label(perf_labels_frame, text="Recog: 0.0ms", font=('', 11))
        self.recog_label.grid(row=0, column=3, padx=20, pady=5)

        # Temperature
        self.temp_label = ttk.Label(perf_labels_frame, text="Temp: --°C", font=('', 11))
        self.temp_label.grid(row=0, column=4, padx=20, pady=5)

        # ===== Status Panel =====
        status_frame = ttk.LabelFrame(right_frame, text="Status Log", padding="5")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.status_text = tk.Text(status_frame, height=6, wrap=tk.WORD, state='disabled')
        self.status_text.pack(fill=tk.BOTH, expand=True)

    def log_status(self, message: str):
        """Log message to status panel"""
        self.status_text.config(state='normal')
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')

    def initialize_system(self):
        """Initialize face recognition system"""
        try:
            self.log_status("🚀 Initializing Jetson Face Recognition System...")

            # Determine model path based on TensorRT setting
            use_tensorrt = self.use_tensorrt_var.get()
            detector = self.detector_var.get()

            if use_tensorrt and JETSON_AVAILABLE:
                model_path = self.model_path_trt
                if not os.path.exists(model_path):
                    self.log_status(f"⚠️ TensorRT engine not found: {model_path}")
                    self.log_status(f"💡 Falling back to PyTorch model")
                    model_path = self.model_path_pt
                    use_tensorrt = False
                else:
                    self.log_status(f"📌 Using TensorRT engine: {model_path}")
            else:
                model_path = self.model_path_pt
                use_tensorrt = False
                self.log_status(f"📌 Using PyTorch model: {model_path}")

            self.system = FaceRecognitionJetsonSystem(
                detector_method=detector,
                edgeface_model_path=model_path,
                edgeface_model_name=self.model_name,
                device='jetson' if use_tensorrt else 'cuda',
                similarity_threshold=self.threshold_var.get(),
                use_tensorrt=use_tensorrt
            )

            # Update mode label
            self.update_mode_label()

            self.log_status("✅ System initialized successfully")
            self.update_reference_list()

        except Exception as e:
            self.log_status(f"❌ Initialization failed: {e}")
            messagebox.showerror("Error", f"Failed to initialize system: {e}")

    def update_mode_label(self):
        """Update mode indicator label"""
        if self.system and self.system.use_tensorrt:
            self.mode_label.config(text="Mode: TensorRT 🚀", foreground='#00FFFF')
        else:
            self.mode_label.config(text="Mode: PyTorch", foreground='orange')

    def on_tensorrt_changed(self):
        """Handle TensorRT checkbox change"""
        if self.camera_running:
            messagebox.showwarning("Warning", "Stop camera before changing TensorRT setting")
            self.use_tensorrt_var.set(not self.use_tensorrt_var.get())
            return

        tensorrt_enabled = self.use_tensorrt_var.get()
        self.log_status(f"🔄 TensorRT: {'Enabled' if tensorrt_enabled else 'Disabled'}")
        self.initialize_system()

    def on_detector_changed(self, event=None):
        """Handle detector change"""
        if self.camera_running:
            messagebox.showwarning("Warning", "Stop camera before changing detector")
            return
        self.log_status(f"🔄 Changing detector to: {self.detector_var.get()}")
        self.initialize_system()

    def start_camera(self):
        """Start camera feed"""
        if self.camera_running:
            return

        camera_id = self.camera_id_var.get()
        self.log_status(f"📹 Opening camera {camera_id}...")

        try:
            # Try V4L2 backend first (for Linux/Jetson)
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(camera_id)

            if self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Test frame capture
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.log_status(f"✅ Camera opened: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    self.cap.release()
                    self.cap = None
            else:
                self.cap = None

        except Exception as e:
            self.log_status(f"❌ Error opening camera: {e}")
            if self.cap:
                self.cap.release()
            self.cap = None

        if self.cap is None or not self.cap.isOpened():
            self.log_status(f"❌ Cannot open camera {camera_id}")
            messagebox.showerror("Error", f"Cannot open camera {camera_id}")
            return

        self.camera_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.capture_btn.config(state='normal')

        # Update threshold
        if self.system:
            self.system.similarity_threshold = self.threshold_var.get()

        self.log_status("✅ Camera started")

        # Start video thread
        self.video_thread = threading.Thread(target=self.update_video, daemon=True)
        self.video_thread.start()

        # Start performance update thread
        self.perf_thread = threading.Thread(target=self.update_performance, daemon=True)
        self.perf_thread.start()

    def stop_camera(self):
        """Stop camera feed"""
        if not self.camera_running:
            return

        self.camera_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')

        if self.cap:
            self.cap.release()

        self.log_status("⬛ Camera stopped")
        self.video_label.config(image='')

    def update_video(self):
        """Update video feed (runs in separate thread)"""
        consecutive_failures = 0
        max_failures = 30

        while self.camera_running:
            if self.cap and self.cap.isOpened():
                if not self.cap.grab():
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self.log_status("❌ Camera disconnected")
                        self.camera_running = False
                        break
                    time.sleep(0.01)
                    continue

                ret, frame = self.cap.retrieve()

                if ret and frame is not None:
                    consecutive_failures = 0
                    self.current_frame = frame.copy()

                    if not self.processing_frame:
                        self.processing_frame = True
                        self.root.after(0, self.process_and_display_frame, frame)
                else:
                    consecutive_failures += 1
            else:
                break

    def update_performance(self):
        """Update performance labels (runs in separate thread)"""
        while self.camera_running:
            if self.system:
                stats = self.system.perf_monitor.get_stats()
                
                # Update labels in main thread
                self.root.after(0, self._update_perf_labels, stats)
            
            time.sleep(0.5)  # Update every 500ms

    def _update_perf_labels(self, stats):
        """Update performance labels (called in main thread)"""
        self.fps_label.config(text=f"FPS: {stats['fps']:.1f}")
        self.detect_label.config(text=f"Detect: {stats['detection_ms']:.1f}ms")
        self.recog_label.config(text=f"Recog: {stats['recognition_ms']:.1f}ms")
        
        if stats['temperature_c'] > 0:
            temp = stats['temperature_c']
            color = 'green' if temp < 60 else 'orange' if temp < 75 else 'red'
            self.temp_label.config(text=f"Temp: {temp:.0f}°C", foreground=color)

    def process_and_display_frame(self, frame: np.ndarray):
        """Process and display frame in main thread"""
        try:
            # Process frame
            annotated_frame, detections = self.system.process_frame(frame)

            # Handle multi-angle capture mode
            if self.capture_mode and len(detections) > 0:
                det = detections[0]
                if det['landmarks'] is not None:
                    self.current_landmarks = det['landmarks']
                    self.auto_capture_angle(det['landmarks'])

            # Draw angle capture overlay if in capture mode
            if self.capture_mode:
                annotated_frame = self.draw_angle_overlay(annotated_frame)

            # Convert to RGB for display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Resize to fit display
            h, w = display_frame.shape[:2]
            max_w, max_h = 800, 600
            if w > max_w or h > max_h:
                scale = min(max_w/w, max_h/h)
                new_w, new_h = int(w*scale), int(h*scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h))

            # Convert to PhotoImage
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        except Exception as e:
            self.log_status(f"⚠️ Error processing frame: {e}")
        finally:
            self.processing_frame = False

    def draw_angle_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw angle capture progress overlay on frame"""
        h, w = frame.shape[:2]

        # Create overlay panel
        overlay = frame.copy()
        panel_width = 200
        panel_height = 250
        panel_x = 10
        panel_y = 150  # Below performance info

        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Title
        cv2.putText(frame, "Capture Progress", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw angle indicators
        y_offset = panel_y + 50
        for i, angle in enumerate(self.required_angles):
            y = y_offset + i * 35

            if angle in self.captured_angles:
                symbol = "✓"
                color = (0, 255, 0)
            else:
                symbol = "○"
                color = (150, 150, 150)

            text = f"{angle.capitalize()}: {symbol}"
            cv2.putText(frame, text, (panel_x + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Progress bar
        progress = len(self.captured_angles) / len(self.required_angles)
        bar_width = panel_width - 20
        bar_height = 15
        bar_x = panel_x + 10
        bar_y = panel_y + panel_height - 30

        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)

        progress_width = int(bar_width * progress)
        if progress_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         (0, 255, 0), -1)

        return frame

    def run_benchmark(self):
        """Run performance benchmark"""
        if self.system is None:
            messagebox.showerror("Error", "System not initialized")
            return

        self.log_status("📊 Running benchmark...")

        # Run benchmark in separate thread
        def benchmark_thread():
            results = self.system.run_benchmark(100)
            
            # Show results
            self.root.after(0, lambda: self.show_benchmark_results(results))

        thread = threading.Thread(target=benchmark_thread, daemon=True)
        thread.start()

    def show_benchmark_results(self, results):
        """Show benchmark results in dialog"""
        msg = f"""
Benchmark Results ({results['mode']})
{'='*40}
Iterations: {results['iterations']}

Latency:
  Average: {results['avg_ms']:.2f} ± {results['std_ms']:.2f} ms
  Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms
  P50/P95/P99: {results['p50_ms']:.2f} / {results['p95_ms']:.2f} / {results['p99_ms']:.2f} ms

Throughput: {results['throughput_fps']:.1f} FPS
"""
        messagebox.showinfo("Benchmark Results", msg)
        self.log_status(f"📊 Benchmark complete: {results['avg_ms']:.2f}ms avg, {results['throughput_fps']:.1f} FPS")

    def start_multi_angle_capture(self):
        """Start multi-angle capture mode"""
        if not self.camera_running or self.current_frame is None:
            messagebox.showwarning("Warning", "Camera must be running to capture")
            return

        person_id = simpledialog.askstring("Person ID", "Enter person name/ID:")

        if not person_id:
            return

        self.capture_mode = True
        self.capture_person_id = person_id
        self.captured_angles = set()
        self.angle_slots = {}

        self.capture_btn.config(state='disabled')
        self.cancel_capture_btn.config(state='normal')

        self.log_status(f"🎯 Multi-angle capture started for {person_id}")
        self.log_status(f"👉 Rotate face: Front, Left, Right, Up, Down")

    def cancel_capture(self):
        """Cancel multi-angle capture mode"""
        self.capture_mode = False
        self.capture_person_id = None
        self.captured_angles = set()
        self.angle_slots = {}

        self.capture_btn.config(state='normal')
        self.cancel_capture_btn.config(state='disabled')

        self.log_status("❌ Multi-angle capture cancelled")

    def auto_capture_angle(self, landmarks: np.ndarray):
        """Auto-capture when face is at specific angle"""
        yaw, pitch, roll = FaceAngleCalculator.calculate_head_pose(landmarks)
        angle_category = FaceAngleCalculator.get_angle_category(yaw, pitch)

        if angle_category in self.required_angles and angle_category not in self.captured_angles:
            try:
                success = self.system.add_reference_from_frame(
                    self.current_frame, 
                    self.capture_person_id, 
                    angle_category
                )

                if success:
                    self.captured_angles.add(angle_category)
                    self.log_status(f"✅ Captured {angle_category} angle (Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°)")
                    self.update_reference_list()

                    if len(self.captured_angles) == len(self.required_angles):
                        self.log_status(f"🎉 All angles captured for {self.capture_person_id}!")
                        messagebox.showinfo("Success", f"All angles captured for {self.capture_person_id}!")
                        self.cancel_capture()

            except Exception as e:
                self.log_status(f"❌ Error capturing {angle_category}: {e}")

    def add_reference(self):
        """Add reference person from image file"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if not file_path:
            return

        person_id = simpledialog.askstring("Person ID", "Enter person name/ID:")

        if not person_id:
            return

        self.log_status(f"➕ Adding reference: {person_id}")

        success = self.system.add_reference_from_image(file_path, person_id)

        if success:
            self.log_status(f"✅ Successfully added {person_id}")
            self.update_reference_list()
            messagebox.showinfo("Success", f"Added {person_id}")
        else:
            self.log_status(f"❌ Failed to add {person_id}")
            messagebox.showerror("Error", f"Failed to add {person_id}")

    def remove_reference(self):
        """Remove reference person"""
        selection = self.ref_listbox.curselection()

        if not selection:
            messagebox.showwarning("Warning", "Please select a person to remove")
            return

        person_id = self.ref_listbox.get(selection[0])

        if messagebox.askyesno("Confirm", f"Remove {person_id} from database?"):
            self.system.ref_db.remove_person(person_id)
            self.log_status(f"➖ Removed {person_id}")
            self.update_reference_list()

    def on_person_selected(self, event=None):
        """Handle person selection in listbox"""
        selection = self.ref_listbox.curselection()
        if not selection:
            self.angles_label.config(text="Captured angles: -")
            return

        person_id = self.ref_listbox.get(selection[0])
        angles = self.system.ref_db.get_person_angles(person_id)

        if angles:
            angle_text = ", ".join(sorted(angles))
            self.angles_label.config(text=f"Captured angles: {angle_text}")
        else:
            self.angles_label.config(text="Captured angles: None")

    def update_reference_list(self):
        """Update reference list display"""
        self.ref_listbox.delete(0, tk.END)
        if self.system:
            persons = self.system.ref_db.get_all_persons()
            for person in sorted(persons):
                self.ref_listbox.insert(tk.END, person)
        self.on_person_selected()

    def on_closing(self):
        """Handle window close"""
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()

    # Set style
    style = ttk.Style()
    style.theme_use('clam')

    app = FaceRecognitionJetsonGUI(root)

    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()


if __name__ == '__main__':
    main()
