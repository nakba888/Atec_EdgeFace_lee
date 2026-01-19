#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Face Recognition System using EdgeFace
실시간 얼굴 인식 시스템

Features:
- Real-time face detection and recognition
- Multiple detector support (MTCNN, YuNet, YOLO, etc.)
- Reference image management
- Display: FPS, Person ID, Similarity score
"""

import os
import sys
import cv2
import numpy as np
import torch
import pickle
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

from backbones import get_model
from face_alignment.unified_detector import UnifiedFaceDetector

# Import NPU recognizer
try:
    from edgeface_npu_recognizer import EdgeFaceNPURecognizer
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    EdgeFaceNPURecognizer = None


class FaceAngleCalculator:
    """Calculate face pose angles from landmarks"""

    @staticmethod
    def calculate_head_pose(landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate head pose angles (yaw, pitch, roll) from facial landmarks

        Args:
            landmarks: Facial landmarks array [x1, y1, x2, y2, ..., x5, y5]
                      (left_eye, right_eye, nose, left_mouth, right_mouth)

        Returns:
            (yaw, pitch, roll) in degrees
            yaw: left(-) / right(+) rotation
            pitch: up(-) / down(+) rotation
            roll: tilt left(-) / right(+)
        """
        if len(landmarks) < 10:
            return 0.0, 0.0, 0.0

        # Extract landmark points (5-point landmarks from most detectors)
        left_eye = np.array([landmarks[0], landmarks[1]])
        right_eye = np.array([landmarks[2], landmarks[3]])
        nose = np.array([landmarks[4], landmarks[5]])
        left_mouth = np.array([landmarks[6], landmarks[7]])
        right_mouth = np.array([landmarks[8], landmarks[9]])

        # Calculate roll (head tilt) from eye positions
        eye_center = (left_eye + right_eye) / 2
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        roll = math.degrees(math.atan2(dY, dX))

        # Calculate yaw (left-right rotation) from eye-nose distances
        left_eye_nose_dist = np.linalg.norm(left_eye - nose)
        right_eye_nose_dist = np.linalg.norm(right_eye - nose)
        eye_distance = np.linalg.norm(right_eye - left_eye)

        # Asymmetry ratio indicates yaw
        if eye_distance > 0:
            asymmetry = (right_eye_nose_dist - left_eye_nose_dist) / eye_distance
            yaw = asymmetry * 90  # Scale to approximate degrees
        else:
            yaw = 0.0

        # Calculate pitch (up-down rotation) from eye-mouth vertical distance
        mouth_center = (left_mouth + right_mouth) / 2
        vertical_dist = mouth_center[1] - eye_center[1]
        expected_vertical = eye_distance * 1.5  # Typical face proportions

        if expected_vertical > 0:
            vertical_ratio = (vertical_dist - expected_vertical) / expected_vertical
            pitch = vertical_ratio * 30  # Scale to approximate degrees
        else:
            pitch = 0.0

        return yaw, pitch, roll

    @staticmethod
    def get_angle_category(yaw: float, pitch: float) -> str:
        """
        Categorize face angle into discrete bins

        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees

        Returns:
            Category string: 'front', 'left', 'right', 'up', 'down'
        """
        yaw_threshold = 20
        pitch_threshold = 15

        # Check pitch first (up/down takes priority)
        if pitch < -pitch_threshold:
            return 'up'
        elif pitch > pitch_threshold:
            return 'down'

        # Then check yaw (left/right)
        if yaw < -yaw_threshold:
            return 'left'
        elif yaw > yaw_threshold:
            return 'right'

        return 'front'


class EdgeFaceRecognizer:
    """EdgeFace based face recognition module"""

    def __init__(self, model_path: str, model_name: str = 'edgeface_xs_gamma_06', device: str = 'cuda'):
        """
        Initialize EdgeFace recognizer

        Args:
            model_path: Path to EdgeFace model checkpoint
            model_name: Model architecture name
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load EdgeFace model
        self.model = get_model(model_name, fp16=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ EdgeFace model loaded: {model_name} on {self.device}")

    @torch.no_grad()
    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from aligned face image

        Args:
            face_img: Aligned face image (112x112x3)

        Returns:
            Face embedding vector (512-d)
        """
        # Preprocess
        if face_img.shape[:2] != (112, 112):
            face_img = cv2.resize(face_img, (112, 112))

        # Convert BGR to RGB
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Transpose to CHW
        img = np.transpose(img, (2, 0, 1))

        # Convert to tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        # Normalize
        img_tensor.div_(255).sub_(0.5).div_(0.5)

        # Extract embedding
        embedding = self.model(img_tensor).cpu().numpy().flatten()

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    @torch.no_grad()
    def extract_embeddings_batch(self, face_imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract face embeddings from multiple aligned face images (batch processing)

        Args:
            face_imgs: List of aligned face images (112x112x3)

        Returns:
            List of face embedding vectors (512-d each)
        """
        if not face_imgs:
            return []

        # Preprocess all images
        batch_imgs = []
        for face_img in face_imgs:
            # Resize if needed
            if face_img.shape[:2] != (112, 112):
                face_img = cv2.resize(face_img, (112, 112))

            # Convert BGR to RGB
            img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # Transpose to CHW
            img = np.transpose(img, (2, 0, 1))

            batch_imgs.append(img)

        # Stack into batch
        batch_tensor = torch.from_numpy(np.stack(batch_imgs)).float().to(self.device)

        # Normalize
        batch_tensor.div_(255).sub_(0.5).div_(0.5)

        # Extract embeddings in batch
        embeddings = self.model(batch_tensor).cpu().numpy()

        # L2 normalize each embedding
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return [emb for emb in embeddings]

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return float(np.dot(emb1, emb2))


class ReferenceDatabase:
    """Reference face database management with multi-angle support"""

    def __init__(self, db_path: str = 'reference_db.pkl'):
        """
        Initialize reference database

        Args:
            db_path: Path to save/load database
        """
        self.db_path = db_path
        # New structure: {person_id: {angle: embedding}}
        self.db: Dict[str, Dict[str, np.ndarray]] = {}
        self.load()

    def add_person(self, person_id: str, embedding: np.ndarray, angle: str = 'front'):
        """
        Add or update a person's embedding for specific angle

        Args:
            person_id: Person identifier
            embedding: Face embedding
            angle: Angle category ('front', 'left', 'right', 'up', 'down')
        """
        if person_id not in self.db:
            self.db[person_id] = {}

        self.db[person_id][angle] = embedding
        self.save()

    def remove_person(self, person_id: str):
        """Remove a person from database"""
        if person_id in self.db:
            del self.db[person_id]
            self.save()

    def remove_person_angle(self, person_id: str, angle: str):
        """Remove specific angle for a person"""
        if person_id in self.db and angle in self.db[person_id]:
            del self.db[person_id][angle]
            if not self.db[person_id]:  # If no angles left, remove person
                del self.db[person_id]
            self.save()

    def get_all_persons(self) -> List[str]:
        """Get list of all person IDs"""
        return list(self.db.keys())

    def get_person_angles(self, person_id: str) -> List[str]:
        """Get list of captured angles for a person"""
        if person_id in self.db:
            return list(self.db[person_id].keys())
        return []

    def find_match(self, query_embedding: np.ndarray, threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Find best matching person across all angles

        Args:
            query_embedding: Query face embedding
            threshold: Minimum similarity threshold

        Returns:
            (person_id, similarity) or (None, 0.0) if no match
        """
        if not self.db:
            return None, 0.0

        best_match = None
        best_similarity = threshold

        # Compare against all angles for all persons
        for person_id, angles_dict in self.db.items():
            for angle, ref_embedding in angles_dict.items():
                similarity = EdgeFaceRecognizer.cosine_similarity(query_embedding, ref_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id

        return best_match, best_similarity

    def save(self):
        """Save database to disk"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.db, f)

    def load(self):
        """Load database from disk and migrate old format if needed"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                loaded_db = pickle.load(f)

            # Check if migration is needed (old format: {person_id: embedding})
            needs_migration = False
            for person_id, value in loaded_db.items():
                if isinstance(value, np.ndarray):
                    needs_migration = True
                    break

            if needs_migration:
                print("🔄 Migrating database to multi-angle format...")
                # Convert old format to new format
                new_db = {}
                for person_id, embedding in loaded_db.items():
                    if isinstance(embedding, np.ndarray):
                        # Old format: single embedding -> convert to {'front': embedding}
                        new_db[person_id] = {'front': embedding}
                    else:
                        # Already in new format
                        new_db[person_id] = embedding
                self.db = new_db
                self.save()  # Save migrated version
                print(f"✅ Migrated {len(self.db)} persons to new format")
            else:
                self.db = loaded_db
                print(f"✅ Loaded {len(self.db)} persons from database")
        else:
            print("📂 New database created")


class FaceRecognitionSystem:
    """Main real-time face recognition system"""

    def __init__(
        self,
        detector_method: str = 'mtcnn',
        edgeface_model_path: str = 'checkpoints/edgeface_xs_gamma_06.pt',
        edgeface_model_name: str = 'edgeface_xs_gamma_06',
        device: str = 'cuda',
        similarity_threshold: float = 0.5,
        use_npu: bool = False
    ):
        """
        Initialize face recognition system

        Args:
            detector_method: Face detection method ('mtcnn', 'yunet', 'yunet_npu', 'yolov5_face', 'yolov8')
            edgeface_model_path: Path to EdgeFace model (.pt for PyTorch, .dxnn for NPU)
            edgeface_model_name: EdgeFace model architecture name
            device: 'cuda', 'cpu', or 'npu'
            similarity_threshold: Minimum similarity for recognition
            use_npu: Whether to use NPU for EdgeFace recognition (overrides device setting)
        """
        print("🚀 Initializing Face Recognition System...")

        # Determine if we should use NPU
        # Automatically enable NPU for EdgeFace when yunet_npu is used
        self.use_npu = use_npu or device == 'npu' or detector_method == 'yunet_npu'

        # Device setup for detector
        if self.use_npu:
            detector_device = 'npu'
            self.device = 'npu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
            detector_device = self.device

        # Override detector device for yunet_npu
        if detector_method == 'yunet_npu':
            detector_device = 'npu'

        # Initialize face detector
        print(f"📷 Initializing face detector: {detector_method}")
        self.detector = UnifiedFaceDetector(detector_method, device=detector_device)

        # Initialize EdgeFace recognizer (NPU or PyTorch)
        print(f"🧠 Initializing EdgeFace recognizer...")
        if self.use_npu:
            if not NPU_AVAILABLE:
                raise ImportError("NPU recognizer not available. Install DeepX NPU SDK.")
            # Use NPU recognizer
            self.recognizer = EdgeFaceNPURecognizer(edgeface_model_path, edgeface_model_name, device='npu')
        else:
            # Use PyTorch recognizer
            self.recognizer = EdgeFaceRecognizer(edgeface_model_path, edgeface_model_name, self.device)

        # Initialize reference database
        print(f"💾 Initializing reference database...")
        self.ref_db = ReferenceDatabase()

        # Settings
        self.similarity_threshold = similarity_threshold

        # FPS calculation
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

        # Face tracking for temporal consistency
        self.tracked_faces = {}  # track_id -> {'bbox', 'person_id', 'similarity', 'last_seen'}
        self.next_track_id = 0
        self.max_tracking_frames = 10  # Max frames to keep track without detection

        print("✅ System initialized successfully!")

    def add_reference_from_image(self, image_path: str, person_id: str) -> bool:
        """
        Add reference person from image file

        Args:
            image_path: Path to reference image
            person_id: Person identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            img = Image.open(image_path)

            # Detect and align face
            aligned_face = self.detector.align(img)

            if aligned_face is None:
                print(f"❌ No face detected in {image_path}")
                return False

            # Convert PIL to numpy
            face_np = np.array(aligned_face)
            face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

            # Extract embedding
            embedding = self.recognizer.extract_embedding(face_np)

            # Add to database
            self.ref_db.add_person(person_id, embedding)

            print(f"✅ Added {person_id} to reference database")
            return True

        except Exception as e:
            print(f"❌ Error adding reference: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame with batch embedding extraction

        Args:
            frame: Input frame (BGR)

        Returns:
            (annotated_frame, detections)
            detections: List of {'person_id', 'similarity', 'bbox', 'landmarks'}
        """
        # Convert to PIL for detector
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Detect faces
        bboxes, landmarks = self.detector.detect_faces(pil_img)

        detections = []

        if bboxes is not None and len(bboxes) > 0:
            # Align all detected faces first
            aligned_faces = []
            valid_indices = []

            for i, (bbox, lm) in enumerate(zip(bboxes, landmarks)):
                # Align face
                aligned_face = self.detector.align_face(pil_img, lm)

                if aligned_face is not None:
                    # Convert to numpy BGR
                    face_np = np.array(aligned_face)
                    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    aligned_faces.append(face_np)
                    valid_indices.append(i)

            # Extract embeddings in batch (much faster!)
            if aligned_faces:
                embeddings = self.recognizer.extract_embeddings_batch(aligned_faces)

                # Build face_embeddings list
                face_embeddings = []
                emb_idx = 0
                for i, (bbox, lm) in enumerate(zip(bboxes, landmarks)):
                    if i in valid_indices:
                        face_embeddings.append({
                            'embedding': embeddings[emb_idx],
                            'bbox': bbox,
                            'landmarks': lm
                        })
                        emb_idx += 1
                    else:
                        face_embeddings.append(None)

                # Perform matching with assignment tracking
                detections = self.assign_identities(face_embeddings)

        # Update FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        # Annotate frame
        annotated_frame = self.draw_results(frame.copy(), detections)

        return annotated_frame, detections

    @staticmethod
    def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            bbox1: First bounding box [x1, y1, x2, y2, ...]
            bbox2: Second bounding box [x1, y1, x2, y2, ...]

        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def assign_identities(self, face_embeddings: List[Dict]) -> List[Dict]:
        """
        Assign identities to detected faces using greedy assignment with tracking

        Args:
            face_embeddings: List of face embedding data

        Returns:
            List of detections with assigned identities
        """
        current_frame_id = self.frame_count

        if not self.ref_db.db:
            # No references, mark all as unknown
            return [{
                'person_id': 'Unknown',
                'similarity': 0.0,
                'bbox': fe['bbox'],
                'landmarks': fe['landmarks']
            } for fe in face_embeddings if fe is not None]

        # Match current detections with tracked faces (temporal consistency)
        detection_to_track = {}  # detection_idx -> track_id
        for i, fe in enumerate(face_embeddings):
            if fe is None:
                continue

            best_iou = 0.3  # Minimum IoU threshold for tracking
            best_track_id = None

            for track_id, track_data in self.tracked_faces.items():
                iou = self.calculate_iou(fe['bbox'], track_data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                detection_to_track[i] = best_track_id

        # Calculate similarity matrix: [num_faces x num_references x num_angles]
        candidates = []
        for i, fe in enumerate(face_embeddings):
            if fe is None:
                continue

            # Check if this detection is tracked and has a stable identity
            if i in detection_to_track:
                track_id = detection_to_track[i]
                tracked_person = self.tracked_faces[track_id]['person_id']

                # If tracked person is known, verify it's still the same person
                if tracked_person != 'Unknown':
                    # Check similarity with tracked person
                    if tracked_person in self.ref_db.db:
                        max_sim = 0.0
                        for angle, ref_embedding in self.ref_db.db[tracked_person].items():
                            sim = EdgeFaceRecognizer.cosine_similarity(fe['embedding'], ref_embedding)
                            max_sim = max(max_sim, sim)

                        # If still similar enough, keep the tracked identity
                        if max_sim >= self.similarity_threshold * 0.8:  # Lower threshold for tracked faces
                            candidates.append({
                                'face_idx': i,
                                'person_id': tracked_person,
                                'similarity': max_sim,
                                'bbox': fe['bbox'],
                                'landmarks': fe['landmarks'],
                                'priority': 1  # High priority for tracked faces
                            })
                            continue

            # Compare against all angles for each person (normal matching)
            for person_id, angles_dict in self.ref_db.db.items():
                max_sim = 0.0
                for angle, ref_embedding in angles_dict.items():
                    similarity = EdgeFaceRecognizer.cosine_similarity(fe['embedding'], ref_embedding)
                    max_sim = max(max_sim, similarity)

                if max_sim >= self.similarity_threshold:
                    candidates.append({
                        'face_idx': i,
                        'person_id': person_id,
                        'similarity': max_sim,
                        'bbox': fe['bbox'],
                        'landmarks': fe['landmarks'],
                        'priority': 0  # Normal priority
                    })

        # Sort by priority (tracked faces first), then by similarity (highest first)
        candidates.sort(key=lambda x: (x['priority'], x['similarity']), reverse=True)

        # Greedy assignment: assign each face to best match, avoiding duplicates
        assigned_faces = set()
        assigned_persons = set()
        detections = []

        for candidate in candidates:
            face_idx = candidate['face_idx']
            person_id = candidate['person_id']

            # Skip if face or person already assigned
            if face_idx in assigned_faces or person_id in assigned_persons:
                continue

            # Assign this match
            detections.append({
                'person_id': person_id,
                'similarity': candidate['similarity'],
                'bbox': candidate['bbox'],
                'landmarks': candidate['landmarks']
            })

            assigned_faces.add(face_idx)
            assigned_persons.add(person_id)

        # Add unmatched faces as Unknown
        for i, fe in enumerate(face_embeddings):
            if fe is not None and i not in assigned_faces:
                detections.append({
                    'person_id': 'Unknown',
                    'similarity': 0.0,
                    'bbox': fe['bbox'],
                    'landmarks': fe['landmarks']
                })

        # Update tracking
        self.update_tracking(detections, current_frame_id)

        return detections

    def update_tracking(self, detections: List[Dict], current_frame_id: int):
        """
        Update face tracking based on current detections

        Args:
            detections: List of current frame detections
            current_frame_id: Current frame number
        """
        # Remove stale tracks
        stale_tracks = []
        for track_id, track_data in self.tracked_faces.items():
            if current_frame_id - track_data['last_seen'] > self.max_tracking_frames:
                stale_tracks.append(track_id)

        for track_id in stale_tracks:
            del self.tracked_faces[track_id]

        # Update or create tracks for current detections
        matched_tracks = set()

        for det in detections:
            # Find best matching track
            best_iou = 0.3
            best_track_id = None

            for track_id, track_data in self.tracked_faces.items():
                if track_id in matched_tracks:
                    continue
                iou = self.calculate_iou(det['bbox'], track_data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self.tracked_faces[best_track_id].update({
                    'bbox': det['bbox'],
                    'person_id': det['person_id'],
                    'similarity': det['similarity'],
                    'last_seen': current_frame_id
                })
                matched_tracks.add(best_track_id)
            else:
                # Create new track
                self.tracked_faces[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'person_id': det['person_id'],
                    'similarity': det['similarity'],
                    'last_seen': current_frame_id
                }
                self.next_track_id += 1

    def draw_results(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on frame"""

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw detections
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])

            # Draw bounding box
            color = (0, 255, 0) if det['person_id'] != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw person ID and similarity
            label = f"{det['person_id']}: {det['similarity']:.2f}"

            # Background for text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)

            # Text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Draw landmarks
            landmarks = det['landmarks']
            if landmarks is not None:
                for i in range(0, len(landmarks), 2):
                    x, y = int(landmarks[i]), int(landmarks[i+1])
                    cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        return frame

    def add_reference_from_frame(self, frame: np.ndarray, person_id: str) -> bool:
        """
        Add reference person from camera frame

        Args:
            frame: Camera frame (BGR)
            person_id: Person identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to PIL for detector
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Detect and align face
            aligned_face = self.detector.align(pil_img)

            if aligned_face is None:
                print(f"❌ No face detected in frame")
                return False

            # Convert PIL to numpy
            face_np = np.array(aligned_face)
            face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

            # Extract embedding
            embedding = self.recognizer.extract_embedding(face_np)

            # Add to database
            self.ref_db.add_person(person_id, embedding)

            # Save captured image
            os.makedirs("captured_references", exist_ok=True)
            save_path = f"captured_references/{person_id}.jpg"
            cv2.imwrite(save_path, cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR))

            print(f"✅ Added {person_id} to reference database")
            print(f"💾 Saved reference image to {save_path}")
            return True

        except Exception as e:
            print(f"❌ Error adding reference: {e}")
            return False

    def run_camera(self, camera_id: int = 0, enable_capture: bool = True):
        """
        Run face recognition on camera feed

        Args:
            camera_id: Camera device ID
            enable_capture: Enable 'c' key to capture reference
        """
        print(f"📹 Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return

        print("✅ Camera opened successfully")
        print("Controls:")
        print("  'q' - Quit")
        if enable_capture:
            print("  'c' - Capture current frame as reference")
        print(f"📊 Registered persons: {self.ref_db.get_all_persons()}")

        current_frame = None

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("❌ Failed to grab frame")
                    break

                current_frame = frame.copy()

                # Process frame
                annotated_frame, detections = self.process_frame(frame)

                # Display
                cv2.imshow('Face Recognition System', annotated_frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF

                # Quit on 'q'
                if key == ord('q'):
                    break

                # Capture on 'c'
                elif key == ord('c') and enable_capture:
                    print("\n📸 Capture mode activated")
                    person_id = input("Enter person name/ID (or press Enter to cancel): ").strip()

                    if person_id:
                        success = self.add_reference_from_frame(current_frame, person_id)
                        if success:
                            print(f"✅ {person_id} added successfully!")
                            print(f"📊 Registered persons: {self.ref_db.get_all_persons()}")
                        else:
                            print(f"❌ Failed to add {person_id}")
                    else:
                        print("⏭️  Capture cancelled")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Camera closed")


def main():
    """Main function for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Face Recognition System')
    parser.add_argument('--detector', type=str, default='mtcnn',
                       choices=['mtcnn', 'yunet', 'yunet_npu', 'yolov5_face', 'yolov8'],
                       help='Face detection method')
    parser.add_argument('--model', type=str, default='checkpoints/edgeface_xs_gamma_06.pt',
                       help='EdgeFace model path (.pt for PyTorch, .dxnn for NPU)')
    parser.add_argument('--model-name', type=str, default='edgeface_xs_gamma_06',
                       help='EdgeFace model architecture name')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'npu'], help='Device to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for recognition')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--add-ref', type=str, nargs=2, metavar=('IMAGE', 'ID'),
                       help='Add reference image: --add-ref path/to/image.jpg person_name')

    args = parser.parse_args()

    # Initialize system
    system = FaceRecognitionSystem(
        detector_method=args.detector,
        edgeface_model_path=args.model,
        edgeface_model_name=args.model_name,
        device=args.device,
        similarity_threshold=args.threshold
    )

    # Add reference if specified
    if args.add_ref:
        image_path, person_id = args.add_ref
        system.add_reference_from_image(image_path, person_id)

    # Run camera
    system.run_camera(args.camera)


if __name__ == '__main__':
    main()
