import cv2
import dlib
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        """Initialize face detector with dlib and OpenCV"""
        try:
            # Initialize face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Initialize dlib face detector and landmark predictor
            self.dlib_detector = dlib.get_frontal_face_detector()
            
            # Try to load the landmark predictor (requires shape_predictor_68_face_landmarks.dat)
            try:
                self.landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
                self.landmarks_available = True
                logger.info("Dlib landmark predictor loaded successfully")
            except:
                self.landmarks_available = False
                logger.warning("Dlib landmark predictor not found. Facial landmark detection disabled.")
            
            logger.info("Face detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise
    
    def detect_faces_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar Cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces.tolist()
    
    def detect_faces_dlib(self, frame: np.ndarray) -> List:
        """Detect faces using dlib"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray)
        return faces
    
    def detect_landmarks(self, frame: np.ndarray, face_rect) -> Optional[np.ndarray]:
        if not self.landmarks_available:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if isinstance(face_rect, (tuple, list)):
                x, y, w, h = face_rect
                dlib_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h) 
            else:
                dlib_rect = face_rect
            landmarks = self.landmark_predictor(gray, dlib_rect)
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
            return landmarks_np  # Fixed typo: was 'landmarksros'

        except Exception as e:
            logger.error(f"Landmark detection failed: {e}")
            return None
    
    def detect_face_and_landmarks(self, frame: np.ndarray) -> Tuple[List, List]:
        """Detect faces and their landmarks in one go"""
        faces = []
        landmarks = []
        
        try:
            # Use dlib for better accuracy if available
            if self.landmarks_available:
                dlib_faces = self.detect_faces_dlib(frame)
                
                for face in dlib_faces:
                    # Convert dlib rectangle to (x, y, w, h) format
                    x, y = face.left(), face.top()
                    w, h = face.width(), face.height()
                    faces.append((x, y, w, h))
                    
                    # Get landmarks for this face
                    face_landmarks = self.detect_landmarks(frame, face)
                    landmarks.append(face_landmarks)
            else:
                # Fallback to OpenCV
                opencv_faces = self.detect_faces_opencv(frame)
                faces = opencv_faces
                landmarks = [None] * len(faces)  # No landmarks available
            
            logger.debug(f"Detected {len(faces)} faces")
            return faces, landmarks
            
        except Exception as e:
            logger.error(f"Face and landmark detection failed: {e}")
            return [], []
    
    def get_face_center(self, face_rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get the center point of a face rectangle"""
        x, y, w, h = face_rect
        center_x = x + w // 2
        center_y = y + h // 2
        return center_x, center_y
    
    def get_face_roi(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int], 
                     padding: float = 0.1) -> np.ndarray:
        """Extract face region of interest with padding"""
        x, y, w, h = face_rect
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate new coordinates with padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        return frame[y1:y2, x1:x2]
    
    def validate_face_quality(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> dict:
        """Validate face quality for health analysis"""
        quality_metrics = {
            'size_adequate': False,
            'brightness_good': False,
            'blur_acceptable': False,
            'pose_frontal': False,
            'overall_quality': 'Poor'
        }
        
        if face_roi is None or face_roi.size == 0:
            return quality_metrics
        
        try:
            h, w = face_roi.shape[:2]
            
            # Check size
            quality_metrics['size_adequate'] = h > 100 and w > 100
            
            # Check brightness
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            mean_brightness = np.mean(gray)
            quality_metrics['brightness_good'] = 50 < mean_brightness < 200
            
            # Check blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['blur_acceptable'] = laplacian_var > 100
            
            # Check pose (if landmarks available)
            if landmarks is not None and len(landmarks) >= 68:
                # Simple pose estimation using nose and eye landmarks
                left_eye = np.mean(landmarks[36:42], axis=0)
                right_eye = np.mean(landmarks[42:48], axis=0)
                nose_tip = landmarks[30]
                
                # Calculate symmetry
                eye_center = (left_eye + right_eye) / 2
                nose_offset = abs(nose_tip[0] - eye_center[0])
                face_width = abs(right_eye[0] - left_eye[0])
                symmetry_ratio = nose_offset / face_width if face_width > 0 else 1
                
                quality_metrics['pose_frontal'] = symmetry_ratio < 0.15
            else:
                quality_metrics['pose_frontal'] = True  # Assume frontal if no landmarks
            
            # Overall quality assessment
            score = sum([
                quality_metrics['size_adequate'],
                quality_metrics['brightness_good'],
                quality_metrics['blur_acceptable'],
                quality_metrics['pose_frontal']
            ])
            
            if score >= 3:
                quality_metrics['overall_quality'] = 'Good'
            elif score >= 2:
                quality_metrics['overall_quality'] = 'Acceptable'
            else:
                quality_metrics['overall_quality'] = 'Poor'
            
        except Exception as e:
            logger.error(f"Face quality validation failed: {e}")
        
        return quality_metrics
    
    def draw_face_detection(self, frame: np.ndarray, faces: List, landmarks: List = None) -> np.ndarray:
        """Draw face detection results on frame for visualization"""
        result_frame = frame.copy()
        
        for i, face in enumerate(faces):
            x, y, w, h = face
            
            # Draw face rectangle
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw landmarks if available
            if landmarks and i < len(landmarks) and landmarks[i] is not None:
                for point in landmarks[i]:
                    cv2.circle(result_frame, tuple(point.astype(int)), 2, (0, 0, 255), -1)
        
        return result_frame