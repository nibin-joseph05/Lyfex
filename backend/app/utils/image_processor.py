import cv2
import numpy as np
from datetime import datetime
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        """Initialize image processor with default parameters"""
        self.target_size = (640, 480)
        self.gaussian_kernel = (5, 5)
        logger.info("Image processor initialized")
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess image for better health analysis"""
        try:
            processed = frame.copy()
            
            # Resize if too large
            h, w = processed.shape[:2]
            if w > 1280 or h > 720:
                processed = self.resize_image(processed, max_width=1280, max_height=720)
            
            # Enhance contrast and brightness
            processed = self.enhance_contrast(processed)
            
            # Reduce noise
            processed = cv2.GaussianBlur(processed, self.gaussian_kernel, 0)
            
            return processed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return frame
    
    def resize_image(self, image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / w, max_height / h)
        
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
            
        except Exception as e:
            logger.error(f"Contrast enhancement failed: {e}")
            return image
    
    def extract_face_roi(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int], 
                        padding: float = 0.1) -> np.ndarray:
        """Extract face region of interest with padding"""
        x, y, w, h = face_rect
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate coordinates with padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        return frame[y1:y2, x1:x2]
    
    def extract_skin_regions(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> dict:
        """Extract specific skin regions for analysis"""
        regions = {}
        
        try:
            if landmarks is not None and len(landmarks) >= 68:
                # Forehead region (good for heart rate)
                forehead_points = landmarks[19:24]  # Approximate forehead area
                regions['forehead'] = self._extract_region_by_points(face_roi, forehead_points)
                
                # Cheek regions
                left_cheek = landmarks[1:3]
                right_cheek = landmarks[15:17]
                regions['left_cheek'] = self._extract_region_by_points(face_roi, left_cheek)
                regions['right_cheek'] = self._extract_region_by_points(face_roi, right_cheek)
                
                # Nose bridge (good for respiratory analysis)
                nose_points = landmarks[27:31]
                regions['nose_bridge'] = self._extract_region_by_points(face_roi, nose_points)
                
            else:
                # Fallback: divide face into regions without landmarks
                h, w = face_roi.shape[:2]
                
                # Forehead (upper 1/3 of face)
                regions['forehead'] = face_roi[0:h//3, w//4:3*w//4]
                
                # Cheeks (middle 1/3, left and right)
                regions['left_cheek'] = face_roi[h//3:2*h//3, 0:w//2]
                regions['right_cheek'] = face_roi[h//3:2*h//3, w//2:w]
                
                # Nose area (center)
                regions['nose_bridge'] = face_roi[h//3:2*h//3, w//3:2*w//3]
            
        except Exception as e:
            logger.error(f"Skin region extraction failed: {e}")
            regions = {'full_face': face_roi}
        
        return regions
    
    def _extract_region_by_points(self, face_roi: np.ndarray, points: np.ndarray, 
                                 padding: int = 10) -> np.ndarray:
        """Extract region around specific landmark points"""
        if len(points) == 0:
            return face_roi
        
        # Get bounding box of points
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # Add padding
        x_min = max(0, int(x_min - padding))
        y_min = max(0, int(y_min - padding))
        x_max = min(face_roi.shape[1], int(x_max + padding))
        y_max = min(face_roi.shape[0], int(y_max + padding))
        
        return face_roi[y_min:y_max, x_min:x_max]
    
    def normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """Normalize lighting conditions in the image"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Check if image is too dark or too bright
            mean_brightness = np.mean(gray)
            
            result = image.copy()
            
            if mean_brightness < 80:  # Too dark
                # Increase brightness and contrast
                alpha = 1.5  # Contrast
                beta = 50    # Brightness
                result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
                
            elif mean_brightness > 180:  # Too bright
                # Decrease brightness, maintain contrast
                alpha = 0.8  # Contrast
                beta = -30   # Brightness
                result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
            
            return result
            
        except Exception as e:
            logger.error(f"Lighting normalization failed: {e}")
            return image
    
    def apply_skin_segmentation(self, face_roi: np.ndarray) -> np.ndarray:
        """Apply skin segmentation to isolate skin pixels"""
        try:
            # Convert to HSV and YCrCb color spaces
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            
            # Define skin color ranges in HSV
            lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
            mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # Define skin color ranges in YCrCb
            lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
            upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Combine masks
            skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply mask to original image
            skin_segmented = cv2.bitwise_and(face_roi, face_roi, mask=skin_mask)
            
            return skin_segmented
            
        except Exception as e:
            logger.error(f"Skin segmentation failed: {e}")
            return face_roi
    
    def calculate_image_quality_metrics(self, image: np.ndarray) -> dict:
        """Calculate various image quality metrics"""
        metrics = {
            'brightness': 0.0,
            'contrast': 0.0,
            'sharpness': 0.0,
            'noise_level': 0.0,
            'quality_score': 0.0
        }
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Brightness (mean pixel intensity)
            metrics['brightness'] = float(np.mean(gray))
            
            # Contrast (standard deviation of pixel intensities)
            metrics['contrast'] = float(np.std(gray))
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            metrics['sharpness'] = float(laplacian.var())
            
            # Noise level (estimate using high-frequency content)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise_estimate = cv2.filter2D(gray, cv2.CV_64F, kernel)
            metrics['noise_level'] = float(np.std(noise_estimate))
            
            # Overall quality score (normalized combination of metrics)
            brightness_score = 1.0 - abs(metrics['brightness'] - 127) / 127  # Optimal at 127
            contrast_score = min(metrics['contrast'] / 50, 1.0)  # Good contrast > 50
            sharpness_score = min(metrics['sharpness'] / 500, 1.0)  # Good sharpness > 500
            noise_score = max(0, 1.0 - metrics['noise_level'] / 100)  # Lower noise is better
            
            metrics['quality_score'] = (brightness_score + contrast_score + 
                                      sharpness_score + noise_score) / 4
            
        except Exception as e:
            logger.error(f"Image quality calculation failed: {e}")
        
        return metrics
    
    def stabilize_frame(self, current_frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply frame stabilization to reduce motion artifacts"""
        if previous_frame is None:
            return current_frame
        
        try:
            # Convert to grayscale
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features in previous frame
            corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, 
                                            qualityLevel=0.01, minDistance=30)
            
            if corners is not None and len(corners) > 0:
                # Track features in current frame
                new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, corners, None
                )
                
                # Filter good matches
                good_corners = corners[status == 1]
                good_new_corners = new_corners[status == 1]
                
                if len(good_corners) >= 4:
                    # Find transformation matrix
                    transform_matrix = cv2.estimateAffinePartial2D(
                        good_corners, good_new_corners
                    )[0]
                    
                    if transform_matrix is not None:
                        # Apply stabilization
                        h, w = current_frame.shape[:2]
                        stabilized = cv2.warpAffine(current_frame, transform_matrix, (w, h))
                        return stabilized
            
        except Exception as e:
            logger.error(f"Frame stabilization failed: {e}")
        
        return current_frame
    
    @staticmethod
    def get_current_timestamp() -> str:
        """Get current timestamp as string"""
        return datetime.now().isoformat()
    
    def create_analysis_overlay(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int], 
                               landmarks: Optional[np.ndarray] = None, 
                               health_data: dict = None) -> np.ndarray:
        """Create overlay with analysis information"""
        overlay_frame = frame.copy()
        x, y, w, h = face_rect
        
        try:
            # Draw face rectangle
            cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw landmarks if available
            if landmarks is not None:
                for i, point in enumerate(landmarks):
                    cv2.circle(overlay_frame, tuple(point.astype(int)), 2, (0, 0, 255), -1)
            
            # Add health data overlay
            if health_data:
                y_offset = y - 10
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (255, 255, 255)
                thickness = 1
                
                for key, value in health_data.items():
                    if isinstance(value, (int, float, str)):
                        text = f"{key}: {value}"
                        cv2.putText(overlay_frame, text, (x, y_offset), 
                                   font, font_scale, color, thickness)
                        y_offset -= 25
            
        except Exception as e:
            logger.error(f"Overlay creation failed: {e}")
        
        return overlay_frame