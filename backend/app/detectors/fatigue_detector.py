import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class FatigueDetector:
    def __init__(self):
        """Initialize fatigue detector using eye and facial analysis"""
        self.ear_history = deque(maxlen=30)  # Eye Aspect Ratio history
        self.perclos_history = deque(maxlen=100)  # PERCLOS values
        self.yawn_history = deque(maxlen=20)
        self.head_pose_history = deque(maxlen=30)
        
        # Fatigue thresholds
        self.ear_threshold = 0.21  # Eye closure threshold
        self.yawn_threshold = 0.6   # Mouth aspect ratio for yawn
        self.perclos_threshold = 0.15  # Percentage of eye closure over time
        
        # Fatigue levels
        self.fatigue_levels = {
            'Very Alert': (0.0, 0.2),
            'Alert': (0.2, 0.4),
            'Slightly Drowsy': (0.4, 0.6),
            'Drowsy': (0.6, 0.8),
            'Very Drowsy': (0.8, 1.0)
        }
        
        logger.info("Fatigue detector initialized with eye and behavioral analysis")
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        try:
            # Vertical eye landmarks
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Horizontal eye landmark
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Eye aspect ratio
            ear = (A + B) / (2.0 * C)
            return float(ear)
            
        except Exception as e:
            logger.error(f"EAR calculation failed: {e}")
            return 0.0
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
        try:
            # Vertical distances
            A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])  # 50-58
            B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])   # 52-56
            C = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[9])   # 51-57
            
            # Horizontal distance
            D = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])   # 48-54
            
            # Mouth aspect ratio
            mar = (A + B + C) / (3.0 * D)
            return float(mar)
            
        except Exception as e:
            logger.error(f"MAR calculation failed: {e}")
            return 0.0
    
    def analyze_eye_closure(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Analyze eye closure patterns for fatigue detection"""
        try:
            eye_analysis = {
                'left_ear': 0.0,
                'right_ear': 0.0,
                'avg_ear': 0.0,
                'eye_closure_rate': 0.0,
                'blink_frequency': 0.0
            }
            
            if landmarks is not None and len(landmarks) >= 68:
                # Get eye landmarks
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                
                # Calculate EAR for both eyes
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2
                
                eye_analysis['left_ear'] = left_ear
                eye_analysis['right_ear'] = right_ear
                eye_analysis['avg_ear'] = avg_ear
                
                # Add to history
                self.ear_history.append(avg_ear)
                
                # Calculate eye closure metrics
                if len(self.ear_history) >= 10:
                    recent_ears = list(self.ear_history)[-10:]
                    
                    # Eye closure rate (percentage of time eyes are closed/nearly closed)
                    closed_count = sum(1 for ear in recent_ears if ear < self.ear_threshold)
                    eye_analysis['eye_closure_rate'] = closed_count / len(recent_ears)
                    
                    # Blink frequency analysis
                    blink_count = 0
                    for i in range(1, len(recent_ears)):
                        if recent_ears[i-1] > self.ear_threshold and recent_ears[i] <= self.ear_threshold:
                            blink_count += 1
                    eye_analysis['blink_frequency'] = blink_count / 10.0  # per second approximation
            
            else:
                # Fallback: image-based eye analysis
                eye_analysis.update(self._analyze_eyes_without_landmarks(face_roi))
            
            return eye_analysis
            
        except Exception as e:
            logger.error(f"Eye closure analysis failed: {e}")
            return {'left_ear': 0.0, 'right_ear': 0.0, 'avg_ear': 0.0, 'eye_closure_rate': 0.0}
    
    def _analyze_eyes_without_landmarks(self, face_roi: np.ndarray) -> Dict:
        """Analyze eyes using image processing when landmarks unavailable"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # Use Haar cascade for eye detection
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(eyes) >= 2:
                # Analyze detected eyes
                eye_openness_scores = []
                for (x, y, w, h) in eyes[:2]:  # Take first two eyes
                    eye_roi = gray[y:y+h, x:x+w]
                    
                    # Analyze eye openness using dark pixel ratio
                    _, binary = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    dark_ratio = np.sum(binary == 0) / binary.size
                    
                    # More dark pixels might indicate closed eyes
                    openness = 1.0 - dark_ratio
                    eye_openness_scores.append(openness)
                
                avg_openness = np.mean(eye_openness_scores)
                
                return {
                    'left_ear': eye_openness_scores[0] if len(eye_openness_scores) > 0 else 0.0,
                    'right_ear': eye_openness_scores[1] if len(eye_openness_scores) > 1 else 0.0,
                    'avg_ear': avg_openness,
                    'eye_closure_rate': max(0, 1 - avg_openness)
                }
            
            return {'left_ear': 0.0, 'right_ear': 0.0, 'avg_ear': 0.0, 'eye_closure_rate': 0.0}
            
        except Exception as e:
            logger.error(f"Image-based eye analysis failed: {e}")
            return {'left_ear': 0.0, 'right_ear': 0.0, 'avg_ear': 0.0, 'eye_closure_rate': 0.0}
    
    def detect_yawning(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Detect yawning as a fatigue indicator"""
        try:
            yawn_analysis = {
                'mouth_aspect_ratio': 0.0,
                'is_yawning': False,
                'yawn_frequency': 0.0,
                'yawn_confidence': 0.0
            }
            
            if landmarks is not None and len(landmarks) >= 68:
                # Get mouth landmarks
                mouth_landmarks = landmarks[48:68]
                
                # Calculate MAR
                mar = self.calculate_mouth_aspect_ratio(mouth_landmarks)
                yawn_analysis['mouth_aspect_ratio'] = mar
                
                # Determine if yawning
                is_yawning = mar > self.yawn_threshold
                yawn_analysis['is_yawning'] = is_yawning
                
                # Calculate confidence based on how much MAR exceeds threshold
                if is_yawning:
                    yawn_analysis['yawn_confidence'] = min((mar - self.yawn_threshold) / 0.3, 1.0)
                
                # Add to history
                self.yawn_history.append(1.0 if is_yawning else 0.0)
                
                # Calculate yawn frequency
                if len(self.yawn_history) >= 10:
                    recent_yawns = list(self.yawn_history)[-10:]
                    yawn_analysis['yawn_frequency'] = sum(recent_yawns) / len(recent_yawns)
            
            else:
                # Fallback: analyze mouth region for opening
                h, w = face_roi.shape[:2]
                mouth_roi = face_roi[2*h//3:h, w//4:3*w//4]
                
                if mouth_roi.size > 0:
                    gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY) if len(mouth_roi.shape) == 3 else mouth_roi
                    
                    # Detect dark regions (open mouth)
                    _, binary = cv2.threshold(gray_mouth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    dark_ratio = np.sum(binary == 0) / binary.size
                    
                    yawn_analysis['mouth_aspect_ratio'] = dark_ratio
                    yawn_analysis['is_yawning'] = dark_ratio > 0.4
                    yawn_analysis['yawn_confidence'] = min(dark_ratio, 1.0)
            
            return yawn_analysis
            
        except Exception as e:
            logger.error(f"Yawn detection failed: {e}")
            return {'mouth_aspect_ratio': 0.0, 'is_yawning': False, 'yawn_frequency': 0.0}
    
    def analyze_head_pose(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Analyze head pose for fatigue-related head dropping"""
        try:
            head_pose = {
                'head_tilt': 0.0,
                'head_nod': 0.0,
                'pose_stability': 1.0,
                'fatigue_indicator': 0.0
            }
            
            if landmarks is not None and len(landmarks) >= 68:
                # Key points for head pose estimation
                nose_tip = landmarks[30]
                chin = landmarks[8]
                left_eye = np.mean(landmarks[36:42], axis=0)
                right_eye = np.mean(landmarks[42:48], axis=0)
                
                # Calculate head tilt (roll)
                eye_center = (left_eye + right_eye) / 2
                eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                head_pose['head_tilt'] = abs(np.degrees(eye_angle))
                
                # Calculate head nod (pitch) - approximation
                face_height = chin[1] - eye_center[1]
                nose_offset = nose_tip[1] - eye_center[1]
                nod_ratio = nose_offset / (face_height + 1e-8)
                head_pose['head_nod'] = abs(nod_ratio)
                
                # Add to history for stability analysis
                self.head_pose_history.append([head_pose['head_tilt'], head_pose['head_nod']])
                
                # Calculate pose stability
                if len(self.head_pose_history) >= 10:
                    recent_poses = np.array(list(self.head_pose_history)[-10:])
                    pose_variance = np.mean(np.var(recent_poses, axis=0))
                    head_pose['pose_stability'] = max(0, 1 - pose_variance / 50)
                
                # Fatigue indicator based on excessive head movement or drooping
                if head_pose['head_tilt'] > 15 or head_pose['head_nod'] > 0.3:
                    head_pose['fatigue_indicator'] = 0.6
                elif head_pose['pose_stability'] < 0.7:
                    head_pose['fatigue_indicator'] = 0.4
            
            return head_pose
            
        except Exception as e:
            logger.error(f"Head pose analysis failed: {e}")
            return {'head_tilt': 0.0, 'head_nod': 0.0, 'pose_stability': 1.0, 'fatigue_indicator': 0.0}
    
    def calculate_perclos(self) -> float:
        """Calculate PERCLOS (Percentage of Eyelid Closure Over the Pupil Over Time)"""
        try:
            if len(self.ear_history) < 10:
                return 0.0
            
            # Count frames where eyes are closed (EAR below threshold)
            recent_ears = list(self.ear_history)
            closed_frames = sum(1 for ear in recent_ears if ear < self.ear_threshold)
            
            perclos = closed_frames / len(recent_ears)
            
            # Add to PERCLOS history
            self.perclos_history.append(perclos)
            
            return float(perclos)
            
        except Exception as e:
            logger.error(f"PERCLOS calculation failed: {e}")
            return 0.0
    
    def analyze_facial_expressions_for_fatigue(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Analyze facial expressions that indicate fatigue"""
        try:
            expression_analysis = {
                'droopy_eyelids': 0.0,
                'slack_jaw': 0.0,
                'facial_asymmetry': 0.0,
                'overall_alertness': 1.0
            }
            
            if landmarks is not None and len(landmarks) >= 68:
                # Analyze eyelid droop
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                
                # Calculate upper eyelid position
                left_upper_lid = np.mean(landmarks[37:39], axis=0)  # Upper eyelid points
                left_lower_lid = np.mean(landmarks[40:42], axis=0)  # Lower eyelid points
                left_lid_distance = np.linalg.norm(left_upper_lid - left_lower_lid)
                
                right_upper_lid = np.mean(landmarks[43:45], axis=0)
                right_lower_lid = np.mean(landmarks[46:48], axis=0)
                right_lid_distance = np.linalg.norm(right_upper_lid - right_lower_lid)
                
                avg_lid_distance = (left_lid_distance + right_lid_distance) / 2
                
                # Droopy eyelids indicator (lower distance = more droop)
                normal_lid_distance = 15.0  # Approximate normal distance
                droop_factor = max(0, 1 - avg_lid_distance / normal_lid_distance)
                expression_analysis['droopy_eyelids'] = min(droop_factor, 1.0)
                
                # Analyze jaw slack
                mouth_landmarks = landmarks[48:68]
                jaw_points = landmarks[5:12]
                
                # Calculate mouth opening
                mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])  # Center mouth height
                mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])   # Mouth width
                
                mouth_ratio = mouth_height / (mouth_width + 1e-8)
                
                # Slack jaw indicator (slightly open mouth at rest)
                if mouth_ratio > 0.15:  # Mouth slightly open
                    expression_analysis['slack_jaw'] = min(mouth_ratio / 0.4, 1.0)
                
                # Analyze facial asymmetry (fatigue can cause asymmetric expressions)
                left_side_points = landmarks[:9]   # Left face contour
                right_side_points = landmarks[8:17] # Right face contour
                
                # Calculate asymmetry
                face_center = landmarks[30]  # Nose tip as center reference
                
                left_distances = [np.linalg.norm(point - face_center) for point in left_side_points]
                right_distances = [np.linalg.norm(point - face_center) for point in right_side_points]
                
                # Reverse right distances for comparison
                right_distances = right_distances[::-1]
                
                if len(left_distances) == len(right_distances):
                    asymmetry_score = np.mean([abs(l - r) for l, r in zip(left_distances, right_distances)])
                    face_width = np.linalg.norm(landmarks[0] - landmarks[16])
                    normalized_asymmetry = asymmetry_score / (face_width + 1e-8)
                    expression_analysis['facial_asymmetry'] = min(normalized_asymmetry * 5, 1.0)
                
                # Overall alertness score
                alertness_factors = [
                    1.0 - expression_analysis['droopy_eyelids'],
                    1.0 - expression_analysis['slack_jaw'],
                    1.0 - expression_analysis['facial_asymmetry'] * 0.5
                ]
                expression_analysis['overall_alertness'] = np.mean(alertness_factors)
            
            return expression_analysis
            
        except Exception as e:
            logger.error(f"Facial expression analysis failed: {e}")
            return {'droopy_eyelids': 0.0, 'slack_jaw': 0.0, 'facial_asymmetry': 0.0, 'overall_alertness': 1.0}
    
    def assess_fatigue(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Comprehensive fatigue assessment"""
        try:
            if face_roi is None or face_roi.size == 0:
                return {
                    'fatigue_level': 'Unknown',
                    'alertness_score': 0.0,
                    'confidence': 0.0,
                    'indicators': {}
                }
            
            # Analyze different fatigue indicators
            eye_analysis = self.analyze_eye_closure(face_roi, landmarks)
            yawn_analysis = self.detect_yawning(face_roi, landmarks)
            head_pose = self.analyze_head_pose(face_roi, landmarks)
            expression_analysis = self.analyze_facial_expressions_for_fatigue(face_roi, landmarks)
            
            # Calculate PERCLOS
            perclos = self.calculate_perclos()
            
            # Fatigue scoring weights
            fatigue_indicators = {
                'eye_closure': eye_analysis['eye_closure_rate'] * 0.25,
                'perclos': perclos * 0.20,
                'yawning': yawn_analysis.get('yawn_frequency', 0.0) * 0.15,
                'head_pose': head_pose['fatigue_indicator'] * 0.15,
                'droopy_eyelids': expression_analysis['droopy_eyelids'] * 0.10,
                'slack_jaw': expression_analysis['slack_jaw'] * 0.10,
                'pose_instability': (1.0 - head_pose['pose_stability']) * 0.05
            }
            
            # Calculate overall fatigue score
            fatigue_score = sum(fatigue_indicators.values())
            
            # Alertness score (inverse of fatigue)
            alertness_score = max(0.0, 1.0 - fatigue_score)
            
            # Determine fatigue level
            fatigue_level = self._categorize_fatigue_level(fatigue_score)
            
            # Calculate confidence based on data availability
            confidence_factors = [
                1.0 if len(self.ear_history) >= 20 else len(self.ear_history) / 20,
                1.0 if len(self.yawn_history) >= 10 else len(self.yawn_history) / 10,
                1.0 if len(self.head_pose_history) >= 10 else len(self.head_pose_history) / 10
            ]
            confidence = np.mean(confidence_factors)
            
            # Detailed analysis
            detailed_indicators = {
                'eye_closure_rate': round(eye_analysis['eye_closure_rate'], 3),
                'perclos': round(perclos, 3),
                'yawn_frequency': round(yawn_analysis.get('yawn_frequency', 0.0), 3),
                'current_yawning': yawn_analysis.get('is_yawning', False),
                'head_tilt': round(head_pose['head_tilt'], 2),
                'head_stability': round(head_pose['pose_stability'], 3),
                'droopy_eyelids': round(expression_analysis['droopy_eyelids'], 3),
                'slack_jaw': round(expression_analysis['slack_jaw'], 3),
                'facial_asymmetry': round(expression_analysis['facial_asymmetry'], 3)
            }
            
            # Generate alerts/recommendations
            alerts = self._generate_fatigue_alerts(fatigue_score, detailed_indicators)
            
            result = {
                'fatigue_level': fatigue_level,
                'alertness_score': round(alertness_score, 3),
                'fatigue_score': round(fatigue_score, 3),
                'confidence': round(confidence, 3),
                'indicators': detailed_indicators,
                'alerts': alerts,
                'perclos': round(perclos, 3),
                'eye_analysis': {
                    'left_ear': round(eye_analysis['left_ear'], 3),
                    'right_ear': round(eye_analysis['right_ear'], 3),
                    'avg_ear': round(eye_analysis['avg_ear'], 3)
                },
                'samples_analyzed': len(self.ear_history),
                'trend': self._calculate_fatigue_trend()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Fatigue assessment failed: {e}")
            return {
                'fatigue_level': 'Error',
                'alertness_score': 0.0,
                'confidence': 0.0,
                'indicators': {}
            }
    
    def _categorize_fatigue_level(self, fatigue_score: float) -> str:
        """Categorize fatigue level based on score"""
        # Invert the levels since we want higher scores to indicate more fatigue
        inverted_levels = {
            'Very Drowsy': (0.8, 1.0),
            'Drowsy': (0.6, 0.8),
            'Slightly Drowsy': (0.4, 0.6),
            'Alert': (0.2, 0.4),
            'Very Alert': (0.0, 0.2)
        }
        
        for level, (min_val, max_val) in inverted_levels.items():
            if min_val <= fatigue_score < max_val:
                return level
        return 'Very Drowsy'
    
    def _generate_fatigue_alerts(self, fatigue_score: float, indicators: Dict) -> List[str]:
        """Generate fatigue-related alerts and warnings"""
        alerts = []
        
        # High fatigue alerts
        if fatigue_score > 0.7:
            alerts.append("High fatigue detected - consider taking a break")
        elif fatigue_score > 0.5:
            alerts.append("Moderate fatigue detected - monitor alertness")
        
        # Specific indicator alerts
        if indicators.get('perclos', 0) > 0.15:
            alerts.append("Frequent eye closure detected")
        
        if indicators.get('yawn_frequency', 0) > 0.3:
            alerts.append("Frequent yawning detected")
        
        if indicators.get('current_yawning', False):
            alerts.append("Currently yawning")
        
        if indicators.get('head_tilt', 0) > 20:
            alerts.append("Excessive head tilting detected")
        
        if indicators.get('head_stability', 1) < 0.5:
            alerts.append("Unstable head position detected")
        
        if indicators.get('droopy_eyelids', 0) > 0.6:
            alerts.append("Droopy eyelids detected")
        
        return alerts
    
    def _calculate_fatigue_trend(self) -> str:
        """Calculate fatigue trend from recent PERCLOS history"""
        try:
            if len(self.perclos_history) < 10:
                return 'Insufficient data'
            
            recent_perclos = list(self.perclos_history)[-10:]
            early_avg = np.mean(recent_perclos[:5])
            recent_avg = np.mean(recent_perclos[-5:])
            
            diff = recent_avg - early_avg
            
            if diff > 0.05:
                return 'Increasing fatigue'
            elif diff < -0.05:
                return 'Decreasing fatigue'
            else:
                return 'Stable'
                
        except Exception as e:
            logger.error(f"Fatigue trend calculation failed: {e}")
            return 'Unknown'
    
    def is_microsleep_detected(self) -> bool:
        """Detect microsleep episodes (very brief eye closures)"""
        try:
            if len(self.ear_history) < 5:
                return False
            
            recent_ears = list(self.ear_history)[-5:]
            
            # Look for pattern: open -> closed -> open in short time
            microsleep_pattern = False
            for i in range(1, len(recent_ears) - 1):
                if (recent_ears[i-1] > self.ear_threshold and 
                    recent_ears[i] < self.ear_threshold * 0.7 and 
                    recent_ears[i+1] > self.ear_threshold):
                    microsleep_pattern = True
                    break
            
            return microsleep_pattern
            
        except Exception as e:
            logger.error(f"Microsleep detection failed: {e}")
            return False
    
    def get_fatigue_statistics(self) -> Dict:
        """Get comprehensive fatigue statistics"""
        try:
            stats = {
                'total_samples': len(self.ear_history),
                'avg_ear': 0.0,
                'ear_variance': 0.0,
                'avg_perclos': 0.0,
                'yawn_episodes': 0,
                'microsleep_detected': False
            }
            
            if len(self.ear_history) > 0:
                ear_values = list(self.ear_history)
                stats['avg_ear'] = round(np.mean(ear_values), 3)
                stats['ear_variance'] = round(np.var(ear_values), 4)
            
            if len(self.perclos_history) > 0:
                stats['avg_perclos'] = round(np.mean(list(self.perclos_history)), 3)
            
            if len(self.yawn_history) > 0:
                stats['yawn_episodes'] = sum(self.yawn_history)
            
            stats['microsleep_detected'] = self.is_microsleep_detected()
            
            return stats
            
        except Exception as e:
            logger.error(f"Fatigue statistics calculation failed: {e}")
            return {'total_samples': 0, 'avg_ear': 0.0, 'ear_variance': 0.0}
    
    def reset_history(self):
        """Reset all fatigue analysis history"""
        self.ear_history.clear()
        self.perclos_history.clear()
        self.yawn_history.clear()
        self.head_pose_history.clear()
        logger.info("Fatigue detector history reset")
    
    def calibrate_thresholds(self, baseline_ear: float):
        """Calibrate detection thresholds based on individual baseline"""
        try:
            if baseline_ear > 0:
                # Adjust EAR threshold based on individual's normal eye opening
                self.ear_threshold = baseline_ear * 0.8  # 80% of normal opening
                logger.info(f"EAR threshold calibrated to {self.ear_threshold:.3f}")
        except Exception as e:
            logger.error(f"Threshold calibration failed: {e}")