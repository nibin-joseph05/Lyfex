import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class NeurologicalDetector:
    """
    Neurological health assessment through facial analysis
    Detects potential neurological issues through facial asymmetry, tremor detection, and eye movement analysis
    """
    
    def __init__(self):
        self.previous_landmarks = None
        self.landmark_history = []
        self.max_history_size = 30
        self.asymmetry_threshold = 0.15
        self.tremor_threshold = 2.0
        
        logger.info("Neurological Detector initialized")
    
    def assess_neurological_health(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray]) -> Dict:
        """
        Comprehensive neurological assessment
        
        Args:
            face_roi: Face region of interest
            landmarks: Facial landmarks array
            
        Returns:
            Dictionary containing neurological assessment results
        """
        try:
            if landmarks is None:
                return self._get_error_response("No facial landmarks detected")
            
            # Store landmark history for temporal analysis
            self._update_landmark_history(landmarks)
            
            # Calculate facial asymmetry
            asymmetry_score = self._calculate_facial_asymmetry(landmarks)
            
            # Detect tremor patterns
            tremor_detected, tremor_intensity = self._detect_tremor()
            
            # Analyze eye movement patterns
            eye_movement_analysis = self._analyze_eye_movement(landmarks)
            
            # Assess muscle tone indicators
            muscle_tone = self._assess_muscle_tone(landmarks)
            
            # Calculate overall neurological risk score
            risk_score = self._calculate_neurological_risk(
                asymmetry_score, tremor_intensity, eye_movement_analysis, muscle_tone
            )
            
            return {
                'facial_asymmetry': self._interpret_asymmetry_score(asymmetry_score),
                'asymmetry_score': round(asymmetry_score, 3),
                'tremor_detected': tremor_detected,
                'tremor_intensity': round(tremor_intensity, 3),
                'eye_movement_analysis': eye_movement_analysis,
                'muscle_tone_indicators': muscle_tone,
                'neurological_risk_score': round(risk_score, 3),
                'risk_level': self._interpret_risk_level(risk_score),
                'recommendations': self._generate_recommendations(risk_score, tremor_detected, asymmetry_score)
            }
            
        except Exception as e:
            logger.error(f"Neurological assessment failed: {e}")
            return self._get_error_response(f"Analysis failed: {str(e)}")
    
    def _update_landmark_history(self, landmarks: np.ndarray):
        """Update landmark history for temporal analysis"""
        self.landmark_history.append(landmarks.copy())
        if len(self.landmark_history) > self.max_history_size:
            self.landmark_history.pop(0)
        self.previous_landmarks = landmarks
    
    def _calculate_facial_asymmetry(self, landmarks: np.ndarray) -> float:
        """
        Calculate facial asymmetry score
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Asymmetry score (0 = perfectly symmetric, 1 = highly asymmetric)
        """
        try:
            if len(landmarks) < 68:  # Standard dlib 68-point landmarks
                return 0.0
            
            # Get key facial points
            left_eye = landmarks[36:42]  # Left eye landmarks
            right_eye = landmarks[42:48]  # Right eye landmarks
            nose_tip = landmarks[30]
            left_mouth = landmarks[48]
            right_mouth = landmarks[54]
            
            # Calculate face center line
            face_center_x = (landmarks[0][0] + landmarks[16][0]) / 2
            
            # Calculate asymmetry for eyes
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            
            # Mirror right eye across face center
            right_eye_mirrored_x = 2 * face_center_x - right_eye_center[0]
            eye_asymmetry = abs(left_eye_center[0] - right_eye_mirrored_x) / (landmarks[16][0] - landmarks[0][0])
            
            # Calculate asymmetry for mouth
            mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
            mouth_asymmetry = abs(mouth_center_x - face_center_x) / (landmarks[16][0] - landmarks[0][0])
            
            # Calculate nose deviation
            nose_asymmetry = abs(nose_tip[0] - face_center_x) / (landmarks[16][0] - landmarks[0][0])
            
            # Combined asymmetry score
            total_asymmetry = (eye_asymmetry + mouth_asymmetry + nose_asymmetry) / 3
            
            return min(total_asymmetry, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Facial asymmetry calculation error: {e}")
            return 0.0
    
    def _detect_tremor(self) -> Tuple[bool, float]:
        """
        Detect tremor patterns from landmark movement history
        
        Returns:
            Tuple of (tremor_detected, tremor_intensity)
        """
        try:
            if len(self.landmark_history) < 10:
                return False, 0.0
            
            # Calculate movement variance for key points
            recent_landmarks = np.array(self.landmark_history[-10:])
            
            # Focus on stable points that should have minimal movement
            # Nose tip, chin, and forehead points
            stable_points = [30, 8, 27]  # Nose tip, chin, forehead center
            
            movement_variances = []
            for point_idx in stable_points:
                if point_idx < recent_landmarks.shape[1]:
                    point_movements = recent_landmarks[:, point_idx, :]
                    variance = np.var(point_movements, axis=0)
                    movement_variances.append(np.mean(variance))
            
            if not movement_variances:
                return False, 0.0
            
            avg_variance = np.mean(movement_variances)
            tremor_intensity = min(avg_variance / 10.0, 1.0)  # Normalize to 0-1 range
            
            tremor_detected = avg_variance > self.tremor_threshold
            
            return tremor_detected, tremor_intensity
            
        except Exception as e:
            logger.error(f"Tremor detection error: {e}")
            return False, 0.0
    
    def _analyze_eye_movement(self, landmarks: np.ndarray) -> Dict:
        """
        Analyze eye movement patterns
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Dictionary with eye movement analysis
        """
        try:
            if len(landmarks) < 68:
                return {'status': 'insufficient_data', 'symmetry': 'unknown', 'alertness': 'unknown'}
            
            # Eye landmarks
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Calculate eye openness (vertical distance)
            left_eye_openness = self._calculate_eye_openness(left_eye)
            right_eye_openness = self._calculate_eye_openness(right_eye)
            
            # Eye symmetry
            eye_symmetry_ratio = min(left_eye_openness, right_eye_openness) / max(left_eye_openness, right_eye_openness)
            
            # Alertness based on eye openness
            avg_openness = (left_eye_openness + right_eye_openness) / 2
            alertness_level = 'high' if avg_openness > 0.3 else 'medium' if avg_openness > 0.15 else 'low'
            
            # Eye symmetry assessment
            symmetry_status = 'symmetric' if eye_symmetry_ratio > 0.8 else 'asymmetric'
            
            return {
                'status': 'analyzed',
                'left_eye_openness': round(left_eye_openness, 3),
                'right_eye_openness': round(right_eye_openness, 3),
                'symmetry': symmetry_status,
                'symmetry_ratio': round(eye_symmetry_ratio, 3),
                'alertness': alertness_level,
                'average_openness': round(avg_openness, 3)
            }
            
        except Exception as e:
            logger.error(f"Eye movement analysis error: {e}")
            return {'status': 'error', 'symmetry': 'unknown', 'alertness': 'unknown'}
    
    def _calculate_eye_openness(self, eye_landmarks: np.ndarray) -> float:
        """Calculate eye openness ratio"""
        try:
            # Vertical distances
            upper_lower_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            upper_lower_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Horizontal distance
            horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Eye aspect ratio
            eye_openness = (upper_lower_1 + upper_lower_2) / (2.0 * horizontal)
            
            return eye_openness
            
        except Exception as e:
            logger.error(f"Eye openness calculation error: {e}")
            return 0.0
    
    def _assess_muscle_tone(self, landmarks: np.ndarray) -> Dict:
        """
        Assess facial muscle tone indicators
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Dictionary with muscle tone assessment
        """
        try:
            if len(landmarks) < 68:
                return {'status': 'insufficient_data', 'tone': 'unknown'}
            
            # Analyze mouth corner positions (indicators of facial muscle tone)
            mouth_left = landmarks[48]
            mouth_right = landmarks[54]
            mouth_center = landmarks[51]
            
            # Calculate mouth symmetry
            left_distance = np.linalg.norm(mouth_left - mouth_center)
            right_distance = np.linalg.norm(mouth_right - mouth_center)
            
            mouth_symmetry = min(left_distance, right_distance) / max(left_distance, right_distance)
            
            # Assess forehead tension (distance between eyebrows and hairline)
            eyebrow_center = landmarks[27]
            forehead_point = landmarks[27]  # Approximate forehead point
            
            # Simple muscle tone assessment based on facial feature positions
            tone_score = (mouth_symmetry + 0.5) / 1.5  # Normalize
            
            tone_level = 'good' if tone_score > 0.8 else 'moderate' if tone_score > 0.6 else 'concern'
            
            return {
                'status': 'assessed',
                'tone': tone_level,
                'tone_score': round(tone_score, 3),
                'mouth_symmetry': round(mouth_symmetry, 3),
                'indicators': {
                    'mouth_alignment': 'good' if mouth_symmetry > 0.8 else 'concern',
                    'overall_symmetry': 'good' if mouth_symmetry > 0.85 else 'monitor'
                }
            }
            
        except Exception as e:
            logger.error(f"Muscle tone assessment error: {e}")
            return {'status': 'error', 'tone': 'unknown'}
    
    def _calculate_neurological_risk(self, asymmetry: float, tremor: float, 
                                   eye_analysis: Dict, muscle_tone: Dict) -> float:
        """Calculate overall neurological risk score"""
        try:
            # Weight factors for different indicators
            asymmetry_weight = 0.3
            tremor_weight = 0.25
            eye_weight = 0.25
            muscle_weight = 0.2
            
            # Normalize asymmetry score
            asymmetry_risk = min(asymmetry / self.asymmetry_threshold, 1.0)
            
            # Tremor risk
            tremor_risk = tremor
            
            # Eye movement risk
            eye_risk = 0.0
            if eye_analysis.get('symmetry') == 'asymmetric':
                eye_risk += 0.5
            if eye_analysis.get('alertness') == 'low':
                eye_risk += 0.5
            eye_risk = min(eye_risk, 1.0)
            
            # Muscle tone risk
            muscle_risk = 0.0
            if muscle_tone.get('tone') == 'concern':
                muscle_risk = 0.8
            elif muscle_tone.get('tone') == 'moderate':
                muscle_risk = 0.4
            
            # Calculate weighted risk score
            total_risk = (
                asymmetry_risk * asymmetry_weight +
                tremor_risk * tremor_weight +
                eye_risk * eye_weight +
                muscle_risk * muscle_weight
            )
            
            return min(total_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Risk calculation error: {e}")
            return 0.0
    
    def _interpret_asymmetry_score(self, score: float) -> str:
        """Interpret asymmetry score"""
        if score < 0.05:
            return "Highly Symmetric"
        elif score < 0.10:
            return "Symmetric"
        elif score < 0.15:
            return "Mild Asymmetry"
        elif score < 0.25:
            return "Moderate Asymmetry"
        else:
            return "Significant Asymmetry"
    
    def _interpret_risk_level(self, risk_score: float) -> str:
        """Interpret neurological risk level"""
        if risk_score < 0.2:
            return "Low Risk"
        elif risk_score < 0.4:
            return "Mild Concern"
        elif risk_score < 0.6:
            return "Moderate Concern"
        elif risk_score < 0.8:
            return "High Concern"
        else:
            return "Significant Concern"
    
    def _generate_recommendations(self, risk_score: float, tremor_detected: bool, 
                                asymmetry_score: float) -> List[str]:
        """Generate neurological health recommendations"""
        recommendations = []
        
        if risk_score < 0.2:
            recommendations.append("Neurological indicators appear normal")
        
        if asymmetry_score > 0.15:
            recommendations.append("Monitor facial symmetry - consider professional evaluation if persistent")
        
        if tremor_detected:
            recommendations.append("Tremor detected - reduce caffeine and stress, consult healthcare provider if persistent")
        
        if risk_score > 0.6:
            recommendations.append("Multiple neurological indicators detected - recommend professional medical evaluation")
            recommendations.append("Maintain regular sleep schedule and manage stress levels")
        
        if risk_score > 0.4:
            recommendations.append("Consider regular neurological health monitoring")
        
        # General recommendations
        recommendations.extend([
            "Maintain regular physical exercise for neurological health",
            "Ensure adequate sleep (7-9 hours) for optimal brain function",
            "Practice stress management techniques"
        ])
        
        return recommendations
    
    def _get_error_response(self, error_message: str) -> Dict:
        """Return error response format"""
        return {
            'facial_asymmetry': 'Analysis Error',
            'asymmetry_score': 0.0,
            'tremor_detected': False,
            'tremor_intensity': 0.0,
            'eye_movement_analysis': {'status': 'error'},
            'muscle_tone_indicators': {'status': 'error'},
            'neurological_risk_score': 0.0,
            'risk_level': 'Unknown',
            'recommendations': [f"Analysis failed: {error_message}"]
        }
    
    def quick_neurological_check(self, landmarks: Optional[np.ndarray]) -> Dict:
        """
        Quick neurological screening
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Basic neurological indicators
        """
        try:
            if landmarks is None:
                return {'status': 'no_landmarks', 'asymmetry': 'unknown'}
            
            asymmetry_score = self._calculate_facial_asymmetry(landmarks)
            
            return {
                'status': 'analyzed',
                'asymmetry': self._interpret_asymmetry_score(asymmetry_score),
                'asymmetry_score': round(asymmetry_score, 3),
                'quick_assessment': 'normal' if asymmetry_score < 0.15 else 'monitor'
            }
            
        except Exception as e:
            logger.error(f"Quick neurological check error: {e}")
            return {'status': 'error', 'asymmetry': 'unknown'}