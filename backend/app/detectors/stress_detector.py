import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class StressDetector:
    def __init__(self):
        """Initialize stress detector using multiple physiological indicators"""
        self.stress_history = deque(maxlen=20)  # Store recent stress assessments
        
        # Stress level thresholds
        self.stress_levels = {
            'Very Low': (0.0, 0.2),
            'Low': (0.2, 0.4),
            'Moderate': (0.4, 0.6),
            'High': (0.6, 0.8),
            'Very High': (0.8, 1.0)
        }
        
        # Physiological stress indicators
        self.stress_indicators = [
            'heart_rate_variability',
            'facial_muscle_tension',
            'eye_blink_rate',
            'micro_expressions',
            'skin_color_changes',
            'breathing_pattern'
        ]
        
        logger.info("Stress detector initialized with multi-modal analysis")
    
    def analyze_facial_muscle_tension(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> float:
        """Analyze facial muscle tension as stress indicator"""
        try:
            if face_roi is None or face_roi.size == 0:
                return 0.0
            
            tension_score = 0.0
            
            if landmarks is not None and len(landmarks) >= 68:
                # Analyze jaw tension (distance between jaw points)
                jaw_points = landmarks[5:12]  # Lower face contour
                if len(jaw_points) > 0:
                    jaw_width = np.linalg.norm(landmarks[3] - landmarks[13])  # Jaw width
                    jaw_tension = self._calculate_jaw_tension(jaw_points, jaw_width)
                    tension_score += jaw_tension * 0.3
                
                # Analyze forehead tension (eyebrow position and wrinkles)
                eyebrow_points = landmarks[17:27]
                forehead_tension = self._analyze_forehead_tension(face_roi, eyebrow_points)
                tension_score += forehead_tension * 0.25
                
                # Analyze eye region tension
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                eye_tension = self._analyze_eye_tension(left_eye, right_eye)
                tension_score += eye_tension * 0.25
                
                # Analyze mouth tension
                mouth_points = landmarks[48:68]
                mouth_tension = self._analyze_mouth_tension(mouth_points)
                tension_score += mouth_tension * 0.2
            else:
                # Fallback: image-based tension analysis
                tension_score = self._analyze_tension_without_landmarks(face_roi)
            
            return min(tension_score, 1.0)
            
        except Exception as e:
            logger.error(f"Facial muscle tension analysis failed: {e}")
            return 0.0
    
    def _calculate_jaw_tension(self, jaw_points: np.ndarray, jaw_width: float) -> float:
        """Calculate jaw muscle tension"""
        try:
            # Analyze jaw line straightness (tension creates straighter jaw line)
            jaw_curve = self._calculate_curve_deviation(jaw_points)
            
            # Normalize based on jaw width
            tension_indicator = jaw_curve / (jaw_width + 1e-8)
            
            return min(tension_indicator / 10.0, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Jaw tension calculation failed: {e}")
            return 0.0
    
    def _analyze_forehead_tension(self, face_roi: np.ndarray, eyebrow_points: np.ndarray) -> float:
        """Analyze forehead wrinkles and eyebrow position for tension"""
        try:
            # Extract forehead region
            if len(eyebrow_points) > 0:
                x_min, y_min = np.min(eyebrow_points, axis=0)
                x_max, y_max = np.max(eyebrow_points, axis=0)
                
                # Expand upward for forehead
                forehead_y = max(0, int(y_min - 40))
                forehead_roi = face_roi[forehead_y:int(y_max), int(x_min):int(x_max)]
            else:
                # Fallback
                h, w = face_roi.shape[:2]
                forehead_roi = face_roi[0:h//4, w//4:3*w//4]
            
            if forehead_roi.size == 0:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(forehead_roi, cv2.COLOR_BGR2GRAY) if len(forehead_roi.shape) == 3 else forehead_roi
            
            # Detect wrinkles using edge detection
            edges = cv2.Canny(gray, 50, 150)
            wrinkle_density = np.sum(edges > 0) / edges.size
            
            # Analyze texture (wrinkles create texture variations)
            texture_var = np.var(gray)
            texture_score = min(texture_var / 500, 1.0)
            
            tension_score = wrinkle_density * 0.6 + texture_score * 0.4
            
            return min(tension_score, 1.0)
            
        except Exception as e:
            logger.error(f"Forehead tension analysis failed: {e}")
            return 0.0
    
    def _analyze_eye_tension(self, left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """Analyze eye region for tension indicators"""
        try:
            tension_indicators = []
            
            # Calculate eye aspect ratio (stressed individuals may squint)
            left_ear = self._calculate_eye_aspect_ratio(left_eye)
            right_ear = self._calculate_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            
            # Lower EAR might indicate squinting/tension
            if avg_ear < 0.18:
                tension_indicators.append(0.3)
            
            # Analyze eye symmetry (stress can cause asymmetric expressions)
            eye_asymmetry = abs(left_ear - right_ear)
            tension_indicators.append(min(eye_asymmetry / 0.1, 0.4))
            
            return sum(tension_indicators)
            
        except Exception as e:
            logger.error(f"Eye tension analysis failed: {e}")
            return 0.0
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio"""
        try:
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            ear = (A + B) / (2.0 * C)
            return float(ear)
        except:
            return 0.0
    
    def _analyze_mouth_tension(self, mouth_points: np.ndarray) -> float:
        """Analyze mouth region for tension"""
        try:
            # Calculate lip compression (tight lips indicate stress)
            upper_lip = mouth_points[13:16]  # Upper lip points
            lower_lip = mouth_points[1:4]    # Lower lip points
            
            lip_distance = np.mean([np.linalg.norm(upper_lip[i] - lower_lip[i]) 
                                   for i in range(len(upper_lip))])
            
            # Mouth width for normalization
            mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[6])
            
            # Normalized lip compression
            compression_ratio = lip_distance / (mouth_width + 1e-8)
            
            # Lower ratio indicates more compression (tension)
            tension_score = max(0, 0.5 - compression_ratio * 5)
            
            return min(tension_score, 1.0)
            
        except Exception as e:
            logger.error(f"Mouth tension analysis failed: {e}")
            return 0.0
    
    def _calculate_curve_deviation(self, points: np.ndarray) -> float:
        """Calculate how much a set of points deviates from a straight line"""
        try:
            if len(points) < 3:
                return 0.0
            
            # Fit a line through the points
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Calculate line of best fit
            coeffs = np.polyfit(x_coords, y_coords, 1)
            line_y = np.polyval(coeffs, x_coords)
            
            # Calculate deviation from line
            deviations = np.abs(y_coords - line_y)
            mean_deviation = np.mean(deviations)
            
            return float(mean_deviation)
            
        except Exception as e:
            logger.error(f"Curve deviation calculation failed: {e}")
            return 0.0
    
    def _analyze_tension_without_landmarks(self, face_roi: np.ndarray) -> float:
        """Analyze tension using image processing when landmarks unavailable"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # Texture analysis - stressed faces may have more texture variations
            texture_score = np.var(gray) / 1000
            
            # Edge density - tension may create more defined facial features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Combine metrics
            tension_score = texture_score * 0.6 + edge_density * 0.4
            
            return min(tension_score, 1.0)
            
        except Exception as e:
            logger.error(f"Image-based tension analysis failed: {e}")
            return 0.0
    
    def analyze_blink_patterns(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Analyze eye blink patterns for stress indicators"""
        try:
            blink_analysis = {
                'blink_rate': 0.0,
                'blink_duration': 0.0,
                'stress_indicator': 0.0
            }
            
            if landmarks is not None and len(landmarks) >= 68:
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                
                # Calculate eye aspect ratios
                left_ear = self._calculate_eye_aspect_ratio(left_eye)
                right_ear = self._calculate_eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2
                
                # Stressed individuals often have increased blink rate
                # EAR < 0.15 typically indicates closed/nearly closed eyes
                if avg_ear < 0.15:
                    blink_analysis['blink_duration'] = 1.0 - avg_ear / 0.15
                
                # Rapid blinking indicator (based on EAR variability would need temporal analysis)
                blink_analysis['blink_rate'] = max(0, 1.0 - avg_ear / 0.2)
                
                # Stress indicator based on abnormal blinking
                if avg_ear < 0.12 or avg_ear > 0.35:  # Too closed or too wide
                    blink_analysis['stress_indicator'] = 0.6
                elif avg_ear < 0.15 or avg_ear > 0.3:
                    blink_analysis['stress_indicator'] = 0.3
            
            return blink_analysis
            
        except Exception as e:
            logger.error(f"Blink pattern analysis failed: {e}")
            return {'blink_rate': 0.0, 'blink_duration': 0.0, 'stress_indicator': 0.0}
    
    def analyze_skin_color_changes(self, face_roi: np.ndarray) -> float:
        """Analyze skin color changes that may indicate stress"""
        try:
            if face_roi is None or face_roi.size == 0:
                return 0.0
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            
            # Analyze saturation (stress can cause flushing or pallor)
            saturation = hsv[:, :, 1]
            mean_saturation = np.mean(saturation)
            saturation_var = np.var(saturation)
            
            # Analyze luminance variations
            luminance = lab[:, :, 0]
            luminance_var = np.var(luminance)
            
            # Stress indicators:
            # 1. High saturation variance (uneven skin tone)
            # 2. Extreme mean saturation (too flushed or too pale)
            # 3. High luminance variance (uneven lighting/skin)
            
            saturation_stress = min(saturation_var / 500, 0.4)
            luminance_stress = min(luminance_var / 300, 0.3)
            
            # Extreme saturation values
            if mean_saturation > 120 or mean_saturation < 30:
                extreme_stress = 0.3
            else:
                extreme_stress = 0.0
            
            total_stress = saturation_stress + luminance_stress + extreme_stress
            
            return min(total_stress, 1.0)
            
        except Exception as e:
            logger.error(f"Skin color analysis failed: {e}")
            return 0.0
    
    def analyze_micro_expressions(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Detect micro-expressions that may indicate stress"""
        try:
            micro_expressions = {
                'lip_tightening': 0.0,
                'eyebrow_flash': 0.0,
                'nostril_flaring': 0.0,
                'jaw_clenching': 0.0
            }
            
            if landmarks is not None and len(landmarks) >= 68:
                # Lip tightening analysis
                mouth_points = landmarks[48:68]
                lip_tightening = self._detect_lip_tightening(mouth_points)
                micro_expressions['lip_tightening'] = lip_tightening
                
                # Eyebrow position analysis
                eyebrow_points = landmarks[17:27]
                eyebrow_tension = self._detect_eyebrow_tension(eyebrow_points)
                micro_expressions['eyebrow_flash'] = eyebrow_tension
                
                # Nostril analysis
                nose_points = landmarks[31:36]  # Nostril area
                nostril_flaring = self._detect_nostril_flaring(nose_points)
                micro_expressions['nostril_flaring'] = nostril_flaring
                
                # Jaw analysis
                jaw_points = landmarks[5:12]
                jaw_tension = self._detect_jaw_clenching(jaw_points)
                micro_expressions['jaw_clenching'] = jaw_tension
            
            return micro_expressions
            
        except Exception as e:
            logger.error(f"Micro-expression analysis failed: {e}")
            return {'lip_tightening': 0.0, 'eyebrow_flash': 0.0, 'nostril_flaring': 0.0, 'jaw_clenching': 0.0}
    
    def _detect_lip_tightening(self, mouth_points: np.ndarray) -> float:
        """Detect lip tightening micro-expression"""
        try:
            # Calculate lip thickness
            upper_lip_thickness = np.mean([
                np.linalg.norm(mouth_points[13] - mouth_points[14]),
                np.linalg.norm(mouth_points[14] - mouth_points[15]),
                np.linalg.norm(mouth_points[15] - mouth_points[16])
            ])
            
            lower_lip_thickness = np.mean([
                np.linalg.norm(mouth_points[10] - mouth_points[11]),
                np.linalg.norm(mouth_points[11] - mouth_points[12]),
                np.linalg.norm(mouth_points[9] - mouth_points[10])
            ])
            
            mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[6])
            
            # Normalized lip thickness
            lip_ratio = (upper_lip_thickness + lower_lip_thickness) / (mouth_width + 1e-8)
            
            # Tighter lips have lower ratio
            tightening_score = max(0, 0.3 - lip_ratio * 10)
            
            return min(tightening_score, 1.0)
            
        except Exception as e:
            logger.error(f"Lip tightening detection failed: {e}")
            return 0.0
    
    def _detect_eyebrow_tension(self, eyebrow_points: np.ndarray) -> float:
        """Detect eyebrow tension/flash"""
        try:
            left_brow = eyebrow_points[:5]
            right_brow = eyebrow_points[5:]
            
            # Calculate eyebrow arch height
            left_arch = np.max(left_brow[:, 1]) - np.min(left_brow[:, 1])
            right_arch = np.max(right_brow[:, 1]) - np.min(right_brow[:, 1])
            
            avg_arch = (left_arch + right_arch) / 2
            
            # Higher arch might indicate raised eyebrows (stress/surprise)
            tension_score = min(avg_arch / 20, 1.0)
            
            return tension_score
            
        except Exception as e:
            logger.error(f"Eyebrow tension detection failed: {e}")
            return 0.0
    
    def _detect_nostril_flaring(self, nose_points: np.ndarray) -> float:
        """Detect nostril flaring"""
        try:
            if len(nose_points) < 5:
                return 0.0
            
            # Calculate nostril width
            left_nostril = nose_points[0]   # Left nostril
            right_nostril = nose_points[4]  # Right nostril
            nostril_width = np.linalg.norm(right_nostril - left_nostril)
            
            # Nose bridge width for normalization
            nose_bridge_width = np.linalg.norm(nose_points[1] - nose_points[3])
            
            # Flaring ratio
            flaring_ratio = nostril_width / (nose_bridge_width + 1e-8)
            
            # Higher ratio indicates flaring
            flaring_score = max(0, flaring_ratio - 1.2) * 2
            
            return min(flaring_score, 1.0)
            
        except Exception as e:
            logger.error(f"Nostril flaring detection failed: {e}")
            return 0.0
    
    def _detect_jaw_clenching(self, jaw_points: np.ndarray) -> float:
        """Detect jaw clenching"""
        try:
            # Calculate jaw muscle definition (clenching creates more defined jaw line)
            jaw_definition = self._calculate_curve_deviation(jaw_points)
            
            # Jaw width
            jaw_width = np.linalg.norm(jaw_points[0] - jaw_points[-1])
            
            # Normalized definition
            definition_ratio = jaw_definition / (jaw_width + 1e-8)
            
            clenching_score = min(definition_ratio * 5, 1.0)
            
            return clenching_score
            
        except Exception as e:
            logger.error(f"Jaw clenching detection failed: {e}")
            return 0.0
    
    def analyze_stress_indicators(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None, 
                                health_metrics: Dict = None) -> Dict:
        """Comprehensive stress analysis using multiple indicators"""
        try:
            stress_scores = {}
            
            # 1. Facial muscle tension
            tension_score = self.analyze_facial_muscle_tension(face_roi, landmarks)
            stress_scores['facial_tension'] = tension_score
            
            # 2. Blink pattern analysis
            blink_analysis = self.analyze_blink_patterns(face_roi, landmarks)
            stress_scores['blink_patterns'] = blink_analysis['stress_indicator']
            
            # 3. Skin color changes
            skin_stress = self.analyze_skin_color_changes(face_roi)
            stress_scores['skin_changes'] = skin_stress
            
            # 4. Micro-expressions
            micro_expr = self.analyze_micro_expressions(face_roi, landmarks)
            micro_stress = np.mean(list(micro_expr.values()))
            stress_scores['micro_expressions'] = micro_stress
            
            # 5. Heart rate variability (if available)
            if health_metrics and 'heartRateVariability' in health_metrics:
                try:
                    hrv = float(health_metrics['heartRateVariability'])
                    # Lower HRV often indicates higher stress
                    hrv_stress = max(0, 1 - hrv)
                    stress_scores['heart_rate_variability'] = hrv_stress
                except:
                    stress_scores['heart_rate_variability'] = 0.0
            
            # Calculate overall stress level
            stress_weights = {
                'facial_tension': 0.25,
                'blink_patterns': 0.15,
                'skin_changes': 0.20,
                'micro_expressions': 0.25,
                'heart_rate_variability': 0.15
            }
            
            weighted_stress = sum(
                stress_scores.get(indicator, 0.0) * weight 
                for indicator, weight in stress_weights.items()
            )
            
            # Determine stress level category
            stress_level = self._categorize_stress_level(weighted_stress)
            
            # Identify primary stress factors
            stress_factors = self._identify_stress_factors(stress_scores, micro_expr)
            
            # Add to history for trending
            self.stress_history.append(weighted_stress)
            
            # Calculate trend
            stress_trend = self._calculate_stress_trend()
            
            result = {
                'stress_level': stress_level,
                'stress_score': round(weighted_stress, 3),
                'factors': stress_factors,
                'detailed_scores': {k: round(v, 3) for k, v in stress_scores.items()},
                'micro_expressions': micro_expr,
                'blink_analysis': blink_analysis,
                'trend': stress_trend,
                'confidence': min(len(self.stress_history) / 10, 1.0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Stress analysis failed: {e}")
            return {
                'stress_level': 'Error',
                'stress_score': 0.0,
                'factors': [],
                'detailed_scores': {},
                'trend': 'Unknown'
            }
    
    def _categorize_stress_level(self, stress_score: float) -> str:
        """Categorize stress level based on score"""
        for level, (min_val, max_val) in self.stress_levels.items():
            if min_val <= stress_score < max_val:
                return level
        return 'Very High'
    
    def _identify_stress_factors(self, stress_scores: Dict, micro_expressions: Dict) -> List[str]:
        """Identify primary contributing stress factors"""
        factors = []
        
        # Check each stress indicator
        if stress_scores.get('facial_tension', 0) > 0.4:
            factors.append('Facial muscle tension')
        
        if stress_scores.get('blink_patterns', 0) > 0.3:
            factors.append('Abnormal blinking')
        
        if stress_scores.get('skin_changes', 0) > 0.3:
            factors.append('Skin color variations')
        
        if stress_scores.get('heart_rate_variability', 0) > 0.5:
            factors.append('Low heart rate variability')
        
        # Check micro-expressions
        for expr, score in micro_expressions.items():
            if score > 0.4:
                factors.append(f'{expr.replace("_", " ").title()}')
        
        return factors
    
    def _calculate_stress_trend(self) -> str:
        """Calculate stress trend from recent history"""
        try:
            if len(self.stress_history) < 5:
                return 'Insufficient data'
            
            recent_scores = list(self.stress_history)[-5:]
            early_avg = np.mean(recent_scores[:2])
            recent_avg = np.mean(recent_scores[-2:])
            
            diff = recent_avg - early_avg
            
            if diff > 0.1:
                return 'Increasing'
            elif diff < -0.1:
                return 'Decreasing'
            else:
                return 'Stable'
                
        except Exception as e:
            logger.error(f"Stress trend calculation failed: {e}")
            return 'Unknown'
    
    def quick_stress_assessment(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> str:
        """Quick stress assessment for rapid scanning"""
        try:
            # Simple stress indicators
            tension = self.analyze_facial_muscle_tension(face_roi, landmarks)
            skin_stress = self.analyze_skin_color_changes(face_roi)
            
            avg_stress = (tension + skin_stress) / 2
            
            return self._categorize_stress_level(avg_stress)
            
        except Exception as e:
            logger.error(f"Quick stress assessment failed: {e}")
            return 'Unknown'
    
    def reset_history(self):
        """Reset stress analysis history"""
        self.stress_history.clear()
        logger.info("Stress detector history reset")