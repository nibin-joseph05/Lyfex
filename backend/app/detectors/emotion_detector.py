import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        """Initialize emotion detector using facial feature analysis"""
        self.emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
        self.emotion_history = deque(maxlen=10)  # Store recent emotions for stability
        
        # Facial feature regions for emotion detection
        self.feature_regions = {
            'eyes': {'landmarks': list(range(36, 48))},  # Eye region landmarks
            'eyebrows': {'landmarks': list(range(17, 27))},  # Eyebrow landmarks  
            'mouth': {'landmarks': list(range(48, 68))},  # Mouth region landmarks
            'nose': {'landmarks': list(range(27, 36))}  # Nose region landmarks
        }
        
        logger.info("Emotion detector initialized with feature-based analysis")
    
    def extract_facial_features(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Extract key facial features for emotion analysis"""
        features = {
            'eye_aspect_ratio': 0.0,
            'mouth_aspect_ratio': 0.0,
            'eyebrow_height': 0.0,
            'mouth_curvature': 0.0,
            'eye_openness': 0.0,
            'facial_symmetry': 0.0
        }
        
        try:
            if landmarks is not None and len(landmarks) >= 68:
                # Eye Aspect Ratio (EAR) - indicates eye openness/closure
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                
                left_ear = self._calculate_eye_aspect_ratio(left_eye)
                right_ear = self._calculate_eye_aspect_ratio(right_eye)
                features['eye_aspect_ratio'] = (left_ear + right_ear) / 2
                features['eye_openness'] = features['eye_aspect_ratio']
                
                # Mouth Aspect Ratio (MAR) - indicates mouth openness
                mouth_landmarks = landmarks[48:68]
                features['mouth_aspect_ratio'] = self._calculate_mouth_aspect_ratio(mouth_landmarks)
                
                # Eyebrow height - indicates surprise/concern
                left_eyebrow = landmarks[17:22]
                right_eyebrow = landmarks[22:27]
                left_eye_center = np.mean(left_eye, axis=0)
                right_eye_center = np.mean(right_eye, axis=0)
                
                left_brow_height = np.mean(left_eyebrow[:, 1]) - left_eye_center[1]
                right_brow_height = np.mean(right_eyebrow[:, 1]) - right_eye_center[1]
                features['eyebrow_height'] = (left_brow_height + right_brow_height) / 2
                
                # Mouth curvature - indicates smile/frown
                features['mouth_curvature'] = self._calculate_mouth_curvature(mouth_landmarks)
                
                # Facial symmetry
                features['facial_symmetry'] = self._calculate_facial_symmetry(landmarks)
                
            else:
                # Fallback: use image-based feature extraction
                features.update(self._extract_image_based_features(face_roi))
                
        except Exception as e:
            logger.error(f"Facial feature extraction failed: {e}")
        
        return features
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
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
    
    def _calculate_mouth_aspect_ratio(self, mouth_landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio (MAR)"""
        try:
            # Vertical distances
            A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])  # 50-58
            B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])   # 52-56
            
            # Horizontal distance  
            C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])   # 48-54
            
            # Mouth aspect ratio
            mar = (A + B) / (2.0 * C)
            return float(mar)
            
        except Exception as e:
            logger.error(f"MAR calculation failed: {e}")
            return 0.0
    
    def _calculate_mouth_curvature(self, mouth_landmarks: np.ndarray) -> float:
        """Calculate mouth curvature (smile/frown indicator)"""
        try:
            # Mouth corners
            left_corner = mouth_landmarks[0]   # Point 48
            right_corner = mouth_landmarks[6]  # Point 54
            
            # Upper and lower lip center points
            upper_lip_center = mouth_landmarks[3]  # Point 51
            lower_lip_center = mouth_landmarks[9]  # Point 57
            
            # Calculate mouth center
            mouth_center = (upper_lip_center + lower_lip_center) / 2
            
            # Calculate corner heights relative to mouth center
            left_height = left_corner[1] - mouth_center[1]
            right_height = right_corner[1] - mouth_center[1]
            
            # Average curvature (negative = smile, positive = frown)
            curvature = (left_height + right_height) / 2
            return float(-curvature)  # Invert so positive = smile
            
        except Exception as e:
            logger.error(f"Mouth curvature calculation failed: {e}")
            return 0.0
    
    def _calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        try:
            # Get face center line
            nose_tip = landmarks[30]
            chin = landmarks[8]
            face_center_x = (nose_tip[0] + chin[0]) / 2
            
            # Compare left and right features
            left_eye_center = np.mean(landmarks[36:42], axis=0)
            right_eye_center = np.mean(landmarks[42:48], axis=0)
            
            left_distance = abs(left_eye_center[0] - face_center_x)
            right_distance = abs(right_eye_center[0] - face_center_x)
            
            # Symmetry score (1.0 = perfect symmetry)
            if max(left_distance, right_distance) > 0:
                symmetry = 1.0 - abs(left_distance - right_distance) / max(left_distance, right_distance)
            else:
                symmetry = 1.0
            
            return float(symmetry)
            
        except Exception as e:
            logger.error(f"Facial symmetry calculation failed: {e}")
            return 0.0
    
    def _extract_image_based_features(self, face_roi: np.ndarray) -> Dict:
        """Extract emotion features directly from image when landmarks unavailable"""
        features = {
            'brightness_variance': 0.0,
            'edge_density': 0.0,
            'texture_contrast': 0.0
        }
        
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # Brightness variance (can indicate facial tension)
            features['brightness_variance'] = float(np.var(gray))
            
            # Edge density (more edges might indicate tension/wrinkles)
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
            # Texture contrast using GLCM approximation
            glcm = self._calculate_simple_glcm(gray)
            features['texture_contrast'] = float(glcm)
            
        except Exception as e:
            logger.error(f"Image-based feature extraction failed: {e}")
        
        return features
    
    def _calculate_simple_glcm(self, gray_image: np.ndarray) -> float:
        """Calculate simplified texture contrast measure"""
        try:
            # Simple texture measure using local standard deviation
            kernel = np.ones((5, 5), np.float32) / 25
            mean_filtered = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            sqr_diff = (gray_image.astype(np.float32) - mean_filtered) ** 2
            texture_measure = np.mean(sqr_diff)
            return texture_measure
        except:
            return 0.0
    
    def classify_emotion(self, features: Dict) -> Dict[str, float]:
        """Classify emotion based on extracted facial features"""
        emotion_scores = {emotion: 0.0 for emotion in self.emotions}
        
        try:
            # Feature thresholds and weights (tuned based on research)
            eye_ratio = features.get('eye_aspect_ratio', 0.0)
            mouth_ratio = features.get('mouth_aspect_ratio', 0.0)
            eyebrow_height = features.get('eyebrow_height', 0.0)
            mouth_curvature = features.get('mouth_curvature', 0.0)
            eye_openness = features.get('eye_openness', 0.0)
            
            # Happy detection
            if mouth_curvature > 2.0:  # Positive mouth curvature (smile)
                emotion_scores['Happy'] += 0.6
                if eye_ratio > 0.2:  # Eyes somewhat closed (smiling eyes)
                    emotion_scores['Happy'] += 0.2
                if mouth_ratio < 0.3:  # Mouth not too open
                    emotion_scores['Happy'] += 0.2
            
            # Sad detection
            if mouth_curvature < -1.5:  # Negative mouth curvature (frown)
                emotion_scores['Sad'] += 0.5
                if eyebrow_height < -2.0:  # Lowered eyebrows
                    emotion_scores['Sad'] += 0.3
                if eye_ratio < 0.18:  # Slightly closed eyes
                    emotion_scores['Sad'] += 0.2
            
            # Surprise detection
            if eyebrow_height > 3.0:  # Raised eyebrows
                emotion_scores['Surprise'] += 0.4
                if eye_ratio > 0.25:  # Wide open eyes
                    emotion_scores['Surprise'] += 0.3
                if mouth_ratio > 0.4:  # Open mouth
                    emotion_scores['Surprise'] += 0.3
            
            # Angry detection
            if eyebrow_height < -3.0:  # Heavily lowered/furrowed brows
                emotion_scores['Angry'] += 0.4
                if mouth_curvature < -1.0:  # Slight frown
                    emotion_scores['Angry'] += 0.3
                if eye_ratio < 0.2:  # Narrowed eyes
                    emotion_scores['Angry'] += 0.3
            
            # Fear detection
            if eyebrow_height > 1.5 and eye_ratio > 0.23:  # Raised brows + wide eyes
                emotion_scores['Fear'] += 0.4
                if mouth_ratio > 0.3:  # Slightly open mouth
                    emotion_scores['Fear'] += 0.3
                if features.get('facial_symmetry', 1.0) < 0.9:  # Slight asymmetry
                    emotion_scores['Fear'] += 0.3
            
            # Disgust detection
            if mouth_curvature < -0.5 and mouth_ratio < 0.25:  # Slight frown, closed mouth
                emotion_scores['Disgust'] += 0.4
                if eyebrow_height < -1.0:  # Lowered brows
                    emotion_scores['Disgust'] += 0.3
                if eye_ratio < 0.19:  # Slightly narrowed eyes
                    emotion_scores['Disgust'] += 0.3
            
            # Neutral detection (baseline when other emotions are low)
            total_other_emotions = sum(score for emotion, score in emotion_scores.items() if emotion != 'Neutral')
            if total_other_emotions < 0.3:
                emotion_scores['Neutral'] = 0.8 - total_other_emotions
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {emotion: score/total_score for emotion, score in emotion_scores.items()}
            else:
                emotion_scores['Neutral'] = 1.0
                
        except Exception as e:
            logger.error(f"Emotion classification failed: {e}")
            emotion_scores = {'Neutral': 1.0}
            for emotion in self.emotions[:-1]:  # All except Neutral
                emotion_scores[emotion] = 0.0
        
        return emotion_scores
    
    def detect_emotion(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Main emotion detection function"""
        try:
            if face_roi is None or face_roi.size == 0:
                return {
                    'primary_emotion': 'Unknown',
                    'confidence': 0.0,
                    'emotion_scores': {emotion: 0.0 for emotion in self.emotions},
                    'features': {}
                }
            
            # Extract facial features
            features = self.extract_facial_features(face_roi, landmarks)
            
            # Classify emotion
            emotion_scores = self.classify_emotion(features)
            
            # Get primary emotion and confidence
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            primary_emotion_name = primary_emotion[0]
            confidence = primary_emotion[1]
            
            # Apply temporal smoothing
            self.emotion_history.append(primary_emotion_name)
            
            # Get most frequent recent emotion for stability
            if len(self.emotion_history) >= 3:
                from collections import Counter
                emotion_counts = Counter(list(self.emotion_history)[-5:])  # Last 5 emotions
                stable_emotion = emotion_counts.most_common(1)[0][0]
                stable_confidence = emotion_counts.most_common(1)[0][1] / len(list(self.emotion_history)[-5:])
                
                # Use stable emotion if confidence is reasonable
                if stable_confidence >= 0.4:
                    primary_emotion_name = stable_emotion
                    confidence = min(confidence + 0.2, 1.0)  # Boost confidence for stable detection
            
            # Additional emotion analysis
            emotional_intensity = self._calculate_emotional_intensity(features)
            micro_expressions = self._detect_micro_expressions(features)
            
            result = {
                'primary_emotion': primary_emotion_name,
                'confidence': round(float(confidence), 3),
                'emotion_scores': {emotion: round(score, 3) for emotion, score in emotion_scores.items()},
                'emotional_intensity': round(emotional_intensity, 3),
                'micro_expressions': micro_expressions,
                'features': {key: round(value, 3) for key, value in features.items()},
                'stability': len(self.emotion_history) >= 5
            }
            
            logger.debug(f"Detected emotion: {primary_emotion_name} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return {
                'primary_emotion': 'Error',
                'confidence': 0.0,
                'emotion_scores': {emotion: 0.0 for emotion in self.emotions},
                'features': {}
            }
    
    def _calculate_emotional_intensity(self, features: Dict) -> float:
        """Calculate overall emotional intensity"""
        try:
            intensity_factors = []
            
            # Mouth curvature intensity
            mouth_curvature = abs(features.get('mouth_curvature', 0.0))
            intensity_factors.append(min(mouth_curvature / 5.0, 1.0))
            
            # Eyebrow movement intensity
            eyebrow_height = abs(features.get('eyebrow_height', 0.0))
            intensity_factors.append(min(eyebrow_height / 5.0, 1.0))
            
            # Eye openness deviation from normal
            eye_ratio = features.get('eye_aspect_ratio', 0.2)
            eye_deviation = abs(eye_ratio - 0.2)  # 0.2 is approximately normal
            intensity_factors.append(min(eye_deviation / 0.1, 1.0))
            
            # Mouth openness
            mouth_ratio = features.get('mouth_aspect_ratio', 0.0)
            intensity_factors.append(min(mouth_ratio / 0.6, 1.0))
            
            # Calculate weighted average
            intensity = sum(intensity_factors) / len(intensity_factors) if intensity_factors else 0.0
            return float(intensity)
            
        except Exception as e:
            logger.error(f"Emotional intensity calculation failed: {e}")
            return 0.0
    
    def _detect_micro_expressions(self, features: Dict) -> List[str]:
        """Detect micro-expressions and subtle facial cues"""
        micro_expressions = []
        
        try:
            # Subtle smile (slight mouth curvature)
            if 0.5 < features.get('mouth_curvature', 0.0) < 2.0:
                micro_expressions.append('subtle_smile')
            
            # Slight frown
            if -1.5 < features.get('mouth_curvature', 0.0) < -0.5:
                micro_expressions.append('slight_frown')
            
            # Raised eyebrows (interest/questioning)
            if 1.0 < features.get('eyebrow_height', 0.0) < 3.0:
                micro_expressions.append('raised_eyebrows')
            
            # Narrowed eyes (concentration/suspicion)
            if features.get('eye_aspect_ratio', 0.2) < 0.18:
                micro_expressions.append('narrowed_eyes')
            
            # Wide eyes (alertness/surprise)
            if features.get('eye_aspect_ratio', 0.2) > 0.25:
                micro_expressions.append('wide_eyes')
            
            # Tight lips (tension)
            if features.get('mouth_aspect_ratio', 0.0) < 0.15:
                micro_expressions.append('tight_lips')
                
        except Exception as e:
            logger.error(f"Micro-expression detection failed: {e}")
        
        return micro_expressions
    
    def get_emotion_trends(self) -> Dict:
        """Get emotion trends from recent history"""
        try:
            if len(self.emotion_history) < 3:
                return {'trend': 'insufficient_data', 'dominant_emotions': []}
            
            from collections import Counter
            emotion_counts = Counter(self.emotion_history)
            
            # Get most common emotions
            dominant_emotions = emotion_counts.most_common(3)
            
            # Analyze trend
            recent_emotions = list(self.emotion_history)[-5:]
            if len(set(recent_emotions)) == 1:
                trend = 'stable'
            elif len(set(recent_emotions)) >= 4:
                trend = 'variable'
            else:
                trend = 'changing'
            
            return {
                'trend': trend,
                'dominant_emotions': [{'emotion': emotion, 'frequency': count} 
                                    for emotion, count in dominant_emotions],
                'recent_stability': len(set(recent_emotions)),
                'total_samples': len(self.emotion_history)
            }
            
        except Exception as e:
            logger.error(f"Emotion trend analysis failed: {e}")
            return {'trend': 'error', 'dominant_emotions': []}
    
    def reset_history(self):
        """Reset emotion detection history"""
        self.emotion_history.clear()
        logger.info("Emotion detector history reset")
    
    def calibrate_for_individual(self, baseline_features: Dict):
        """Calibrate detector for individual facial characteristics (future enhancement)"""
        # This could be implemented to improve accuracy by learning individual baselines
        logger.info("Individual calibration called (not yet implemented)")
        pass