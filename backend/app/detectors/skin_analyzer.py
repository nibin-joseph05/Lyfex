import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class SkinAnalyzer:
    """
    Comprehensive skin health analysis from facial images
    Analyzes skin color, texture, hydration, and potential health indicators
    """
    
    def __init__(self):
        self.skin_color_ranges = {
            'pale': [(0, 0, 180), (25, 80, 255)],  # HSV ranges
            'light': [(0, 20, 180), (30, 100, 255)],
            'medium': [(5, 50, 120), (25, 150, 220)],
            'olive': [(20, 40, 100), (40, 120, 200)],
            'dark': [(10, 30, 60), (30, 100, 180)]
        }
        
        self.health_indicators = {
            'redness_threshold': 0.3,
            'yellowness_threshold': 0.4,
            'brightness_min': 100,
            'brightness_max': 200
        }
        
        logger.info("Skin Analyzer initialized")
    
    def analyze_skin_health(self, face_roi: np.ndarray) -> Dict:
        """
        Comprehensive skin health analysis
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Dictionary containing skin analysis results
        """
        try:
            if face_roi is None or face_roi.size == 0:
                return self._get_error_response("Invalid face region")
            
            # Preprocess image for skin analysis
            processed_face = self._preprocess_for_skin_analysis(face_roi)
            
            # Extract skin regions (avoid eyes, mouth, etc.)
            skin_mask = self._create_skin_mask(processed_face)
            skin_pixels = processed_face[skin_mask > 0]
            
            if len(skin_pixels) == 0:
                return self._get_error_response("No skin pixels detected")
            
            # Analyze color characteristics
            color_analysis = self._analyze_skin_color(skin_pixels)
            
            # Assess skin texture and quality
            texture_analysis = self._analyze_skin_texture(processed_face, skin_mask)
            
            # Estimate hydration levels
            hydration_analysis = self._estimate_hydration(skin_pixels)
            
            # Detect potential health indicators
            health_indicators = self._detect_health_indicators(skin_pixels, color_analysis)
            
            # Calculate overall skin health score
            overall_score = self._calculate_skin_health_score(
                color_analysis, texture_analysis, hydration_analysis, health_indicators
            )
            
            return {
                'overall_health': self._interpret_health_score(overall_score),
                'health_score': round(overall_score, 3),
                'color_analysis': color_analysis,
                'texture_analysis': texture_analysis,
                'hydration_estimate': hydration_analysis,
                'health_indicators': health_indicators,
                'recommendations': self._generate_skin_recommendations(
                    overall_score, color_analysis, hydration_analysis, health_indicators
                )
            }
            
        except Exception as e:
            logger.error(f"Skin analysis failed: {e}")
            return self._get_error_response(f"Analysis failed: {str(e)}")
    
    def _preprocess_for_skin_analysis(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face image for skin analysis"""
        try:
            # Convert to RGB if needed
            if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
                # Assume BGR input, convert to RGB
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_roi
            
            # Apply slight Gaussian blur to reduce noise
            face_smooth = cv2.GaussianBlur(face_rgb, (3, 3), 0)
            
            # Enhance contrast slightly
            face_enhanced = cv2.convertScaleAbs(face_smooth, alpha=1.1, beta=5)
            
            return face_enhanced
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return face_roi
    
    def _create_skin_mask(self, face_rgb: np.ndarray) -> np.ndarray:
        """Create mask to isolate skin regions"""
        try:
            # Convert to different color spaces for skin detection
            hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
            ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
            
            # HSV skin color range
            lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
            mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # YCrCb skin color range
            lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
            upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Combine masks
            skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            return skin_mask
            
        except Exception as e:
            logger.error(f"Skin mask creation error: {e}")
            # Return full mask if skin detection fails
            return np.ones((face_rgb.shape[0], face_rgb.shape[1]), dtype=np.uint8) * 255
    
    def _analyze_skin_color(self, skin_pixels: np.ndarray) -> Dict:
        """Analyze skin color characteristics"""
        try:
            # Calculate average color values
            avg_color = np.mean(skin_pixels, axis=0)
            
            # Convert to different color spaces for analysis
            avg_color_bgr = avg_color[::-1] if len(avg_color) == 3 else avg_color
            avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Classify skin tone
            skin_tone = self._classify_skin_tone(avg_color_hsv)
            
            # Analyze color distribution
            color_std = np.std(skin_pixels, axis=0)
            color_uniformity = 1.0 - (np.mean(color_std) / 255.0)  # Higher = more uniform
            
            # Calculate color temperature (warmth/coolness)
            red_component = avg_color[0] if len(avg_color) > 0 else 0
            blue_component = avg_color[2] if len(avg_color) > 2 else 0
            color_temperature = "warm" if red_component > blue_component else "cool"
            
            return {
                'skin_tone': skin_tone,
                'average_color_rgb': [int(c) for c in avg_color],
                'average_color_hsv': [int(c) for c in avg_color_hsv],
                'color_uniformity': round(color_uniformity, 3),
                'color_temperature': color_temperature,
                'red_component': int(red_component),
                'blue_component': int(blue_component),
                'color_variance': round(np.mean(color_std), 3)
            }
            
        except Exception as e:
            logger.error(f"Color analysis error: {e}")
            return {
                'skin_tone': 'unknown',
                'average_color_rgb': [0, 0, 0],
                'color_uniformity': 0.0,
                'color_temperature': 'unknown'
            }
    
    def _classify_skin_tone(self, hsv_color: np.ndarray) -> str:
        """Classify skin tone based on HSV values"""
        try:
            h, s, v = hsv_color
            
            # Simple skin tone classification based on value (brightness) and saturation
            if v > 200:
                return "Very Light"
            elif v > 160:
                return "Light"
            elif v > 120:
                return "Medium"
            elif v > 80:
                return "Dark"
            else:
                return "Very Dark"
                
        except Exception as e:
            logger.error(f"Skin tone classification error: {e}")
            return "Unknown"
    
    def _analyze_skin_texture(self, face_rgb: np.ndarray, skin_mask: np.ndarray) -> Dict:
        """Analyze skin texture and quality"""
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
            
            # Apply mask to focus on skin regions
            skin_gray = cv2.bitwise_and(gray, gray, mask=skin_mask)
            
            # Calculate texture metrics
            
            # 1. Smoothness (inverse of standard deviation)
            smoothness = 255 - np.std(skin_gray[skin_mask > 0])
            smoothness_score = smoothness / 255.0
            
            # 2. Calculate local binary patterns (simplified)
            edges = cv2.Canny(skin_gray, 50, 150)
            edge_density = np.sum(edges[skin_mask > 0]) / np.sum(skin_mask > 0) if np.sum(skin_mask > 0) > 0 else 0
            texture_score = 1.0 - (edge_density / 255.0)
            
            # 3. Calculate contrast (local variance)
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(skin_gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((skin_gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            avg_contrast = np.mean(local_variance[skin_mask > 0]) if np.sum(skin_mask > 0) > 0 else 0
            contrast_normalized = min(avg_contrast / 1000.0, 1.0)  # Normalize
            
            # Overall texture quality
            texture_quality = (smoothness_score * 0.4 + texture_score * 0.4 + (1 - contrast_normalized) * 0.2)
            
            return {
                'texture_quality': self._interpret_texture_quality(texture_quality),
                'texture_score': round(texture_quality, 3),
                'smoothness': round(smoothness_score, 3),
                'edge_density': round(edge_density, 3),
                'contrast': round(contrast_normalized, 3),
                'overall_assessment': 'good' if texture_quality > 0.7 else 'moderate' if texture_quality > 0.5 else 'concern'
            }
            
        except Exception as e:
            logger.error(f"Texture analysis error: {e}")
            return {
                'texture_quality': 'unknown',
                'texture_score': 0.0,
                'overall_assessment': 'error'
            }
    
    def _estimate_hydration(self, skin_pixels: np.ndarray) -> Dict:
        """Estimate skin hydration levels"""
        try:
            # Convert to HSV for better hydration assessment
            avg_color_bgr = np.mean(skin_pixels, axis=0)[::-1]
            hsv_color = cv2.cvtColor(np.uint8([[avg_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            
            h, s, v = hsv_color
            
            # Hydration indicators
            # Well-hydrated skin typically has:
            # - Higher saturation (more vibrant)
            # - Balanced brightness
            # - Even color distribution
            
            saturation_score = s / 255.0
            brightness_score = 1.0 - abs(v - 140) / 140.0  # Optimal around 140
            brightness_score = max(0, brightness_score)
            
            # Calculate color variance (well-hydrated skin has less variance)
            color_variance = np.var(skin_pixels, axis=0)
            variance_score = 1.0 - (np.mean(color_variance) / 255.0)
            
            # Combined hydration score
            hydration_score = (saturation_score * 0.4 + brightness_score * 0.3 + variance_score * 0.3)
            
            hydration_level = self._interpret_hydration_level(hydration_score)
            
            return {
                'hydration_level': hydration_level,
                'hydration_score': round(hydration_score, 3),
                'saturation_component': round(saturation_score, 3),
                'brightness_component': round(brightness_score, 3),
                'variance_component': round(variance_score, 3),
                'assessment': 'well_hydrated' if hydration_score > 0.7 else 'moderately_hydrated' if hydration_score > 0.5 else 'dehydrated'
            }
            
        except Exception as e:
            logger.error(f"Hydration analysis error: {e}")
            return {
                'hydration_level': 'unknown',
                'hydration_score': 0.0,
                'assessment': 'error'
            }
    
    def _detect_health_indicators(self, skin_pixels: np.ndarray, color_analysis: Dict) -> Dict:
        """Detect potential health indicators from skin color and appearance"""
        try:
            indicators = {}
            warnings = []
            
            # Analyze color components
            avg_color = np.mean(skin_pixels, axis=0)
            red_component = avg_color[0] if len(avg_color) > 0 else 0
            green_component = avg_color[1] if len(avg_color) > 1 else 0
            blue_component = avg_color[2] if len(avg_color) > 2 else 0
            
            # Redness analysis (potential inflammation, rosacea, etc.)
            red_dominance = red_component / (green_component + blue_component + 1)
            if red_dominance > self.health_indicators['redness_threshold']:
                indicators['excessive_redness'] = True
                warnings.append("Elevated redness detected - may indicate inflammation or circulation issues")
            else:
                indicators['excessive_redness'] = False
            
            # Yellowness analysis (potential liver issues, jaundice)
            yellow_ratio = (red_component + green_component) / (blue_component + 1)
            if yellow_ratio > self.health_indicators['yellowness_threshold']:
                indicators['yellowish_tint'] = True
                warnings.append("Yellowish tint detected - monitor for potential health concerns")
            else:
                indicators['yellowish_tint'] = False
            
            # Paleness analysis
            avg_brightness = np.mean(avg_color)
            if avg_brightness < self.health_indicators['brightness_min']:
                indicators['unusual_paleness'] = True
                warnings.append("Unusual paleness detected - may indicate circulation or nutritional concerns")
            elif avg_brightness > self.health_indicators['brightness_max']:
                indicators['excessive_brightness'] = True
                warnings.append("Very bright skin tone - ensure proper sun protection")
            else:
                indicators['normal_brightness'] = True
            
            # Color uniformity (potential pigmentation issues)
            color_uniformity = color_analysis.get('color_uniformity', 0)
            if color_uniformity < 0.6:
                indicators['uneven_pigmentation'] = True
                warnings.append("Uneven pigmentation detected - consider skincare routine evaluation")
            else:
                indicators['even_pigmentation'] = True
            
            # Overall health assessment based on indicators
            concern_count = sum([
                indicators.get('excessive_redness', False),
                indicators.get('yellowish_tint', False),
                indicators.get('unusual_paleness', False),
                indicators.get('uneven_pigmentation', False)
            ])
            
            if concern_count == 0:
                health_status = "healthy"
            elif concern_count == 1:
                health_status = "minor_concerns"
            elif concern_count == 2:
                health_status = "moderate_concerns"
            else:
                health_status = "multiple_concerns"
            
            return {
                'status': health_status,
                'indicators': indicators,
                'warnings': warnings,
                'concern_count': concern_count,
                'red_dominance': round(red_dominance, 3),
                'yellow_ratio': round(yellow_ratio, 3),
                'brightness_level': round(avg_brightness, 3)
            }
            
        except Exception as e:
            logger.error(f"Health indicators detection error: {e}")
            return {
                'status': 'error',
                'indicators': {},
                'warnings': ['Health analysis failed'],
                'concern_count': 0
            }
    
    def _calculate_skin_health_score(self, color_analysis: Dict, texture_analysis: Dict, 
                                   hydration_analysis: Dict, health_indicators: Dict) -> float:
        """Calculate overall skin health score"""
        try:
            # Weight factors for different components
            color_weight = 0.25
            texture_weight = 0.30
            hydration_weight = 0.25
            health_weight = 0.20
            
            # Color score
            color_score = color_analysis.get('color_uniformity', 0) * color_weight
            
            # Texture score
            texture_score = texture_analysis.get('texture_score', 0) * texture_weight
            
            # Hydration score
            hydration_score = hydration_analysis.get('hydration_score', 0) * hydration_weight
            
            # Health indicators score (inverse of concerns)
            concern_count = health_indicators.get('concern_count', 0)
            health_score = max(0, (1 - concern_count / 4)) * health_weight
            
            # Total score
            total_score = color_score + texture_score + hydration_score + health_score
            
            return min(total_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Skin health score calculation error: {e}")
            return 0.0
    
    def _interpret_health_score(self, score: float) -> str:
        """Interpret overall skin health score"""
        if score > 0.8:
            return "Excellent"
        elif score > 0.7:
            return "Very Good"
        elif score > 0.6:
            return "Good"
        elif score > 0.5:
            return "Fair"
        elif score > 0.3:
            return "Needs Attention"
        else:
            return "Poor"
    
    def _interpret_texture_quality(self, score: float) -> str:
        """Interpret texture quality score"""
        if score > 0.8:
            return "Very Smooth"
        elif score > 0.6:
            return "Smooth"
        elif score > 0.4:
            return "Moderate"
        elif score > 0.2:
            return "Rough"
        else:
            return "Very Rough"
    
    def _interpret_hydration_level(self, score: float) -> str:
        """Interpret hydration level score"""
        if score > 0.8:
            return "Well Hydrated"
        elif score > 0.6:
            return "Adequately Hydrated"
        elif score > 0.4:
            return "Moderately Hydrated"
        elif score > 0.2:
            return "Mildly Dehydrated"
        else:
            return "Dehydrated"
    
    def _generate_skin_recommendations(self, health_score: float, color_analysis: Dict, 
                                     hydration_analysis: Dict, health_indicators: Dict) -> List[str]:
        """Generate personalized skin care recommendations"""
        recommendations = []
        
        # Base recommendations
        if health_score > 0.7:
            recommendations.append("Your skin appears healthy - maintain current skincare routine")
        
        # Hydration recommendations
        hydration_score = hydration_analysis.get('hydration_score', 0)
        if hydration_score < 0.5:
            recommendations.extend([
                "Increase water intake to improve skin hydration",
                "Consider using a moisturizer with hyaluronic acid",
                "Use a humidifier in dry environments"
            ])
        elif hydration_score < 0.7:
            recommendations.append("Maintain good hydration habits")
        
        # Color and health indicator recommendations
        indicators = health_indicators.get('indicators', {})
        
        if indicators.get('excessive_redness'):
            recommendations.extend([
                "Consider anti-inflammatory skincare products",
                "Avoid harsh scrubs or irritating ingredients",
                "Protect skin from wind and extreme temperatures"
            ])
        
        if indicators.get('yellowish_tint'):
            recommendations.extend([
                "Monitor for persistent yellowing - consult healthcare provider if concerned",
                "Ensure adequate sleep and stress management"
            ])
        
        if indicators.get('unusual_paleness'):
            recommendations.extend([
                "Ensure adequate iron and vitamin intake",
                "Consider light exercise to improve circulation"
            ])
        
        if indicators.get('uneven_pigmentation'):
            recommendations.extend([
                "Use broad-spectrum sunscreen daily",
                "Consider vitamin C serum for pigmentation support",
                "Avoid excessive sun exposure"
            ])
        
        # General recommendations
        recommendations.extend([
            "Maintain a balanced diet rich in antioxidants",
            "Get adequate sleep for skin repair",
            "Use gentle, fragrance-free skincare products"
        ])
        
        # Skin tone specific recommendations
        skin_tone = color_analysis.get('skin_tone', '')
        if 'Light' in skin_tone:
            recommendations.append("Use high SPF sunscreen - lighter skin is more susceptible to UV damage")
        elif 'Dark' in skin_tone:
            recommendations.append("Monitor for ashy appearance - may indicate dehydration")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _get_error_response(self, error_message: str) -> Dict:
        """Return error response format"""
        return {
            'overall_health': 'Analysis Error',
            'health_score': 0.0,
            'color_analysis': {
                'skin_tone': 'unknown',
                'color_uniformity': 0.0,
                'color_temperature': 'unknown'
            },
            'texture_analysis': {
                'texture_quality': 'unknown',
                'texture_score': 0.0,
                'overall_assessment': 'error'
            },
            'hydration_estimate': {
                'hydration_level': 'unknown',
                'hydration_score': 0.0,
                'assessment': 'error'
            },
            'health_indicators': {
                'status': 'error',
                'indicators': {},
                'warnings': [f"Analysis failed: {error_message}"]
            },
            'recommendations': [f"Skin analysis failed: {error_message}"]
        }
    
    def quick_skin_assessment(self, face_roi: np.ndarray) -> Dict:
        """
        Quick skin health screening
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Basic skin health indicators
        """
        try:
            if face_roi is None or face_roi.size == 0:
                return {'status': 'no_face', 'health': 'unknown'}
            
            # Basic color analysis
            processed_face = self._preprocess_for_skin_analysis(face_roi)
            skin_mask = self._create_skin_mask(processed_face)
            skin_pixels = processed_face[skin_mask > 0]
            
            if len(skin_pixels) == 0:
                return {'status': 'no_skin', 'health': 'unknown'}
            
            # Quick assessments
            avg_color = np.mean(skin_pixels, axis=0)
            brightness = np.mean(avg_color)
            color_variance = np.var(skin_pixels, axis=0).mean()
            
            # Simple health assessment
            if brightness < 80:
                quick_assessment = 'pale'
            elif brightness > 180:
                quick_assessment = 'bright'
            elif color_variance > 800:
                quick_assessment = 'uneven'
            else:
                quick_assessment = 'normal'
            
            return {
                'status': 'analyzed',
                'health': quick_assessment,
                'brightness': round(brightness, 2),
                'color_variance': round(color_variance, 2),
                'quick_recommendation': self._get_quick_recommendation(quick_assessment)
            }
            
        except Exception as e:
            logger.error(f"Quick skin assessment error: {e}")
            return {'status': 'error', 'health': 'unknown'}
    
    def _get_quick_recommendation(self, assessment: str) -> str:
        """Get quick recommendation based on assessment"""
        recommendations = {
            'pale': 'Consider improving circulation through exercise and proper nutrition',
            'bright': 'Ensure adequate sun protection',
            'uneven': 'Focus on consistent skincare routine and sun protection',
            'normal': 'Maintain current healthy habits'
        }
        return recommendations.get(assessment, 'Continue monitoring skin health')