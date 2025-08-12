import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthAssessor:
    """
    Comprehensive health assessment system that analyzes multiple health metrics
    and provides overall health scoring, risk assessment, and recommendations
    """
    
    def __init__(self):
        # Health metric weights for overall score calculation
        self.metric_weights = {
            'cardiovascular': 0.25,    # Heart rate, HRV
            'respiratory': 0.20,       # Breathing rate, pattern
            'neurological': 0.15,      # Fatigue, cognitive function
            'psychological': 0.15,     # Stress, emotion
            'physical': 0.15,          # Skin health, appearance
            'behavioral': 0.10         # Alertness, posture
        }
        
        # Normal ranges for various health metrics
        self.normal_ranges = {
            'heart_rate': {'min': 60, 'max': 100, 'optimal': (70, 85)},
            'heart_rate_variability': {'min': 20, 'max': 100, 'optimal': (30, 60)},
            'respiratory_rate': {'min': 12, 'max': 20, 'optimal': (14, 18)},
            'stress_level': {'low': 0.3, 'moderate': 0.6, 'high': 0.8},
            'fatigue_score': {'alert': 0.8, 'moderate': 0.6, 'tired': 0.4},
            'emotion_stability': {'stable': 0.7, 'moderate': 0.5, 'unstable': 0.3}
        }
        
        # Risk factor thresholds
        self.risk_thresholds = {
            'high_risk_heart_rate': {'min': 45, 'max': 120},
            'high_risk_stress': 0.8,
            'high_risk_fatigue': 0.3,
            'multiple_risk_factors': 3
        }
        
        # Health status categories
        self.health_categories = {
            'excellent': (0.90, 1.00),
            'very_good': (0.80, 0.90),
            'good': (0.70, 0.80),
            'fair': (0.60, 0.70),
            'poor': (0.40, 0.60),
            'concerning': (0.00, 0.40)
        }
        
        logger.info("Health Assessor initialized with comprehensive evaluation metrics")
    
    def calculate_health_score(self, health_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall health score based on multiple health metrics
        
        Args:
            health_metrics: Dictionary containing all health measurement results
            
        Returns:
            Dictionary with overall health assessment
        """
        try:
            # Initialize component scores
            component_scores = {
                'cardiovascular': 0.0,
                'respiratory': 0.0,
                'neurological': 0.0,
                'psychological': 0.0,
                'physical': 0.0,
                'behavioral': 0.0
            }
            
            # Calculate cardiovascular score
            component_scores['cardiovascular'] = self._assess_cardiovascular_health(health_metrics)
            
            # Calculate respiratory score
            component_scores['respiratory'] = self._assess_respiratory_health(health_metrics)
            
            # Calculate neurological score
            component_scores['neurological'] = self._assess_neurological_health(health_metrics)
            
            # Calculate psychological score
            component_scores['psychological'] = self._assess_psychological_health(health_metrics)
            
            # Calculate physical score
            component_scores['physical'] = self._assess_physical_health(health_metrics)
            
            # Calculate behavioral score
            component_scores['behavioral'] = self._assess_behavioral_health(health_metrics)
            
            # Calculate weighted overall score
            overall_score = sum(
                score * self.metric_weights[component] 
                for component, score in component_scores.items()
            )
            
            # Determine health status
            health_status = self._categorize_health_status(overall_score)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(health_metrics)
            
            # Generate alerts
            alerts = self._generate_health_alerts(health_metrics, risk_factors, overall_score)
            
            # Calculate trend if historical data available
            health_trend = self._calculate_health_trend(health_metrics)
            
            return {
                'score': round(overall_score, 3),
                'status': health_status,
                'component_scores': {k: round(v, 3) for k, v in component_scores.items()},
                'risk_factors': risk_factors,
                'alerts': alerts,
                'trend': health_trend,
                'assessment_timestamp': datetime.now().isoformat(),
                'confidence': self._calculate_assessment_confidence(health_metrics)
            }
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return self._get_error_assessment()
    
    def _assess_cardiovascular_health(self, metrics: Dict[str, Any]) -> float:
        """Assess cardiovascular health component"""
        try:
            cardiovascular_score = 0.0
            factors_assessed = 0
            
            # Heart rate assessment
            heart_rate = self._extract_numeric_value(metrics.get('heartRate'))
            if heart_rate is not None:
                hr_score = self._score_heart_rate(heart_rate)
                cardiovascular_score += hr_score * 0.6
                factors_assessed += 1
            
            # Heart rate variability assessment
            hrv = self._extract_numeric_value(metrics.get('heartRateVariability'))
            if hrv is not None:
                hrv_score = self._score_hrv(hrv)
                cardiovascular_score += hrv_score * 0.4
                factors_assessed += 1
            
            # Return average if any factors assessed
            return cardiovascular_score / factors_assessed if factors_assessed > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Cardiovascular assessment error: {e}")
            return 0.5
    
    def _assess_respiratory_health(self, metrics: Dict[str, Any]) -> float:
        """Assess respiratory health component"""
        try:
            respiratory_score = 0.0
            factors_assessed = 0
            
            # Respiratory rate assessment
            resp_rate = self._extract_numeric_value(metrics.get('respiratoryRate'))
            if resp_rate is not None:
                resp_score = self._score_respiratory_rate(resp_rate)
                respiratory_score += resp_score * 0.7
                factors_assessed += 1
            
            # Breathing pattern assessment
            breathing_pattern = metrics.get('breathingPattern', '')
            if breathing_pattern:
                pattern_score = self._score_breathing_pattern(breathing_pattern)
                respiratory_score += pattern_score * 0.3
                factors_assessed += 1
            
            return respiratory_score / factors_assessed if factors_assessed > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Respiratory assessment error: {e}")
            return 0.5
    
    def _assess_neurological_health(self, metrics: Dict[str, Any]) -> float:
        """Assess neurological health component"""
        try:
            neurological_score = 0.0
            factors_assessed = 0
            
            # Fatigue assessment
            alertness_score = self._extract_numeric_value(metrics.get('alertnessScore'))
            if alertness_score is not None:
                neurological_score += alertness_score * 0.4
                factors_assessed += 1
            
            # Facial asymmetry assessment
            facial_asymmetry = metrics.get('facialAsymmetry')
            if facial_asymmetry and isinstance(facial_asymmetry, (int, float)):
                asymmetry_score = max(0, 1.0 - facial_asymmetry)  # Lower asymmetry = better score
                neurological_score += asymmetry_score * 0.3
                factors_assessed += 1
            
            # Eye movement assessment
            eye_movement = metrics.get('eyeMovement')
            if eye_movement and eye_movement != 'N/A':
                movement_score = 0.8 if 'normal' in str(eye_movement).lower() else 0.4
                neurological_score += movement_score * 0.3
                factors_assessed += 1
            
            return neurological_score / factors_assessed if factors_assessed > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Neurological assessment error: {e}")
            return 0.5
    
    def _assess_psychological_health(self, metrics: Dict[str, Any]) -> float:
        """Assess psychological health component"""
        try:
            psychological_score = 0.0
            factors_assessed = 0
            
            # Stress level assessment
            stress_level = metrics.get('stressLevel', '')
            if stress_level:
                stress_score = self._score_stress_level(stress_level)
                psychological_score += stress_score * 0.6
                factors_assessed += 1
            
            # Emotion assessment
            emotion = metrics.get('emotion', '')
            emotion_confidence = self._extract_numeric_value(metrics.get('emotionConfidence', 0))
            if emotion and emotion_confidence:
                emotion_score = self._score_emotion(emotion, emotion_confidence)
                psychological_score += emotion_score * 0.4
                factors_assessed += 1
            
            return psychological_score / factors_assessed if factors_assessed > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Psychological assessment error: {e}")
            return 0.5
    
    def _assess_physical_health(self, metrics: Dict[str, Any]) -> float:
        """Assess physical health component"""
        try:
            physical_score = 0.0
            factors_assessed = 0
            
            # Skin analysis assessment
            skin_analysis = metrics.get('skinAnalysis')
            if skin_analysis and skin_analysis != 'N/A':
                skin_score = self._score_skin_health(skin_analysis)
                physical_score += skin_score * 0.5
                factors_assessed += 1
            
            # Hydration status assessment
            hydration_status = metrics.get('hydrationStatus')
            if hydration_status and hydration_status != 'N/A':
                hydration_score = self._score_hydration(hydration_status)
                physical_score += hydration_score * 0.3
                factors_assessed += 1
            
            # Skin color assessment
            skin_color = metrics.get('skinColor')
            if skin_color and skin_color != 'N/A':
                color_score = 0.7  # Default good score if skin color is analyzed
                physical_score += color_score * 0.2
                factors_assessed += 1
            
            return physical_score / factors_assessed if factors_assessed > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Physical assessment error: {e}")
            return 0.5
    
    def _assess_behavioral_health(self, metrics: Dict[str, Any]) -> float:
        """Assess behavioral health component"""
        try:
            behavioral_score = 0.0
            factors_assessed = 0
            
            # Fatigue level assessment
            fatigue = metrics.get('fatigue', '')
            if fatigue:
                fatigue_score = self._score_fatigue_level(fatigue)
                behavioral_score += fatigue_score * 0.7
                factors_assessed += 1
            
            # Alertness score
            alertness = self._extract_numeric_value(metrics.get('alertnessScore'))
            if alertness is not None:
                behavioral_score += alertness * 0.3
                factors_assessed += 1
            
            return behavioral_score / factors_assessed if factors_assessed > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Behavioral assessment error: {e}")
            return 0.5
    
    def _score_heart_rate(self, heart_rate: float) -> float:
        """Score heart rate based on normal ranges"""
        ranges = self.normal_ranges['heart_rate']
        
        if ranges['optimal'][0] <= heart_rate <= ranges['optimal'][1]:
            return 1.0
        elif ranges['min'] <= heart_rate <= ranges['max']:
            # Linear scoring within normal range
            if heart_rate < ranges['optimal'][0]:
                return 0.7 + 0.3 * (heart_rate - ranges['min']) / (ranges['optimal'][0] - ranges['min'])
            else:
                return 0.7 + 0.3 * (ranges['max'] - heart_rate) / (ranges['max'] - ranges['optimal'][1])
        else:
            # Outside normal range
            return 0.3 if ranges['min'] * 0.8 <= heart_rate <= ranges['max'] * 1.2 else 0.1
    
    def _score_hrv(self, hrv: float) -> float:
        """Score heart rate variability"""
        ranges = self.normal_ranges['heart_rate_variability']
        
        if ranges['optimal'][0] <= hrv <= ranges['optimal'][1]:
            return 1.0
        elif ranges['min'] <= hrv <= ranges['max']:
            return 0.7
        else:
            return 0.3 if hrv >= ranges['min'] * 0.5 else 0.1
    
    def _score_respiratory_rate(self, resp_rate: float) -> float:
        """Score respiratory rate"""
        ranges = self.normal_ranges['respiratory_rate']
        
        if ranges['optimal'][0] <= resp_rate <= ranges['optimal'][1]:
            return 1.0
        elif ranges['min'] <= resp_rate <= ranges['max']:
            return 0.8
        else:
            return 0.4 if ranges['min'] * 0.8 <= resp_rate <= ranges['max'] * 1.2 else 0.2
    
    def _score_breathing_pattern(self, pattern: str) -> float:
        """Score breathing pattern"""
        pattern_lower = pattern.lower()
        
        if 'regular' in pattern_lower or 'normal' in pattern_lower:
            return 1.0
        elif 'slightly irregular' in pattern_lower:
            return 0.7
        elif 'irregular' in pattern_lower:
            return 0.4
        else:
            return 0.5  # Default for unknown patterns
    
    def _score_stress_level(self, stress_level: str) -> float:
        """Score stress level"""
        stress_lower = stress_level.lower()
        
        if 'low' in stress_lower or 'relaxed' in stress_lower:
            return 1.0
        elif 'moderate' in stress_lower or 'mild' in stress_lower:
            return 0.7
        elif 'high' in stress_lower or 'elevated' in stress_lower:
            return 0.4
        elif 'severe' in stress_lower or 'extreme' in stress_lower:
            return 0.2
        else:
            return 0.5
    
    def _score_emotion(self, emotion: str, confidence: float) -> float:
        """Score emotional state"""
        emotion_lower = emotion.lower()
        
        # Positive emotions
        if emotion_lower in ['happy', 'content', 'calm', 'peaceful']:
            base_score = 1.0
        elif emotion_lower in ['neutral', 'focused']:
            base_score = 0.8
        elif emotion_lower in ['surprised', 'excited']:
            base_score = 0.7
        elif emotion_lower in ['sad', 'worried', 'concerned']:
            base_score = 0.4
        elif emotion_lower in ['angry', 'fearful', 'anxious']:
            base_score = 0.2
        else:
            base_score = 0.5
        
        # Adjust by confidence
        return base_score * confidence
    
    def _score_skin_health(self, skin_health: str) -> float:
        """Score skin health"""
        if isinstance(skin_health, str):
            health_lower = skin_health.lower()
            
            if 'excellent' in health_lower or 'very good' in health_lower:
                return 1.0
            elif 'good' in health_lower:
                return 0.8
            elif 'fair' in health_lower or 'moderate' in health_lower:
                return 0.6
            elif 'poor' in health_lower or 'needs attention' in health_lower:
                return 0.3
            else:
                return 0.5
        else:
            return 0.5
    
    def _score_hydration(self, hydration_status: str) -> float:
        """Score hydration status"""
        if isinstance(hydration_status, str):
            hydration_lower = hydration_status.lower()
            
            if 'well hydrated' in hydration_lower:
                return 1.0
            elif 'adequately hydrated' in hydration_lower:
                return 0.8
            elif 'moderately hydrated' in hydration_lower:
                return 0.6
            elif 'mildly dehydrated' in hydration_lower:
                return 0.4
            elif 'dehydrated' in hydration_lower:
                return 0.2
            else:
                return 0.5
        else:
            return 0.5
    
    def _score_fatigue_level(self, fatigue: str) -> float:
        """Score fatigue level"""
        fatigue_lower = fatigue.lower()
        
        if 'very alert' in fatigue_lower:
            return 1.0
        elif 'alert' in fatigue_lower:
            return 0.9
        elif 'slightly drowsy' in fatigue_lower:
            return 0.7
        elif 'drowsy' in fatigue_lower:
            return 0.4
        elif 'very drowsy' in fatigue_lower:
            return 0.2
        else:
            return 0.5
    
    def _categorize_health_status(self, score: float) -> str:
        """Categorize overall health status based on score"""
        for status, (min_score, max_score) in self.health_categories.items():
            if min_score <= score < max_score:
                return status.replace('_', ' ').title()
        return 'Unknown'
    
    def _identify_risk_factors(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify health risk factors"""
        risk_factors = []
        
        try:
            # Heart rate risks
            heart_rate = self._extract_numeric_value(metrics.get('heartRate'))
            if heart_rate is not None:
                hr_thresholds = self.risk_thresholds['high_risk_heart_rate']
                if heart_rate < hr_thresholds['min'] or heart_rate > hr_thresholds['max']:
                    risk_factors.append("Abnormal heart rate detected")
            
            # Stress risks
            stress_level = metrics.get('stressLevel', '').lower()
            if 'high' in stress_level or 'severe' in stress_level or 'extreme' in stress_level:
                risk_factors.append("Elevated stress levels")
            
            # Fatigue risks
            alertness_score = self._extract_numeric_value(metrics.get('alertnessScore'))
            if alertness_score is not None and alertness_score < self.risk_thresholds['high_risk_fatigue']:
                risk_factors.append("Significant fatigue detected")
            
            # Multiple risk factor warning
            if len(risk_factors) >= self.risk_thresholds['multiple_risk_factors']:
                risk_factors.append("Multiple health concerns identified")
            
            # Neurological risks
            tremor = metrics.get('tremor')
            if tremor and str(tremor).lower() not in ['false', 'no', 'none', 'n/a']:
                risk_factors.append("Potential motor control issues")
            
            # Respiratory risks
            resp_rate = self._extract_numeric_value(metrics.get('respiratoryRate'))
            if resp_rate is not None:
                if resp_rate < 10 or resp_rate > 25:
                    risk_factors.append("Abnormal breathing rate")
            
        except Exception as e:
            logger.error(f"Risk factor identification error: {e}")
            risk_factors.append("Risk assessment incomplete")
        
        return risk_factors
    
    def _generate_health_alerts(self, metrics: Dict[str, Any], risk_factors: List[str], overall_score: float) -> List[str]:
        """Generate health alerts and warnings"""
        alerts = []
        
        # Overall score alerts
        if overall_score < 0.4:
            alerts.append("⚠️ Multiple health concerns detected - consider medical consultation")
        elif overall_score < 0.6:
            alerts.append("⚠️ Health monitoring recommended - some metrics need attention")
        
        # Specific metric alerts
        heart_rate = self._extract_numeric_value(metrics.get('heartRate'))
        if heart_rate is not None:
            if heart_rate > 120:
                alerts.append("🔴 Elevated heart rate detected")
            elif heart_rate < 50:
                alerts.append("🔵 Low heart rate detected")
        
        # Stress alerts
        stress_level = metrics.get('stressLevel', '').lower()
        if 'high' in stress_level or 'severe' in stress_level:
            alerts.append("😰 High stress levels detected - relaxation recommended")
        
        # Fatigue alerts
        fatigue = metrics.get('fatigue', '').lower()
        if 'very drowsy' in fatigue or 'drowsy' in fatigue:
            alerts.append("😴 Significant fatigue detected - rest recommended")
        
        # Risk factor alerts
        if len(risk_factors) > 0:
            alerts.append(f"⚠️ {len(risk_factors)} risk factor(s) identified")
        
        return alerts
    
    def _calculate_health_trend(self, metrics: Dict[str, Any]) -> str:
        """Calculate health trend (placeholder for historical data analysis)"""
        # This would require historical data storage and comparison
        # For now, return based on current metrics
        overall_indicators = [
            metrics.get('heartRate'),
            metrics.get('stressLevel'),
            metrics.get('fatigue'),
            metrics.get('emotion')
        ]
        
        # Simple trend analysis based on current state
        concerning_indicators = 0
        for indicator in overall_indicators:
            if indicator and isinstance(indicator, str):
                indicator_lower = indicator.lower()
                if any(word in indicator_lower for word in ['high', 'elevated', 'poor', 'severe', 'drowsy']):
                    concerning_indicators += 1
        
        if concerning_indicators >= 3:
            return 'Declining'
        elif concerning_indicators <= 1:
            return 'Stable'
        else:
            return 'Monitoring'
    
    def _calculate_assessment_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence in the health assessment"""
        try:
            available_metrics = 0
            total_metrics = 10  # Expected total metrics
            
            key_metrics = [
                'heartRate', 'respiratoryRate', 'emotion', 'stressLevel', 
                'fatigue', 'skinAnalysis', 'facialAsymmetry', 'tremor',
                'hydrationStatus', 'alertnessScore'
            ]
            
            for metric in key_metrics:
                value = metrics.get(metric)
                if value and str(value).lower() not in ['n/a', 'analysis error', 'detection error', 'unknown']:
                    available_metrics += 1
            
            confidence = available_metrics / total_metrics
            return round(confidence, 3)
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _extract_numeric_value(self, value: Any) -> Optional[float]:
        """Extract numeric value from various input types"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Try to extract number from string
                import re
                numbers = re.findall(r'-?\d+\.?\d*', value)
                if numbers:
                    return float(numbers[0])
            return None
        except:
            return None
    
    def _get_error_assessment(self) -> Dict[str, Any]:
        """Return error assessment when calculation fails"""
        return {
            'score': 0.0,
            'status': 'Assessment Error',
            'component_scores': {k: 0.0 for k in self.metric_weights.keys()},
            'risk_factors': ['Health assessment failed'],
            'alerts': ['Unable to complete health assessment'],
            'trend': 'Unknown',
            'assessment_timestamp': datetime.now().isoformat(),
            'confidence': 0.0
        }
    
    def generate_recommendations(self, health_metrics: Dict[str, Any]) -> List[str]:
        """Generate personalized health recommendations"""
        recommendations = []
        
        try:
            # Cardiovascular recommendations
            heart_rate = self._extract_numeric_value(health_metrics.get('heartRate'))
            if heart_rate is not None:
                if heart_rate > 100:
                    recommendations.extend([
                        "Consider stress reduction techniques",
                        "Monitor caffeine and stimulant intake",
                        "Ensure adequate rest and hydration"
                    ])
                elif heart_rate < 60:
                    recommendations.extend([
                        "Monitor for symptoms of dizziness or fatigue",
                        "Consider consultation if accompanied by other symptoms"
                    ])
            
            # Stress recommendations
            stress_level = health_metrics.get('stressLevel', '').lower()
            if 'high' in stress_level or 'elevated' in stress_level:
                recommendations.extend([
                    "Practice deep breathing exercises",
                    "Consider meditation or mindfulness techniques",
                    "Ensure adequate sleep (7-9 hours)",
                    "Engage in regular physical activity"
                ])
            
            # Fatigue recommendations
            fatigue = health_metrics.get('fatigue', '').lower()
            if 'drowsy' in fatigue or 'tired' in fatigue:
                recommendations.extend([
                    "Take regular breaks during work",
                    "Ensure proper sleep hygiene",
                    "Consider a brief nap if possible (10-20 minutes)",
                    "Stay hydrated throughout the day"
                ])
            
            # Respiratory recommendations
            resp_rate = self._extract_numeric_value(health_metrics.get('respiratoryRate'))
            if resp_rate is not None:
                if resp_rate > 20:
                    recommendations.extend([
                        "Practice controlled breathing exercises",
                        "Reduce anxiety and stress levels"
                    ])
                elif resp_rate < 12:
                    recommendations.append("Monitor breathing patterns and consult if concerning")
            
            # Skin and hydration recommendations
            hydration = health_metrics.get('hydrationStatus', '').lower()
            if 'dehydrated' in hydration or 'dry' in hydration:
                recommendations.extend([
                    "Increase daily water intake",
                    "Use a humidifier in dry environments",
                    "Apply moisturizer regularly"
                ])
            
            # Emotion and psychological recommendations
            emotion = health_metrics.get('emotion', '').lower()
            if emotion in ['sad', 'angry', 'anxious', 'fearful']:
                recommendations.extend([
                    "Consider talking to a friend or counselor",
                    "Practice relaxation techniques",
                    "Engage in enjoyable activities"
                ])
            
            # General health recommendations
            recommendations.extend([
                "Maintain a balanced diet rich in fruits and vegetables",
                "Get regular exercise (at least 30 minutes daily)",
                "Maintain consistent sleep schedule",
                "Stay connected with friends and family"
            ])
            
            # Limit recommendations to prevent overwhelming the user
            return recommendations[:8]
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return [
                "Maintain healthy lifestyle habits",
                "Stay hydrated and get adequate rest",
                "Monitor your health regularly",
                "Consult healthcare provider if concerns arise"
            ]
    
    def get_detailed_health_report(self, health_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            # Get overall assessment
            assessment = self.calculate_health_score(health_metrics)
            
            # Get recommendations
            recommendations = self.generate_recommendations(health_metrics)
            
            # Detailed component analysis
            detailed_analysis = {
                'cardiovascular': {
                    'heart_rate': health_metrics.get('heartRate', 'N/A'),
                    'heart_rate_variability': health_metrics.get('heartRateVariability', 'N/A'),
                    'assessment': 'Normal' if assessment['component_scores']['cardiovascular'] > 0.7 else 'Needs Attention'
                },
                'respiratory': {
                    'respiratory_rate': health_metrics.get('respiratoryRate', 'N/A'),
                    'breathing_pattern': health_metrics.get('breathingPattern', 'N/A'),
                    'assessment': 'Normal' if assessment['component_scores']['respiratory'] > 0.7 else 'Needs Attention'
                },
                'psychological': {
                    'stress_level': health_metrics.get('stressLevel', 'N/A'),
                    'emotion': health_metrics.get('emotion', 'N/A'),
                    'assessment': 'Good' if assessment['component_scores']['psychological'] > 0.7 else 'Monitor'
                }
            }
            
            return {
                'overall_assessment': assessment,
                'detailed_analysis': detailed_analysis,
                'recommendations': recommendations,
                'report_generated': datetime.now().isoformat(),
                'next_assessment_recommended': (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Detailed report generation error: {e}")
            return {
                'overall_assessment': self._get_error_assessment(),
                'detailed_analysis': {},
                'recommendations': ["Health report generation failed"],
                'report_generated': datetime.now().isoformat()
            }