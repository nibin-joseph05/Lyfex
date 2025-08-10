import cv2
import numpy as np
import scipy.signal
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class RespiratoryRateDetector:
    def __init__(self, buffer_size: int = 300, fps: int = 30):
        """
        Initialize respiratory rate detector
        
        Args:
            buffer_size: Number of frames to store for analysis
            fps: Frames per second for timing calculations
        """
        self.buffer_size = buffer_size
        self.fps = fps
        self.chest_motion_history = deque(maxlen=buffer_size)
        self.nose_motion_history = deque(maxlen=buffer_size)
        self.respiratory_rates = deque(maxlen=5)
        self.timestamps = deque(maxlen=buffer_size)
        
        # Normal respiratory rate range (12-20 breaths per minute)
        self.min_rr = 8   # Minimum respiratory rate
        self.max_rr = 30  # Maximum respiratory rate
        self.min_freq = self.min_rr / 60.0
        self.max_freq = self.max_rr / 60.0
        
        logger.info(f"Respiratory rate detector initialized (buffer: {buffer_size}, fps: {fps})")
    
    def extract_breathing_regions(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Extract regions for breathing analysis"""
        regions = {}
        
        try:
            if landmarks is not None and len(landmarks) >= 68:
                # Nose region for nostril movement analysis
                nose_points = landmarks[27:36]  # Nose landmarks
                if len(nose_points) > 0:
                    x_min, y_min = np.min(nose_points, axis=0)
                    x_max, y_max = np.max(nose_points, axis=0)
                    
                    # Expand region slightly
                    padding = 10
                    x_min = max(0, int(x_min - padding))
                    y_min = max(0, int(y_min - padding))
                    x_max = min(face_roi.shape[1], int(x_max + padding))
                    y_max = min(face_roi.shape[0], int(y_max + padding))
                    
                    regions['nose'] = face_roi[y_min:y_max, x_min:x_max]
                
                # Mouth region for mouth breathing
                mouth_points = landmarks[48:68]
                if len(mouth_points) > 0:
                    x_min, y_min = np.min(mouth_points, axis=0)
                    x_max, y_max = np.max(mouth_points, axis=0)
                    
                    padding = 15
                    x_min = max(0, int(x_min - padding))
                    y_min = max(0, int(y_min - padding))
                    x_max = min(face_roi.shape[1], int(x_max + padding))
                    y_max = min(face_roi.shape[0], int(y_max + padding))
                    
                    regions['mouth'] = face_roi[y_min:y_max, x_min:x_max]
            else:
                # Fallback regions
                h, w = face_roi.shape[:2]
                regions['nose'] = face_roi[h//3:2*h//3, w//3:2*w//3]
                regions['mouth'] = face_roi[2*h//3:h, w//4:3*w//4]
            
            # Chest region estimation (below face)
            h, w = face_roi.shape[:2]
            regions['chest'] = face_roi[3*h//4:h, w//4:3*w//4]
            
        except Exception as e:
            logger.error(f"Breathing region extraction failed: {e}")
            h, w = face_roi.shape[:2]
            regions = {
                'nose': face_roi[h//3:2*h//3, w//3:2*w//3],
                'mouth': face_roi[2*h//3:h, w//4:3*w//4],
                'chest': face_roi[3*h//4:h, w//4:3*w//4]
            }
        
        return regions
    
    def analyze_nostril_motion(self, nose_roi: np.ndarray) -> float:
        """Analyze nostril movement for breathing detection"""
        try:
            if nose_roi is None or nose_roi.size == 0:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(nose_roi, cv2.COLOR_BGR2GRAY) if len(nose_roi.shape) == 3 else nose_roi
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate variance in the nostril region
            # Higher variance indicates nostril dilation/contraction
            nostril_variance = np.var(blurred)
            
            # Also analyze pixel intensity changes
            mean_intensity = np.mean(blurred)
            
            # Combine metrics
            motion_signal = nostril_variance * 0.7 + mean_intensity * 0.3
            
            return float(motion_signal)
            
        except Exception as e:
            logger.error(f"Nostril motion analysis failed: {e}")
            return 0.0
    
    def analyze_mouth_breathing(self, mouth_roi: np.ndarray) -> float:
        """Analyze mouth opening for mouth breathing detection"""
        try:
            if mouth_roi is None or mouth_roi.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY) if len(mouth_roi.shape) == 3 else mouth_roi
            
            # Detect mouth opening by analyzing dark regions (mouth cavity)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate the ratio of dark pixels (mouth opening)
            dark_pixel_ratio = np.sum(binary == 0) / binary.size
            
            # Also consider mouth region variance
            mouth_variance = np.var(gray)
            
            # Combine metrics
            breathing_signal = dark_pixel_ratio * 0.6 + (mouth_variance / 1000) * 0.4
            
            return float(breathing_signal)
            
        except Exception as e:
            logger.error(f"Mouth breathing analysis failed: {e}")
            return 0.0
    
    def analyze_chest_movement(self, chest_roi: np.ndarray) -> float:
        """Analyze chest movement (limited in facial video)"""
        try:
            if chest_roi is None or chest_roi.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY) if len(chest_roi.shape) == 3 else chest_roi
            
            # Calculate optical flow magnitude as chest movement indicator
            # This is limited since we're analyzing face region
            chest_variance = np.var(gray)
            mean_intensity = np.mean(gray)
            
            # Simple chest movement estimation
            movement_signal = chest_variance * 0.8 + mean_intensity * 0.2
            
            return float(movement_signal)
            
        except Exception as e:
            logger.error(f"Chest movement analysis failed: {e}")
            return 0.0
    
    def preprocess_breathing_signal(self, signal: List[float]) -> np.ndarray:
        """Preprocess breathing signal for respiratory rate calculation"""
        try:
            signal_array = np.array(signal)
            
            if len(signal_array) < 30:
                return signal_array
            
            # Remove DC component
            detrended = scipy.signal.detrend(signal_array)
            
            # Normalize
            if np.std(detrended) > 0:
                normalized = (detrended - np.mean(detrended)) / np.std(detrended)
            else:
                normalized = detrended
            
            # Apply bandpass filter for breathing rate range
            nyquist_freq = self.fps / 2
            low_freq = self.min_freq / nyquist_freq
            high_freq = self.max_freq / nyquist_freq
            
            # Clamp frequencies
            low_freq = max(0.01, min(low_freq, 0.99))
            high_freq = max(low_freq + 0.01, min(high_freq, 0.99))
            
            # Butterworth bandpass filter
            b, a = scipy.signal.butter(3, [low_freq, high_freq], btype='band')
            filtered_signal = scipy.signal.filtfilt(b, a, normalized)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Breathing signal preprocessing failed: {e}")
            return np.array(signal)
    
    def calculate_respiratory_rate_fft(self, signal: np.ndarray) -> Tuple[float, float]:
        """Calculate respiratory rate using FFT"""
        try:
            if len(signal) < 60:  # Need at least 2 seconds
                return 0.0, 0.0
            
            # Perform FFT
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/self.fps)
            
            # Get positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # Focus on breathing frequency range
            breathing_mask = (positive_freqs >= self.min_freq) & (positive_freqs <= self.max_freq)
            breathing_freqs = positive_freqs[breathing_mask]
            breathing_fft = positive_fft[breathing_mask]
            
            if len(breathing_fft) == 0:
                return 0.0, 0.0
            
            # Find dominant frequency
            peak_idx = np.argmax(breathing_fft)
            dominant_freq = breathing_freqs[peak_idx]
            peak_amplitude = breathing_fft[peak_idx]
            
            # Convert to breaths per minute
            respiratory_rate = dominant_freq * 60
            
            # Calculate confidence
            mean_amplitude = np.mean(breathing_fft)
            confidence = min(peak_amplitude / (mean_amplitude + 1e-8), 5.0) / 5.0
            
            return float(respiratory_rate), float(confidence)
            
        except Exception as e:
            logger.error(f"FFT respiratory rate calculation failed: {e}")
            return 0.0, 0.0
    
    def calculate_respiratory_rate_peaks(self, signal: np.ndarray) -> Tuple[float, float]:
        """Calculate respiratory rate using peak detection"""
        try:
            if len(signal) < 60:
                return 0.0, 0.0
            
            # Find peaks
            signal_std = np.std(signal)
            height_threshold = signal_std * 0.3
            
            peaks, properties = scipy.signal.find_peaks(
                signal,
                height=height_threshold,
                distance=self.fps//2  # Minimum 0.5 second between breaths
            )
            
            if len(peaks) < 2:
                return 0.0, 0.0
            
            # Calculate intervals between peaks
            peak_intervals = np.diff(peaks) / self.fps
            
            # Filter realistic intervals
            valid_intervals = peak_intervals[
                (peak_intervals >= 60/self.max_rr) & 
                (peak_intervals <= 60/self.min_rr)
            ]
            
            if len(valid_intervals) == 0:
                return 0.0, 0.0
            
            # Calculate respiratory rate
            mean_interval = np.mean(valid_intervals)
            respiratory_rate = 60 / mean_interval
            
            # Calculate confidence
            interval_std = np.std(valid_intervals)
            confidence = max(0, 1 - interval_std / mean_interval)
            
            return float(respiratory_rate), float(confidence)
            
        except Exception as e:
            logger.error(f"Peak-based respiratory rate calculation failed: {e}")
            return 0.0, 0.0
    
    def analyze_breathing_pattern(self, signal: np.ndarray) -> Dict:
        """Analyze breathing pattern characteristics"""
        try:
            if len(signal) < 60:
                return {'pattern': 'Unknown', 'regularity': 0.0, 'depth': 0.0}
            
            # Find peaks and troughs
            peaks, _ = scipy.signal.find_peaks(signal, distance=self.fps//2)
            troughs, _ = scipy.signal.find_peaks(-signal, distance=self.fps//2)
            
            pattern_info = {
                'pattern': 'Regular',
                'regularity': 0.0,
                'depth': 0.0,
                'rhythm': 'Normal'
            }
            
            if len(peaks) >= 3:
                # Analyze regularity
                peak_intervals = np.diff(peaks)
                interval_cv = np.std(peak_intervals) / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 1
                
                pattern_info['regularity'] = max(0, 1 - interval_cv)
                
                # Classify pattern based on regularity
                if interval_cv < 0.1:
                    pattern_info['pattern'] = 'Very Regular'
                elif interval_cv < 0.2:
                    pattern_info['pattern'] = 'Regular'
                elif interval_cv < 0.4:
                    pattern_info['pattern'] = 'Irregular'
                else:
                    pattern_info['pattern'] = 'Very Irregular'
            
            # Analyze breathing depth
            if len(peaks) > 0 and len(troughs) > 0:
                peak_values = signal[peaks] if len(peaks) > 0 else [0]
                trough_values = signal[troughs] if len(troughs) > 0 else [0]
                
                breathing_depth = np.mean(peak_values) - np.mean(trough_values)
                pattern_info['depth'] = float(abs(breathing_depth))
                
                # Classify depth
                if breathing_depth > np.std(signal):
                    pattern_info['rhythm'] = 'Deep'
                elif breathing_depth < np.std(signal) * 0.5:
                    pattern_info['rhythm'] = 'Shallow'
                else:
                    pattern_info['rhythm'] = 'Normal'
            
            return pattern_info
            
        except Exception as e:
            logger.error(f"Breathing pattern analysis failed: {e}")
            return {'pattern': 'Unknown', 'regularity': 0.0, 'depth': 0.0, 'rhythm': 'Unknown'}
    
    def analyze_single_frame(self, frame: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Analyze single frame for respiratory rate"""
        try:
            # Extract breathing regions
            regions = self.extract_breathing_regions(frame, landmarks)
            
            # Analyze different breathing signals
            nostril_signal = self.analyze_nostril_motion(regions.get('nose'))
            mouth_signal = self.analyze_mouth_breathing(regions.get('mouth'))
            chest_signal = self.analyze_chest_movement(regions.get('chest'))
            
            # Combine signals (weighted average)
            combined_signal = nostril_signal * 0.5 + mouth_signal * 0.3 + chest_signal * 0.2
            
            # Add to history
            self.nose_motion_history.append(combined_signal)
            self.timestamps.append(time.time())
            
            # Need sufficient data for analysis
            if len(self.nose_motion_history) < 90:  # 3 seconds minimum
                return {
                    'respiratory_rate': 'Analyzing...',
                    'pattern': 'Building buffer',
                    'confidence': 0.0,
                    'samples_collected': len(self.nose_motion_history),
                    'samples_needed': 90
                }
            
            # Convert to list for processing
            signal_list = list(self.nose_motion_history)
            
            # Preprocess signal
            processed_signal = self.preprocess_breathing_signal(signal_list)
            
            # Calculate respiratory rate using both methods
            rr_fft, conf_fft = self.calculate_respiratory_rate_fft(processed_signal)
            rr_peaks, conf_peaks = self.calculate_respiratory_rate_peaks(processed_signal)
            
            # Use method with higher confidence
            if conf_fft > conf_peaks:
                respiratory_rate = rr_fft
                confidence = conf_fft
                method = "FFT"
            else:
                respiratory_rate = rr_peaks
                confidence = conf_peaks
                method = "Peaks"
            
            # Analyze breathing pattern
            pattern_info = self.analyze_breathing_pattern(processed_signal)
            
            # Validate respiratory rate
            if not (self.min_rr <= respiratory_rate <= self.max_rr):
                respiratory_rate_str = "Out of Range"
                confidence = 0.0
            else:
                # Store valid measurement
                self.respiratory_rates.append(respiratory_rate)
                
                # Use moving average for stability
                if len(self.respiratory_rates) >= 3:
                    respiratory_rate = np.mean(list(self.respiratory_rates)[-3:])
                
                respiratory_rate_str = f"{int(round(respiratory_rate))} breaths/min"
            
            return {
                'respiratory_rate': respiratory_rate_str,
                'pattern': pattern_info['pattern'],
                'regularity': round(pattern_info['regularity'], 2),
                'breathing_depth': round(pattern_info['depth'], 2),
                'rhythm': pattern_info['rhythm'],
                'confidence': round(confidence, 2),
                'method_used': method,
                'signal_quality': self._assess_signal_quality(processed_signal, confidence)
            }
            
        except Exception as e:
            logger.error(f"Single frame respiratory analysis failed: {e}")
            return {
                'respiratory_rate': 'Error',
                'pattern': 'Unknown',
                'confidence': 0.0,
                'signal_quality': 'Poor'
            }
    
    def _assess_signal_quality(self, signal: np.ndarray, confidence: float) -> str:
        """Assess breathing signal quality"""
        try:
            if len(signal) < 30:
                return "Insufficient data"
            
            signal_power = np.var(signal)
            
            if confidence > 0.7 and signal_power > 0.1:
                return "Excellent"
            elif confidence > 0.5 and signal_power > 0.05:
                return "Good"
            elif confidence > 0.3:
                return "Fair"
            else:
                return "Poor"
                
        except Exception as e:
            logger.error(f"Signal quality assessment failed: {e}")
            return "Unknown"
    
    def reset_buffer(self):
        """Reset analysis buffer"""
        self.chest_motion_history.clear()
        self.nose_motion_history.clear()
        self.respiratory_rates.clear()
        self.timestamps.clear()
        logger.info("Respiratory rate detector buffer reset")