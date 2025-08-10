import cv2
import numpy as np
import scipy.signal
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class HeartRateDetector:
    def __init__(self, buffer_size: int = 300, fps: int = 30):
        """
        Initialize heart rate detector using photoplethysmography (PPG)
        
        Args:
            buffer_size: Number of frames to store for analysis (10 seconds at 30fps)
            fps: Frames per second for timing calculations
        """
        self.buffer_size = buffer_size
        self.fps = fps
        self.roi_history = deque(maxlen=buffer_size)
        self.heart_rates = deque(maxlen=10)  # Store last 10 HR measurements
        self.timestamps = deque(maxlen=buffer_size)
        
        # Filter parameters for heart rate range (48-210 BPM)
        self.min_hr = 48
        self.max_hr = 210
        self.min_freq = self.min_hr / 60.0  # 0.8 Hz
        self.max_freq = self.max_hr / 60.0  # 3.5 Hz
        
        logger.info(f"Heart rate detector initialized (buffer: {buffer_size}, fps: {fps})")
    
    def extract_forehead_roi(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract forehead region - best area for PPG signal detection"""
        try:
            if landmarks is not None and len(landmarks) >= 68:
                # Use landmarks to identify forehead region more precisely
                # Points around forehead area
                forehead_points = landmarks[19:24]  # Eyebrow area
                
                if len(forehead_points) > 0:
                    x_min, y_min = np.min(forehead_points, axis=0)
                    x_max, y_max = np.max(forehead_points, axis=0)
                    
                    # Expand the region upward for forehead
                    h, w = face_roi.shape[:2]
                    y_min = max(0, int(y_min - 30))  # Move up for forehead
                    y_max = max(y_min + 40, int(y_max + 10))  # Ensure minimum height
                    x_min = max(0, int(x_min - 20))
                    x_max = min(w, int(x_max + 20))
                    
                    forehead_roi = face_roi[y_min:y_max, x_min:x_max]
                    
                    # Validate ROI size
                    if forehead_roi.shape[0] > 20 and forehead_roi.shape[1] > 20:
                        return forehead_roi
            
            # Fallback: use upper portion of face as forehead
            h, w = face_roi.shape[:2]
            forehead_roi = face_roi[0:h//3, w//4:3*w//4]
            
            return forehead_roi
            
        except Exception as e:
            logger.error(f"Forehead ROI extraction failed: {e}")
            # Return central upper region as last resort
            h, w = face_roi.shape[:2]
            return face_roi[0:h//4, w//3:2*w//3]
    
    def extract_ppg_signal(self, roi: np.ndarray) -> float:
        """Extract PPG signal from ROI by analyzing green channel"""
        try:
            if roi is None or roi.size == 0:
                return 0.0
            
            # Convert to different color spaces and extract relevant channels
            # Green channel is most sensitive to blood volume changes
            if len(roi.shape) == 3:
                # Extract green channel
                green_channel = roi[:, :, 1]
                
                # Also try RGB to YUV conversion for better signal
                yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
                y_channel = yuv[:, :, 0]  # Luminance channel
                
                # Combine signals (green channel primary, Y channel secondary)
                green_mean = np.mean(green_channel)
                y_mean = np.mean(y_channel)
                
                # Weighted combination
                signal_value = 0.7 * green_mean + 0.3 * y_mean
            else:
                # Grayscale image
                signal_value = np.mean(roi)
            
            return float(signal_value)
            
        except Exception as e:
            logger.error(f"PPG signal extraction failed: {e}")
            return 0.0
    
    def preprocess_signal(self, signal: List[float]) -> np.ndarray:
        """Preprocess PPG signal for heart rate calculation"""
        try:
            signal_array = np.array(signal)
            
            if len(signal_array) < 30:  # Need minimum samples
                return signal_array
            
            # Remove DC component (detrend)
            detrended = scipy.signal.detrend(signal_array)
            
            # Normalize signal
            if np.std(detrended) > 0:
                normalized = (detrended - np.mean(detrended)) / np.std(detrended)
            else:
                normalized = detrended
            
            # Apply bandpass filter for heart rate range
            nyquist_freq = self.fps / 2
            low_freq = self.min_freq / nyquist_freq
            high_freq = self.max_freq / nyquist_freq
            
            # Ensure frequencies are in valid range
            low_freq = max(0.01, min(low_freq, 0.99))
            high_freq = max(low_freq + 0.01, min(high_freq, 0.99))
            
            # Butterworth bandpass filter
            b, a = scipy.signal.butter(4, [low_freq, high_freq], btype='band')
            filtered_signal = scipy.signal.filtfilt(b, a, normalized)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Signal preprocessing failed: {e}")
            return np.array(signal)
    
    def calculate_heart_rate_fft(self, signal: np.ndarray) -> Tuple[float, float]:
        """Calculate heart rate using FFT analysis"""
        try:
            if len(signal) < 60:  # Need at least 2 seconds of data
                return 0.0, 0.0
            
            # Perform FFT
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/self.fps)
            
            # Get positive frequencies only
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # Focus on heart rate frequency range
            hr_mask = (positive_freqs >= self.min_freq) & (positive_freqs <= self.max_freq)
            hr_freqs = positive_freqs[hr_mask]
            hr_fft = positive_fft[hr_mask]
            
            if len(hr_fft) == 0:
                return 0.0, 0.0
            
            # Find dominant frequency
            peak_idx = np.argmax(hr_fft)
            dominant_freq = hr_freqs[peak_idx]
            peak_amplitude = hr_fft[peak_idx]
            
            # Convert to BPM
            heart_rate = dominant_freq * 60
            
            # Calculate confidence based on peak prominence
            mean_amplitude = np.mean(hr_fft)
            confidence = min(peak_amplitude / (mean_amplitude + 1e-8), 5.0) / 5.0
            
            return float(heart_rate), float(confidence)
            
        except Exception as e:
            logger.error(f"FFT heart rate calculation failed: {e}")
            return 0.0, 0.0
    
    def calculate_heart_rate_peaks(self, signal: np.ndarray) -> Tuple[float, float]:
        """Calculate heart rate using peak detection method"""
        try:
            if len(signal) < 60:
                return 0.0, 0.0
            
            # Find peaks in the signal
            # Use adaptive thresholding
            signal_std = np.std(signal)
            height_threshold = signal_std * 0.5
            
            peaks, properties = scipy.signal.find_peaks(
                signal,
                height=height_threshold,
                distance=self.fps//3  # Minimum distance between peaks (200ms)
            )
            
            if len(peaks) < 2:
                return 0.0, 0.0
            
            # Calculate intervals between peaks
            peak_intervals = np.diff(peaks) / self.fps  # Convert to seconds
            
            # Filter out unrealistic intervals
            valid_intervals = peak_intervals[
                (peak_intervals >= 60/self.max_hr) & 
                (peak_intervals <= 60/self.min_hr)
            ]
            
            if len(valid_intervals) == 0:
                return 0.0, 0.0
            
            # Calculate heart rate from mean interval
            mean_interval = np.mean(valid_intervals)
            heart_rate = 60 / mean_interval
            
            # Calculate confidence based on interval consistency
            interval_std = np.std(valid_intervals)
            confidence = max(0, 1 - interval_std / mean_interval)
            
            return float(heart_rate), float(confidence)
            
        except Exception as e:
            logger.error(f"Peak-based heart rate calculation failed: {e}")
            return 0.0, 0.0
    
    def calculate_hrv_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """Calculate Heart Rate Variability metrics"""
        try:
            # Find R-R intervals using peak detection
            peaks, _ = scipy.signal.find_peaks(
                signal,
                distance=self.fps//3,
                height=np.std(signal) * 0.3
            )
            
            if len(peaks) < 3:
                return {'hrv_score': 0.0, 'rmssd': 0.0, 'sdnn': 0.0}
            
            # Calculate R-R intervals in milliseconds
            rr_intervals = np.diff(peaks) / self.fps * 1000
            
            # Filter realistic intervals (300-2000ms)
            valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
            
            if len(valid_rr) < 2:
                return {'hrv_score': 0.0, 'rmssd': 0.0, 'sdnn': 0.0}
            
            # SDNN: Standard deviation of NN intervals
            sdnn = np.std(valid_rr)
            
            # RMSSD: Root mean square of successive differences
            successive_diffs = np.diff(valid_rr)
            rmssd = np.sqrt(np.mean(successive_diffs ** 2))
            
            # Simple HRV score (normalized)
            hrv_score = min(rmssd / 50, 1.0)  # Normalize to 0-1 range
            
            return {
                'hrv_score': float(hrv_score),
                'rmssd': float(rmssd),
                'sdnn': float(sdnn)
            }
            
        except Exception as e:
            logger.error(f"HRV calculation failed: {e}")
            return {'hrv_score': 0.0, 'rmssd': 0.0, 'sdnn': 0.0}
    
    def analyze_single_frame(self, face_roi: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """Analyze single frame for heart rate (limited accuracy)"""
        try:
            # Extract forehead ROI
            forehead_roi = self.extract_forehead_roi(face_roi, landmarks)
            
            # Extract PPG signal
            signal_value = self.extract_ppg_signal(forehead_roi)
            
            # Add to history
            self.roi_history.append(signal_value)
            self.timestamps.append(time.time())
            
            # Need sufficient history for analysis
            if len(self.roi_history) < 60:  # 2 seconds minimum
                return {
                    'heart_rate': 'Analyzing...',
                    'confidence': 0.0,
                    'hrv_score': 0.0,
                    'signal_quality': 'Building buffer',
                    'samples_collected': len(self.roi_history),
                    'samples_needed': 60
                }
            
            # Convert to list for processing
            signal_list = list(self.roi_history)
            
            # Preprocess signal
            processed_signal = self.preprocess_signal(signal_list)
            
            # Calculate heart rate using both methods
            hr_fft, conf_fft = self.calculate_heart_rate_fft(processed_signal)
            hr_peaks, conf_peaks = self.calculate_heart_rate_peaks(processed_signal)
            
            # Use the method with higher confidence
            if conf_fft > conf_peaks:
                heart_rate = hr_fft
                confidence = conf_fft
                method = "FFT"
            else:
                heart_rate = hr_peaks
                confidence = conf_peaks
                method = "Peaks"
            
            # Calculate HRV if enough data
            hrv_metrics = self.calculate_hrv_metrics(processed_signal)
            
            # Validate heart rate
            if not (self.min_hr <= heart_rate <= self.max_hr):
                heart_rate_str = "Out of Range"
                confidence = 0.0
            else:
                # Store valid heart rate
                self.heart_rates.append(heart_rate)
                
                # Use moving average for stability
                if len(self.heart_rates) >= 3:
                    heart_rate = np.mean(list(self.heart_rates)[-3:])
                
                heart_rate_str = f"{int(round(heart_rate))} bpm"
            
            # Assess signal quality
            signal_quality = self._assess_signal_quality(processed_signal, confidence)
            
            return {
                'heart_rate': heart_rate_str,
                'confidence': round(confidence, 2),
                'hrv_score': round(hrv_metrics['hrv_score'], 2),
                'signal_quality': signal_quality,
                'method_used': method,
                'rmssd': round(hrv_metrics['rmssd'], 1),
                'sdnn': round(hrv_metrics['sdnn'], 1)
            }
            
        except Exception as e:
            logger.error(f"Single frame heart rate analysis failed: {e}")
            return {
                'heart_rate': 'Error',
                'confidence': 0.0,
                'hrv_score': 0.0,
                'signal_quality': 'Poor'
            }
    
    def quick_heart_rate_estimate(self, face_roi: np.ndarray) -> str:
        """Quick heart rate estimate for rapid scanning"""
        try:
            # Extract simple forehead region
            h, w = face_roi.shape[:2]
            forehead = face_roi[0:h//4, w//4:3*w//4]
            
            # Simple color analysis
            if len(forehead.shape) == 3:
                green_mean = np.mean(forehead[:, :, 1])
                # Very rough estimation based on green channel intensity
                # This is not accurate but provides immediate feedback
                estimated_hr = int(60 + (green_mean - 100) / 2)
                estimated_hr = max(60, min(100, estimated_hr))  # Clamp to reasonable range
                return f"~{estimated_hr} bpm (estimate)"
            else:
                return "Face detected"
                
        except Exception as e:
            logger.error(f"Quick heart rate estimate failed: {e}")
            return "Unable to estimate"
    
    def _assess_signal_quality(self, signal: np.ndarray, confidence: float) -> str:
        """Assess the quality of PPG signal for heart rate detection"""
        try:
            if len(signal) < 30:
                return "Insufficient data"
            
            # Signal-to-noise ratio assessment
            signal_power = np.var(signal)
            
            if confidence > 0.8 and signal_power > 0.1:
                return "Excellent"
            elif confidence > 0.6 and signal_power > 0.05:
                return "Good"
            elif confidence > 0.4:
                return "Fair"
            else:
                return "Poor"
                
        except Exception as e:
            logger.error(f"Signal quality assessment failed: {e}")
            return "Unknown"
    
    def reset_buffer(self):
        """Reset the signal buffer"""
        self.roi_history.clear()
        self.heart_rates.clear()
        self.timestamps.clear()
        logger.info("Heart rate detector buffer reset")