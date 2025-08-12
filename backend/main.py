from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import logging
import json
import base64
import time
from datetime import datetime

# Import our health detection modules
from app.detectors.heart_rate_detector import HeartRateDetector
from app.detectors.respiratory_detector import RespiratoryRateDetector
from app.detectors.emotion_detector import EmotionDetector
from app.detectors.stress_detector import StressDetector
from app.detectors.fatigue_detector import FatigueDetector
from app.detectors.neurological_detector import NeurologicalDetector
from app.detectors.skin_analyzer import SkinAnalyzer
from app.analyzers.health_assessor import HealthAssessor
from app.utils.image_processor import ImageProcessor
from app.utils.face_detector import FaceDetector

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LyfeX Health Monitor API", version="1.0.0")

frontend_ip = os.getenv("FRONTEND_IP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all detectors for HTTP POST endpoints (single image)
class HealthMonitorSystem:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()
        self.heart_rate_detector = HeartRateDetector()
        self.respiratory_detector = RespiratoryRateDetector()
        self.emotion_detector = EmotionDetector()
        self.stress_detector = StressDetector()
        self.fatigue_detector = FatigueDetector()
        self.neurological_detector = NeurologicalDetector()
        self.skin_analyzer = SkinAnalyzer()
        self.health_assessor = HealthAssessor()

        logger.info("Health Monitor System initialized successfully for HTTP endpoints")

# Global system instance for HTTP POST (single image)
health_system = HealthMonitorSystem()

@app.get("/")
def read_root():
    return {"message": "LyfeX Health Monitor Backend is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "detectors_loaded": True,
        "message": "All health detection systems operational"
    }

@app.post("/analyze")
async def analyze_health(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Processing image: {file.filename}")
        
        # Detect face and landmarks
        faces, landmarks = health_system.face_detector.detect_face_and_landmarks(frame)
        
        if len(faces) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No face detected",
                    "message": "Please ensure your face is clearly visible in the image"
                }
            )
        
        # Use the first detected face
        primary_face = faces[0]
        primary_landmarks = landmarks[0] if landmarks else None
        
        # Initialize results
        health_metrics = {}
        alerts = []
        
        # Process image through preprocessor
        processed_frame = health_system.image_processor.preprocess_image(frame)
        face_roi = health_system.image_processor.extract_face_roi(processed_frame, primary_face)
        
        # Cardiovascular Analysis
        try:
            heart_rate_data = health_system.heart_rate_detector.analyze_single_frame(
                face_roi, primary_landmarks
            )
            health_metrics['heartRate'] = heart_rate_data['heart_rate']
            health_metrics['heartRateVariability'] = heart_rate_data['hrv_score']
        except Exception as e:
            logger.error(f"Heart rate detection error: {e}")
            health_metrics['heartRate'] = 'Analysis Error'
            health_metrics['heartRateVariability'] = 'N/A'
        
        # Respiratory Analysis
        try:
            respiratory_data = health_system.respiratory_detector.analyze_single_frame(
                frame, primary_landmarks
            )
            health_metrics['respiratoryRate'] = respiratory_data['respiratory_rate']
            health_metrics['breathingPattern'] = respiratory_data['pattern']
        except Exception as e:
            logger.error(f"Respiratory detection error: {e}")
            health_metrics['respiratoryRate'] = 'Analysis Error'
            health_metrics['breathingPattern'] = 'N/A'
        
        # Emotion Analysis
        try:
            emotion_data = health_system.emotion_detector.detect_emotion(face_roi)
            health_metrics['emotion'] = emotion_data['primary_emotion']
            health_metrics['emotionConfidence'] = emotion_data['confidence']
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            health_metrics['emotion'] = 'Detection Error'
            health_metrics['emotionConfidence'] = 0.0
        
        # Stress Level Analysis
        try:
            stress_data = health_system.stress_detector.analyze_stress_indicators(
                face_roi, primary_landmarks, health_metrics
            )
            health_metrics['stressLevel'] = stress_data['stress_level']
            health_metrics['stressFactors'] = stress_data['factors']
        except Exception as e:
            logger.error(f"Stress detection error: {e}")
            health_metrics['stressLevel'] = 'Analysis Error'
            health_metrics['stressFactors'] = []
        
        # Fatigue Analysis
        try:
            fatigue_data = health_system.fatigue_detector.assess_fatigue(
                face_roi, primary_landmarks
            )
            health_metrics['fatigue'] = fatigue_data['fatigue_level']
            health_metrics['alertnessScore'] = fatigue_data['alertness_score']
        except Exception as e:
            logger.error(f"Fatigue detection error: {e}")
            health_metrics['fatigue'] = 'Analysis Error'
            health_metrics['alertnessScore'] = 0.0
        
        # Neurological Assessment
        try:
            neuro_data = health_system.neurological_detector.assess_neurological_health(
                face_roi, primary_landmarks
            )
            health_metrics['facialAsymmetry'] = neuro_data['facial_asymmetry']
            health_metrics['tremor'] = neuro_data['tremor_detected']
            health_metrics['eyeMovement'] = neuro_data['eye_movement_analysis']
        except Exception as e:
            logger.error(f"Neurological assessment error: {e}")
            health_metrics['facialAsymmetry'] = 'Analysis Error'
            health_metrics['tremor'] = 'N/A'
            health_metrics['eyeMovement'] = 'N/A'
        
        # Skin Analysis
        try:
            skin_data = health_system.skin_analyzer.analyze_skin_health(face_roi)
            health_metrics['skinAnalysis'] = skin_data['overall_health']
            health_metrics['skinColor'] = skin_data['color_analysis']
            health_metrics['hydrationStatus'] = skin_data['hydration_estimate']
        except Exception as e:
            logger.error(f"Skin analysis error: {e}")
            health_metrics['skinAnalysis'] = 'Analysis Error'
            health_metrics['skinColor'] = 'N/A'
            health_metrics['hydrationStatus'] = 'N/A'
        
        # Generate overall health assessment
        try:
            overall_assessment = health_system.health_assessor.calculate_health_score(health_metrics)
            health_metrics['overallHealthScore'] = overall_assessment['score']
            health_metrics['healthStatus'] = overall_assessment['status']
            alerts.extend(overall_assessment['alerts'])
        except Exception as e:
            logger.error(f"Health assessment error: {e}")
            health_metrics['overallHealthScore'] = 0.0
            health_metrics['healthStatus'] = 'Assessment Error'
        
        # Generate recommendations
        try:
            recommendations = health_system.health_assessor.generate_recommendations(health_metrics)
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            recommendations = ["Health recommendations unavailable at this time"]
        
        # Prepare response
        response_data = {
            **health_metrics,
            'alerts': alerts,
            'recommendations': recommendations,
            'analysis_timestamp': health_system.image_processor.get_current_timestamp(),
            'face_detected': True,
            'analysis_quality': 'Good' if len(alerts) == 0 else 'Warning'
        }
        
        logger.info("Health analysis completed successfully")
        return response_data
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Health analysis failed: {str(e)}"
        )

@app.post("/quick-scan")
async def quick_health_scan(file: UploadFile = File(...)):
    """Quick scan for basic vital signs only"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        faces, landmarks = health_system.face_detector.detect_face_and_landmarks(frame)
        
        if len(faces) == 0:
            return {"error": "No face detected"}
        
        face_roi = health_system.image_processor.extract_face_roi(frame, faces[0])
        
        # Quick analysis - only basic metrics
        heart_rate = health_system.heart_rate_detector.quick_heart_rate_estimate(face_roi)
        emotion = health_system.emotion_detector.detect_emotion(face_roi)['primary_emotion']
        stress = health_system.stress_detector.quick_stress_assessment(face_roi, landmarks[0])
        
        return {
            'heartRate': heart_rate,
            'emotion': emotion,
            'stressLevel': stress,
            'scanType': 'quick',
            'timestamp': health_system.image_processor.get_current_timestamp()
        }
        
    except Exception as e:
        logger.error(f"Quick scan failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quick scan failed: {str(e)}")

@app.post("/detailed-report")
async def generate_detailed_report(file: UploadFile = File(...)):
    """Generate comprehensive health report"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        faces, landmarks = health_system.face_detector.detect_face_and_landmarks(frame)
        
        if len(faces) == 0:
            return {"error": "No face detected"}
        
        primary_face = faces[0]
        primary_landmarks = landmarks[0] if landmarks else None
        processed_frame = health_system.image_processor.preprocess_image(frame)
        face_roi = health_system.image_processor.extract_face_roi(processed_frame, primary_face)
        
        # Collect all health metrics
        health_metrics = {}
        
        # All health analyses (same as /analyze endpoint)
        try:
            heart_rate_data = health_system.heart_rate_detector.analyze_single_frame(face_roi, primary_landmarks)
            health_metrics['heartRate'] = heart_rate_data['heart_rate']
            health_metrics['heartRateVariability'] = heart_rate_data['hrv_score']
        except:
            health_metrics['heartRate'] = 'Analysis Error'
            health_metrics['heartRateVariability'] = 'N/A'
        
        try:
            respiratory_data = health_system.respiratory_detector.analyze_single_frame(frame, primary_landmarks)
            health_metrics['respiratoryRate'] = respiratory_data['respiratory_rate']
            health_metrics['breathingPattern'] = respiratory_data['pattern']
        except:
            health_metrics['respiratoryRate'] = 'Analysis Error'
            health_metrics['breathingPattern'] = 'N/A'
        
        try:
            emotion_data = health_system.emotion_detector.detect_emotion(face_roi)
            health_metrics['emotion'] = emotion_data['primary_emotion']
            health_metrics['emotionConfidence'] = emotion_data['confidence']
        except:
            health_metrics['emotion'] = 'Detection Error'
            health_metrics['emotionConfidence'] = 0.0
        
        try:
            stress_data = health_system.stress_detector.analyze_stress_indicators(face_roi, primary_landmarks, health_metrics)
            health_metrics['stressLevel'] = stress_data['stress_level']
            health_metrics['stressFactors'] = stress_data['factors']
        except:
            health_metrics['stressLevel'] = 'Analysis Error'
            health_metrics['stressFactors'] = []
        
        try:
            fatigue_data = health_system.fatigue_detector.assess_fatigue(face_roi, primary_landmarks)
            health_metrics['fatigue'] = fatigue_data['fatigue_level']
            health_metrics['alertnessScore'] = fatigue_data['alertness_score']
        except:
            health_metrics['fatigue'] = 'Analysis Error'
            health_metrics['alertnessScore'] = 0.0
        
        try:
            neuro_data = health_system.neurological_detector.assess_neurological_health(face_roi, primary_landmarks)
            health_metrics['facialAsymmetry'] = neuro_data['facial_asymmetry']
            health_metrics['tremor'] = neuro_data['tremor_detected']
            health_metrics['eyeMovement'] = neuro_data['eye_movement_analysis']
        except:
            health_metrics['facialAsymmetry'] = 'Analysis Error'
            health_metrics['tremor'] = 'N/A'
            health_metrics['eyeMovement'] = 'N/A'
        
        try:
            skin_data = health_system.skin_analyzer.analyze_skin_health(face_roi)
            health_metrics['skinAnalysis'] = skin_data['overall_health']
            health_metrics['skinColor'] = skin_data['color_analysis']
            health_metrics['hydrationStatus'] = skin_data['hydration_estimate']
        except:
            health_metrics['skinAnalysis'] = 'Analysis Error'
            health_metrics['skinColor'] = 'N/A'
            health_metrics['hydrationStatus'] = 'N/A'
        
        # Generate detailed report
        detailed_report = health_system.health_assessor.get_detailed_health_report(health_metrics)
        
        return detailed_report
        
    except Exception as e:
        logger.error(f"Detailed report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# --- WebSocket Endpoint for Real-time Video Stream Analysis ---

class VideoStreamHealthMonitor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()
        
        # These detectors will accumulate data across multiple frames
        self.heart_rate_detector = HeartRateDetector() 
        self.respiratory_detector = RespiratoryRateDetector()

        # These might operate on a per-frame basis or accumulate less
        self.emotion_detector = EmotionDetector()
        self.stress_detector = StressDetector()
        self.fatigue_detector = FatigueDetector()
        self.neurological_detector = NeurologicalDetector()
        self.skin_analyzer = SkinAnalyzer()
        self.health_assessor = HealthAssessor()
        
        logger.info("Video Stream Health Monitor initialized for real-time analysis")

    async def process_video_frame(self, frame):
        """Process a single video frame and return real-time metrics and face detection data"""
        realtime_metrics = {
            'heart_rate': None,
            'respiratory_rate': None,
            'stress_level': None,
            'emotion': None,
            'confidence': 0.0,
            'face_detected': False
        }
        
        face_detection = {
            'bounding_box': None,
            'landmarks': [],
            'quality': 'Searching...'
        }
        
        analysis_data = {}
        
        try:
            faces, landmarks = self.face_detector.detect_face_and_landmarks(frame)
            
            if len(faces) == 0:
                face_detection['quality'] = 'No Face'
                return {
                    'realtime_metrics': realtime_metrics,
                    'face_detection': face_detection,
                    'analysis_data': analysis_data,
                    'timestamp': datetime.now().isoformat()
                }

            primary_face = faces[0]
            primary_landmarks = landmarks[0] if landmarks else None
            processed_frame = self.image_processor.preprocess_image(frame)
            face_roi = self.image_processor.extract_face_roi(processed_frame, primary_face)

            # Update face detection data
            x, y, w, h = primary_face
            face_detection = {
                'bounding_box': {
                    'x': (x / frame.shape[1]) * 100,  # Convert to percentage
                    'y': (y / frame.shape[0]) * 100,
                    'width': (w / frame.shape[1]) * 100,
                    'height': (h / frame.shape[0]) * 100
                },
                'landmarks': primary_landmarks.tolist() if primary_landmarks is not None else [],
                'quality': 'Good'  # Start with Good, will be adjusted based on actual quality
            }
            
            realtime_metrics['face_detected'] = True

            # Assess face detection quality with more lenient criteria
            quality_metrics = self.face_detector.validate_face_quality(face_roi, primary_landmarks)
            
            # More lenient quality assessment
            quality_score = 0
            if w > 80 and h > 80:  # Reduced minimum size requirement
                quality_score += 1
            if quality_metrics.get('brightness_good', False):
                quality_score += 1
            if quality_metrics.get('blur_acceptable', False):
                quality_score += 1
            if quality_metrics.get('pose_frontal', True):  # Default to True if no landmarks
                quality_score += 1
                
            # Set quality based on score (more lenient)
            if quality_score >= 3:
                face_detection['quality'] = 'Good'
            elif quality_score >= 2:
                face_detection['quality'] = 'Acceptable'
            else:
                face_detection['quality'] = 'Poor'

            # Real-time heart rate analysis
            try:
                heart_rate_data = self.heart_rate_detector.analyze_single_frame(face_roi, primary_landmarks)
                confidence = 0.0
                
                if isinstance(heart_rate_data.get('heart_rate'), str):
                    # Parse string format like "75 bpm"
                    hr_str = heart_rate_data['heart_rate'].lower()
                    if 'bpm' in hr_str:
                        try:
                            hr_value = float(hr_str.split()[0])
                            if 40 <= hr_value <= 200:  # Reasonable HR range
                                realtime_metrics['heart_rate'] = hr_value
                                confidence = 0.8  # High confidence for valid HR
                        except (ValueError, IndexError):
                            realtime_metrics['heart_rate'] = None
                    else:
                        realtime_metrics['heart_rate'] = None
                elif isinstance(heart_rate_data.get('heart_rate'), (int, float)):
                    hr_value = float(heart_rate_data['heart_rate'])
                    if 40 <= hr_value <= 200:  # Reasonable HR range
                        realtime_metrics['heart_rate'] = hr_value
                        confidence = 0.8
                    else:
                        realtime_metrics['heart_rate'] = None
                else:
                    realtime_metrics['heart_rate'] = None
                
                analysis_data['heart_rate'] = realtime_metrics['heart_rate']
                analysis_data['confidence'] = max(confidence, heart_rate_data.get('confidence', 0.0))
                realtime_metrics['confidence'] = max(realtime_metrics['confidence'], confidence)
                
            except Exception as e:
                logger.error(f"Real-time heart rate error: {e}")
                realtime_metrics['heart_rate'] = None

            # Real-time respiratory rate analysis
            try:
                respiratory_data = self.respiratory_detector.analyze_single_frame(frame, primary_landmarks)
                confidence = 0.0
                
                if isinstance(respiratory_data.get('respiratory_rate'), str):
                    # Parse string format like "16 breaths/min"
                    rr_str = respiratory_data['respiratory_rate'].lower()
                    if 'breaths/min' in rr_str or '/min' in rr_str:
                        try:
                            rr_value = float(rr_str.split()[0])
                            if 8 <= rr_value <= 40:  # Reasonable RR range
                                realtime_metrics['respiratory_rate'] = rr_value
                                confidence = 0.7
                        except (ValueError, IndexError):
                            realtime_metrics['respiratory_rate'] = None
                    else:
                        realtime_metrics['respiratory_rate'] = None
                elif isinstance(respiratory_data.get('respiratory_rate'), (int, float)):
                    rr_value = float(respiratory_data['respiratory_rate'])
                    if 8 <= rr_value <= 40:  # Reasonable RR range
                        realtime_metrics['respiratory_rate'] = rr_value
                        confidence = 0.7
                    else:
                        realtime_metrics['respiratory_rate'] = None
                else:
                    realtime_metrics['respiratory_rate'] = None
                
                analysis_data['respiratory_rate'] = realtime_metrics['respiratory_rate']
                realtime_metrics['confidence'] = max(realtime_metrics['confidence'], confidence)
                
            except Exception as e:
                logger.error(f"Real-time respiratory rate error: {e}")
                realtime_metrics['respiratory_rate'] = None

            # Real-time emotion detection
            try:
                emotion_data = self.emotion_detector.detect_emotion(face_roi)
                emotion = emotion_data.get('primary_emotion', 'Unknown')
                
                # Filter out invalid emotions
                valid_emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Fear', 'Disgust', 'Neutral']
                if emotion in valid_emotions:
                    realtime_metrics['emotion'] = emotion
                    analysis_data['emotion'] = emotion
                    confidence = emotion_data.get('confidence', 0.0)
                    realtime_metrics['confidence'] = max(realtime_metrics['confidence'], confidence)
                else:
                    realtime_metrics['emotion'] = 'Unknown'
                    analysis_data['emotion'] = 'Unknown'
                    
            except Exception as e:
                logger.error(f"Real-time emotion detection error: {e}")
                realtime_metrics['emotion'] = 'Unknown'

            # Real-time stress level analysis
            try:
                stress_data = self.stress_detector.analyze_stress_indicators(face_roi, primary_landmarks, {})
                
                if isinstance(stress_data.get('stress_level'), (int, float)):
                    stress_value = float(stress_data['stress_level'])
                    if 0 <= stress_value <= 10:  # Valid stress range
                        realtime_metrics['stress_level'] = stress_value
                        analysis_data['stress_level'] = stress_value
                        realtime_metrics['confidence'] = max(realtime_metrics['confidence'], 0.6)
                elif isinstance(stress_data.get('stress_level'), str) and '/' in stress_data.get('stress_level', ''):
                    try:
                        stress_value = float(stress_data['stress_level'].split('/')[0])
                        if 0 <= stress_value <= 10:
                            realtime_metrics['stress_level'] = stress_value
                            analysis_data['stress_level'] = stress_value
                            realtime_metrics['confidence'] = max(realtime_metrics['confidence'], 0.6)
                    except (ValueError, IndexError):
                        realtime_metrics['stress_level'] = None
                else:
                    realtime_metrics['stress_level'] = None
                    
            except Exception as e:
                logger.error(f"Real-time stress detection error: {e}")
                realtime_metrics['stress_level'] = None

            # Ensure minimum confidence based on successful detections
            successful_detections = sum([
                1 for metric in [realtime_metrics['heart_rate'], realtime_metrics['respiratory_rate'], 
                               realtime_metrics['emotion'], realtime_metrics['stress_level']] 
                if metric is not None
            ])
            
            if successful_detections > 0:
                base_confidence = min(0.5 + (successful_detections * 0.1), 0.9)
                realtime_metrics['confidence'] = max(realtime_metrics['confidence'], base_confidence)

            return {
                'realtime_metrics': realtime_metrics,
                'face_detection': face_detection,
                'analysis_data': analysis_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Video frame processing error: {e}")
            return {
                'realtime_metrics': realtime_metrics,
                'face_detection': face_detection,
                'analysis_data': analysis_data,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


@app.websocket("/ws/video_stream")
async def websocket_video_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time video stream processing"""
    await websocket.accept()
    logger.info("Video stream WebSocket connection established")
    
    # Create a new health monitor instance for this session
    video_monitor = VideoStreamHealthMonitor()

    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_json({
                    "error": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
                continue

            # Handle ping messages
            if payload.get('type') == 'ping':
                await websocket.send_json({"type": "pong"})
                continue

            # Process video frame
            if payload.get('type') == 'video_frame':
                base64_image = payload.get('image')
                
                if not base64_image:
                    await websocket.send_json({
                        "error": "No image data provided",
                        "timestamp": datetime.now().isoformat()
                    })
                    continue

                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(base64_image)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is None:
                        await websocket.send_json({
                            "error": "Failed to decode image",
                            "timestamp": datetime.now().isoformat()
                        })
                        continue

                    # Process the video frame
                    result = await video_monitor.process_video_frame(frame)
                    
                    # Send results back to client
                    await websocket.send_json(result)
                    
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    await websocket.send_json({
                        "error": f"Frame processing failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })

    except WebSocketDisconnect:
        logger.info("Video stream WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Video stream WebSocket error: {e}")
        try:
            await websocket.send_json({
                "error": f"Server error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
    finally:
        logger.info("Video stream WebSocket session ended")


# Keep the old endpoint for backward compatibility
@app.websocket("/ws/analyze_stream")
async def websocket_analyze_stream(websocket: WebSocket):
    """Legacy WebSocket endpoint - redirects to video_stream"""
    await websocket.accept()
    logger.info("Legacy WebSocket connection - redirecting to video_stream endpoint")
    await websocket.send_json({
        "message": "Please use /ws/video_stream endpoint for real-time analysis",
        "timestamp": datetime.now().isoformat()
    })
    await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)