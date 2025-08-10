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

# Import our health detection modules
from app.detectors.heart_rate_detector import HeartRateDetector
from app.detectors.respiratory_detector import RespiratoryRateDetector
from app.detectors.emotion_detector import EmotionDetector
from app.detectors.stress_detector import StressDetector
from app.detectors.fatigue_detector import FatigueDetector
# from app.detectors.neurological_detector import NeurologicalDetector
# from app.detectors.skin_analyzer import SkinAnalyzer
# from app.analyzers.health_assessor import HealthAssessor
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
        # self.neurological_detector = NeurologicalDetector()
        # self.skin_analyzer = SkinAnalyzer()
        # self.health_assessor = HealthAssessor() # This will be uncommented later

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
        
        # # Neurological Assessment (intentionally commented)
        # try:
        #     neuro_data = health_system.neurological_detector.assess_neurological_health(
        #         face_roi, primary_landmarks
        #     )
        #     health_metrics['facialAsymmetry'] = neuro_data['facial_asymmetry']
        #     health_metrics['tremor'] = neuro_data['tremor_detected']
        #     health_metrics['eyeMovement'] = neuro_data['eye_movement_analysis']
        # except Exception as e:
        #     logger.error(f"Neurological assessment error: {e}")
        #     health_metrics['facialAsymmetry'] = 'Analysis Error'
        #     health_metrics['tremor'] = 'N/A'
        #     health_metrics['eyeMovement'] = 'N/A'
        
        # # Skin Analysis (intentionally commented)
        # try:
        #     skin_data = health_system.skin_analyzer.analyze_skin_health(face_roi)
        #     health_metrics['skinAnalysis'] = skin_data['overall_health']
        #     health_metrics['skinColor'] = skin_data['color_analysis']
        #     health_metrics['hydrationStatus'] = skin_data['hydration_estimate']
        # except Exception as e:
        #     logger.error(f"Skin analysis error: {e}")
        #     health_metrics['skinAnalysis'] = 'Analysis Error'
        #     health_metrics['skinColor'] = 'N/A'
        #     health_metrics['hydrationStatus'] = 'N/A'
        
        # # Generate overall health assessment (intentionally commented)
        # try:
        #     overall_assessment = health_system.health_assessor.calculate_health_score(health_metrics)
        #     health_metrics['overallHealthScore'] = overall_assessment['score']
        #     health_metrics['healthStatus'] = overall_assessment['status']
        #     alerts.extend(overall_assessment['alerts'])
        # except Exception as e:
        #     logger.error(f"Health assessment error: {e}")
        #     health_metrics['overallHealthScore'] = 0.0
        #     health_metrics['healthStatus'] = 'Assessment Error'
        
        # Generate recommendations (Placeholder if HealthAssessor is commented out)
        recommendations = ["Further analysis needed after all modules are enabled."] if not hasattr(health_system, 'health_assessor') else health_system.health_assessor.generate_recommendations(health_metrics)
        
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

# --- WebSocket Endpoint for Real-time Analysis ---

# This class will hold stateful detector instances for a single WebSocket session.
# This is a conceptual representation. The actual HeartRateDetector/RespiratoryRateDetector
# classes will need internal buffers and logic to accumulate frames over time.
class StreamHealthMonitorSystem:
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
        # self.neurological_detector = NeurologicalDetector() # Uncomment when implemented
        # self.skin_analyzer = SkinAnalyzer() # Uncomment when implemented
        # self.health_assessor = HealthAssessor() # Uncomment when implemented
        
        logger.info("Stream Health Monitor System initialized for a new WebSocket session")

    async def process_frame(self, frame):
        metrics = {}
        alerts = []
        face_data = {}
        
        faces, landmarks = self.face_detector.detect_face_and_landmarks(frame)
        
        if len(faces) == 0:
            return {"error": "No face detected", "metrics": {}, "alerts": [], "analysis_quality": "No Face"}

        primary_face = faces[0]
        primary_landmarks = landmarks[0] if landmarks else None
        processed_frame = self.image_processor.preprocess_image(frame)
        face_roi = self.image_processor.extract_face_roi(processed_frame, primary_face)

        x, y, w, h = primary_face
        face_data = {
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "landmarks": primary_landmarks.tolist() if primary_landmarks is not None else []
        }

        # Cardiovascular Analysis
        try:
            heart_rate_data = self.heart_rate_detector.analyze_single_frame(face_roi, primary_landmarks)
            # The heart_rate_detector should internally buffer and return a result only when stable
            metrics['heartRate'] = heart_rate_data.get('heart_rate', 'Analyzing...')
            metrics['heartRateVariability'] = heart_rate_data.get('hrv_score', 'N/A')
        except Exception as e:
            logger.error(f"Stream Heart rate detection error: {e}")
            metrics['heartRate'] = 'Analysis Error'
            metrics['heartRateVariability'] = 'N/A'
        
        # Respiratory Analysis
        try:
            respiratory_data = self.respiratory_detector.analyze_single_frame(frame, primary_landmarks)
            metrics['respiratoryRate'] = respiratory_data.get('respiratory_rate', 'Analyzing...')
            metrics['breathingPattern'] = respiratory_data.get('pattern', 'N/A')
        except Exception as e:
            logger.error(f"Stream Respiratory detection error: {e}")
            metrics['respiratoryRate'] = 'Analysis Error'
            metrics['breathingPattern'] = 'N/A'
        
        # Emotion Analysis
        try:
            emotion_data = self.emotion_detector.detect_emotion(face_roi)
            metrics['emotion'] = emotion_data.get('primary_emotion', 'Detection Error')
            metrics['emotionConfidence'] = emotion_data.get('confidence', 0.0)
        except Exception as e:
            logger.error(f"Stream Emotion detection error: {e}")
            metrics['emotion'] = 'Detection Error'
            metrics['emotionConfidence'] = 0.0
        
        # Stress Level Analysis
        try:
            stress_data = self.stress_detector.analyze_stress_indicators(face_roi, primary_landmarks, metrics)
            metrics['stressLevel'] = stress_data.get('stress_level', 'Analysis Error')
            metrics['stressFactors'] = stress_data.get('factors', [])
        except Exception as e:
            logger.error(f"Stream Stress detection error: {e}")
            metrics['stressLevel'] = 'Analysis Error'
            metrics['stressFactors'] = []
        
        # Fatigue Analysis
        try:
            fatigue_data = self.fatigue_detector.assess_fatigue(face_roi, primary_landmarks)
            metrics['fatigue'] = fatigue_data.get('fatigue_level', 'Analysis Error')
            metrics['alertnessScore'] = fatigue_data.get('alertness_score', 0.0)
        except Exception as e:
            logger.error(f"Stream Fatigue detection error: {e}")
            metrics['fatigue'] = 'Analysis Error'
            metrics['alertnessScore'] = 0.0
        
        # # Neurological Assessment (intentionally commented)
        # try:
        #     neuro_data = self.neurological_detector.assess_neurological_health(face_roi, primary_landmarks)
        #     metrics['facialAsymmetry'] = neuro_data.get('facial_asymmetry', 'Analysis Error')
        #     metrics['tremor'] = neuro_data.get('tremor_detected', 'N/A')
        #     metrics['eyeMovement'] = neuro_data.get('eye_movement_analysis', 'N/A')
        # except Exception as e:
        #     logger.error(f"Stream Neurological assessment error: {e}")
        #     metrics['facialAsymmetry'] = 'Analysis Error'
        #     metrics['tremor'] = 'N/A'
        #     metrics['eyeMovement'] = 'N/A'
        
        # # Skin Analysis (intentionally commented)
        # try:
        #     skin_data = self.skin_analyzer.analyze_skin_health(face_roi)
        #     metrics['skinAnalysis'] = skin_data.get('overall_health', 'Analysis Error')
        #     metrics['skinColor'] = skin_data.get('color_analysis', 'N/A')
        #     metrics['hydrationStatus'] = skin_data.get('hydration_estimate', 'N/A')
        # except Exception as e:
        #     logger.error(f"Stream Skin analysis error: {e}")
        #     metrics['skinAnalysis'] = 'Analysis Error'
        #     metrics['skinColor'] = 'N/A'
        #     metrics['hydrationStatus'] = 'N/A'
        
        # # Overall health assessment (intentionally commented)
        # overall_assessment = {}
        # if hasattr(self, 'health_assessor'): # Check if assessor is enabled
        #     try:
        #         overall_assessment = self.health_assessor.calculate_health_score(metrics)
        #         metrics['overallHealthScore'] = overall_assessment.get('score', 0.0)
        #         metrics['healthStatus'] = overall_assessment.get('status', 'Assessment Error')
        #         alerts.extend(overall_assessment.get('alerts', []))
        #     except Exception as e:
        #         logger.error(f"Stream Health assessment error: {e}")
        #         metrics['overallHealthScore'] = 0.0
        #         metrics['healthStatus'] = 'Assessment Error'

        recommendations = ["Further analysis needed after all modules are enabled."]
        # if hasattr(self, 'health_assessor'):
        #     recommendations = self.health_assessor.generate_recommendations(metrics)


        return {
            **metrics,
            'alerts': alerts,
            'recommendations': recommendations,
            'analysis_timestamp': self.image_processor.get_current_timestamp(),
            'face_detected': True,
            'analysis_quality': 'Good' if len(alerts) == 0 else 'Warning',
            'face_data': face_data
        }


@app.websocket("/ws/analyze_stream")
async def websocket_analyze_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    session_health_system = StreamHealthMonitorSystem()

    try:
        while True:
            data = await websocket.receive_text()
            logger.info("Received WebSocket message")
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_json({"error": "Invalid JSON format received"})
                continue

            if payload.get('type') == 'ping':
                await websocket.send_json({"type": "pong"})
                logger.info("Received ping, sent pong")
                continue

            base64_image = payload.get('image')
            if not base64_image:
                logger.warning("Received payload without image data.")
                await websocket.send_json({"error": "No image data in payload", "timestamp": datetime.now().isoformat()})
                continue

            try:
                logger.info("Decoding base64 image")
                image_bytes = base64.b64decode(base64_image)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    logger.warning("Failed to decode image from received data.")
                    await websocket.send_json({"error": "Invalid image data received", "timestamp": datetime.now().isoformat()})
                    continue

                logger.info("Processing frame")
                current_metrics = await session_health_system.process_frame(frame)
                logger.info(f"Sending metrics: {current_metrics}")
                await websocket.send_json(current_metrics)
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                await websocket.send_json({"error": f"Frame processing failed: {str(e)}", "timestamp": datetime.now().isoformat()})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket processing error: {e}")
        await websocket.send_json({"error": f"Internal server error: {str(e)}", "timestamp": datetime.now().isoformat()})
    finally:
        logger.info("WebSocket session ended.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

