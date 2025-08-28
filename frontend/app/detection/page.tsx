import React, { useState, useRef, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Dimensions, ActivityIndicator, Platform } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import Animated, { FadeIn, FadeInDown, useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming } from 'react-native-reanimated';

type FaceBox = { x: number; y: number; width: number; height: number };
type RealTimePoint = {
  timestamp: number;
  heart_rate?: number;
  respiratory_rate?: number;
  stress_level?: number;
  emotion?: string;
  confidence: number;
  // Additional analysis fields from backend
  fatigue_level?: string | number;
  alertness_score?: number;
  facial_asymmetry?: string;
  tremor_detected?: boolean | string;
  eye_movement_analysis?: string;
  skin_overall_health?: string;
  skin_color_analysis?: string;
  hydration_estimate?: string | number;
  hrv_score?: number;
  blood_pressure?: string;
  cognitive_load?: number;
  pain_level?: number;
};
type CurrentMetrics = {
  heartRate: number | null;
  respiratoryRate: number | null;
  stressLevel: number | null;
  emotion: string | null;
  confidence: number;
  faceDetected: boolean;
};
type FinalResultsType = {
  heartRate: string;
  respiratoryRate: string;
  stressLevel: string;
  emotion: string;
  fatigue: string;
  facialAsymmetry: string;
  tremor: string;
  eyeMovement: string;
  skinAnalysis: string;
  skinColor: string;
  hydrationStatus: string;
  hrv: string;
  bloodPressure: string;
  cognitiveLoad: string;
  painLevel: string;
  overallHealthScore: string;
  healthStatus: string;
  recommendations: string[];
  sessionDuration: string;
  dataPoints: number;
};

const { width, height } = Dimensions.get('window');
// Prefer env; fallback to backend host observed in ipconfig logs
const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'http://192.168.220.4:8000';
const WEBSOCKET_URL = BACKEND_URL.replace('http', 'ws') + '/ws/video_stream';
const VIDEO_STREAM_FPS = 6; // Lower FPS to reduce capture failures
const DETECTION_CONFIDENCE_THRESHOLD = 0.5;

export default function RealTimeDetectionPage() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [detectionActive, setDetectionActive] = useState<boolean>(false);
  const [sessionData, setSessionData] = useState<any>(null);
  
  // Real-time detection states
  const [currentMetrics, setCurrentMetrics] = useState<CurrentMetrics>({
    heartRate: null,
    respiratoryRate: null,
    stressLevel: null,
    emotion: null,
    confidence: 0,
    faceDetected: false
  });

  // Face detection and tracking
  const [faceBox, setFaceBox] = useState<FaceBox | null>(null);
  const [landmarks, setLandmarks] = useState<number[][]>([]);
  const [detectionQuality, setDetectionQuality] = useState<string>('Searching...');
  
  const [finalResults, setFinalResults] = useState<FinalResultsType>({
    heartRate: 'N/A', 
    respiratoryRate: 'N/A', 
    stressLevel: 'N/A', 
    emotion: 'N/A',
    fatigue: 'N/A', 
    facialAsymmetry: 'N/A', 
    tremor: 'N/A', 
    eyeMovement: 'N/A',
    skinAnalysis: 'N/A', 
    skinColor: 'N/A', 
    hydrationStatus: 'N/A',
    hrv: 'N/A',
    bloodPressure: 'N/A',
    cognitiveLoad: 'N/A',
    painLevel: 'N/A',
    overallHealthScore: 'N/A', 
    healthStatus: 'N/A', 
    recommendations: [],
    sessionDuration: '0:00',
    dataPoints: 0
  });

  const cameraRef = useRef<CameraView | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const router = useRouter();
  const streamInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionStartTime = useRef<number | null>(null);
  const timerInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const analysisData = useRef<RealTimePoint[]>([]);
  const reconnectAttempts = useRef<number>(0);
  const cameraIsReady = useRef<boolean>(false);
  const isCapturingFrame = useRef<boolean>(false);

  // Animation values
  const scanningOpacity = useSharedValue(0);
  const pulseScale = useSharedValue(1);

  // Animated styles
  const scanningStyle = useAnimatedStyle(() => ({
    opacity: scanningOpacity.value,
  }));

  const pulseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  useEffect(() => {
    if (detectionActive) {
      scanningOpacity.value = withRepeat(
        withSequence(
          withTiming(0.3, { duration: 1000 }),
          withTiming(1, { duration: 1000 })
        ),
        -1,
        true
      );
      
      pulseScale.value = withRepeat(
        withSequence(
          withTiming(1.02, { duration: 800 }),
          withTiming(1, { duration: 800 })
        ),
        -1,
        true
      );
    } else {
      scanningOpacity.value = withTiming(0, { duration: 300 });
      pulseScale.value = withTiming(1, { duration: 300 });
    }
  }, [detectionActive]);

  // WebSocket connection for real-time video streaming
  useEffect(() => {
    if (!isStreaming) return;

    console.log('Establishing WebSocket connection for video streaming');
    ws.current = new WebSocket(WEBSOCKET_URL);

    ws.current.onopen = () => {
      console.log('Video stream WebSocket connected');
      setDetectionActive(true);
      sessionStartTime.current = Date.now();
      setElapsedSeconds(0);
      if (timerInterval.current) {
        clearInterval(timerInterval.current);
      }
      timerInterval.current = setInterval(() => {
        if (sessionStartTime.current) {
          const secs = Math.floor((Date.now() - sessionStartTime.current) / 1000);
          setElapsedSeconds(secs);
        }
      }, 1000);
      analysisData.current = [];
      reconnectAttempts.current = 0;
      startVideoStream();
    };

    ws.current.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Real-time detection data:', data);

        // Update real-time metrics
        if (data.realtime_metrics) {
          setCurrentMetrics(prev => ({
            ...prev,
            heartRate: typeof data.realtime_metrics.heart_rate === 'number' ? data.realtime_metrics.heart_rate : null,
            respiratoryRate: typeof data.realtime_metrics.respiratory_rate === 'number' ? data.realtime_metrics.respiratory_rate : null,
            stressLevel: typeof data.realtime_metrics.stress_level === 'number' ? data.realtime_metrics.stress_level : null,
            emotion: data.realtime_metrics.emotion,
            confidence: data.realtime_metrics.confidence || 0,
            faceDetected: data.realtime_metrics.face_detected || false
          }));
        }

        // Update face detection visualization
        if (data.face_detection) {
          setFaceBox(data.face_detection.bounding_box);
          setLandmarks(Array.isArray(data.face_detection.landmarks) ? data.face_detection.landmarks : []);
          setDetectionQuality(data.face_detection.quality || 'Good');
        }

        // Store analysis data for final results
        if (data.analysis_data) {
          analysisData.current.push({
            timestamp: Date.now(),
            ...data.analysis_data
          });
        }

      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };

    ws.current.onerror = (error: Event) => {
      console.error('WebSocket error:', error);
      setDetectionActive(false);
    };

    ws.current.onclose = (event: CloseEvent) => {
      console.log('Video stream WebSocket disconnected', event.code, event.reason);
      setDetectionActive(false);
      stopVideoStream();

      // Auto-reconnect logic
      if (isStreaming && reconnectAttempts.current < 3) {
        reconnectAttempts.current++;
        console.log(`Attempting to reconnect... (${reconnectAttempts.current}/3)`);
        setTimeout(() => {
          if (isStreaming) {
            // Trigger re-connection by updating the effect dependency
            setIsStreaming(false);
            setTimeout(() => setIsStreaming(true), 500);
          }
        }, 2000);
      }
    };

    return () => {
      if (ws.current) {
        try {
          ws.current.close();
        } catch {}
        ws.current = null;
      }
      stopVideoStream();
      if (timerInterval.current) {
        clearInterval(timerInterval.current);
        timerInterval.current = null;
      }
    };
  }, [isStreaming]);

  const startVideoStream = useCallback(() => {
    if (!cameraRef.current) return;

    streamInterval.current = setInterval(async () => {
      if (!cameraIsReady.current) return;
      if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
        console.warn('WebSocket not ready for video streaming');
        return;
      }
      if (isCapturingFrame.current) return;
      isCapturingFrame.current = true;

      try {
        // Capture frame for video stream: lighter to emulate video
        const camera = cameraRef.current;
        if (!camera) return;
        const photo = await camera.takePictureAsync({
          base64: true,
          quality: 0.6,
          skipProcessing: false,
        });

        if (photo && photo.base64) {
          // Send frame data with metadata
          const frameData = {
            type: 'video_frame',
            image: photo.base64,
            timestamp: Date.now(),
            frame_metadata: {
              width: photo.width,
              height: photo.height,
              fps: VIDEO_STREAM_FPS
            }
          };

          try {
            ws.current.send(JSON.stringify(frameData));
          } catch {}
        }
      } catch (error) {
        console.error('Error capturing video frame:', error);
      } finally {
        isCapturingFrame.current = false;
      }
    }, 1000 / VIDEO_STREAM_FPS);
  }, []);

  const stopVideoStream = useCallback(() => {
    if (streamInterval.current) {
      clearInterval(streamInterval.current);
      streamInterval.current = null;
    }
  }, []);

  const handleStartDetection = async () => {
    if (!permission || !permission.granted) {
      const { granted } = await requestPermission();
      if (!granted) {
        console.warn('Camera permission denied');
        return;
      }
    }

    setIsStreaming(true);
    setShowResults(false);
    setCurrentMetrics({
      heartRate: null,
      respiratoryRate: null,
      stressLevel: null,
      emotion: null,
      confidence: 0,
      faceDetected: false
    });
  };

  const handleStopDetection = useCallback(() => {
    setIsStreaming(false);
    setDetectionActive(false);
    if (timerInterval.current) {
      clearInterval(timerInterval.current);
      timerInterval.current = null;
    }
    
    // Calculate session duration
    const sessionDuration = sessionStartTime.current 
      ? Math.floor((Date.now() - sessionStartTime.current) / 1000)
      : 0;
    
    const minutes = Math.floor(sessionDuration / 60);
    const seconds = sessionDuration % 60;
    const durationString = `${minutes}:${seconds.toString().padStart(2, '0')}`;

    // Process accumulated analysis data for final results
    const processedResults = processAnalysisData(analysisData.current);
    
    setFinalResults({
      ...processedResults,
      sessionDuration: durationString,
      dataPoints: analysisData.current.length
    });

    setShowResults(true);
    
    // Close WebSocket connection
    if (ws.current) {
      try {
        ws.current.close(1000, 'Detection stopped by user');
      } catch {}
      ws.current = null;
    }
  }, []);

  const processAnalysisData = (dataPoints: RealTimePoint[]) => {
    if (dataPoints.length === 0) {
      return {
        heartRate: 'N/A',
        respiratoryRate: 'N/A',
        stressLevel: 'N/A',
        emotion: 'N/A',
        fatigue: 'N/A',
        facialAsymmetry: 'N/A',
        tremor: 'N/A',
        eyeMovement: 'N/A',
        skinAnalysis: 'N/A',
        skinColor: 'N/A',
        hydrationStatus: 'N/A',
        hrv: 'N/A',
        bloodPressure: 'N/A',
        cognitiveLoad: 'N/A',
        painLevel: 'N/A',
        overallHealthScore: 'N/A',
        healthStatus: 'Low Confidence',
        recommendations: ['Insufficient data - longer session recommended']
      };
    }

    // Calculate averages and trends from collected data
    const validData = dataPoints.filter((d: RealTimePoint) => d.confidence > DETECTION_CONFIDENCE_THRESHOLD);
    
    if (validData.length === 0) {
      return {
        heartRate: 'N/A',
        respiratoryRate: 'N/A',
        stressLevel: 'N/A',
        emotion: 'N/A',
        fatigue: 'N/A',
        facialAsymmetry: 'N/A',
        tremor: 'N/A',
        eyeMovement: 'N/A',
        skinAnalysis: 'N/A',
        skinColor: 'N/A',
        hydrationStatus: 'N/A',
        hrv: 'N/A',
        bloodPressure: 'N/A',
        cognitiveLoad: 'N/A',
        painLevel: 'N/A',
        overallHealthScore: 'N/A',
        healthStatus: 'Poor Detection Quality',
        recommendations: ['Improve lighting conditions', 'Ensure face is clearly visible']
      };
    }

    // Calculate averages
    const avgHeartRate = Math.round(validData.reduce((sum: number, d: RealTimePoint) => sum + (d.heart_rate || 0), 0) / validData.length);
    const avgRespRate = Math.round(validData.reduce((sum: number, d: RealTimePoint) => sum + (d.respiratory_rate || 0), 0) / validData.length);
    const avgStressNum = validData.reduce((sum: number, d: RealTimePoint) => sum + (d.stress_level || 0), 0) / validData.length;
    const avgStress = avgStressNum.toFixed(1);
    
    // Most common emotion
    const emotions = validData.map((d: RealTimePoint) => d.emotion).filter((e): e is string => Boolean(e));
    const emotionCounts = emotions.reduce((acc: Record<string, number>, emotion: string) => {
      acc[emotion] = (acc[emotion] || 0) + 1;
      return acc;
    }, {});
    const dominantEmotion = Object.keys(emotionCounts).reduce((a, b) => emotionCounts[a] > emotionCounts[b] ? a : b, 'Neutral');

    // Derive additional metrics (fatigue, neurological, skin)
    const pickMostFrequent = (items: (string | undefined | null)[]) => {
      const filtered = items.filter((v): v is string => typeof v === 'string' && v.length > 0);
      if (filtered.length === 0) return 'N/A';
      const counts: Record<string, number> = {};
      filtered.forEach((v) => { counts[v] = (counts[v] || 0) + 1; });
      return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    };

    // Fatigue level: prefer string categorical mode, else bucket numeric average
    let fatigue: string = pickMostFrequent(validData.map(d => typeof d.fatigue_level === 'string' ? d.fatigue_level : undefined));
    if (fatigue === 'N/A') {
      const numericFatigue = validData.map(d => typeof d.fatigue_level === 'number' ? d.fatigue_level : null).filter((v): v is number => v !== null);
      if (numericFatigue.length > 0) {
        const avgFatigue = numericFatigue.reduce((a, b) => a + b, 0) / numericFatigue.length;
        fatigue = avgFatigue >= 7 ? 'High' : avgFatigue >= 4 ? 'Moderate' : 'Low';
      } else {
        fatigue = 'Analyzing...';
      }
    }

    const facialAsymmetry = pickMostFrequent(validData.map(d => d.facial_asymmetry));
    const tremorDetectedAny = validData.some(d => d.tremor_detected === true || (typeof d.tremor_detected === 'string' && d.tremor_detected.toLowerCase().includes('true')));
    const tremor = tremorDetectedAny ? 'Detected' : 'Not Detected';
    const eyeMovement = pickMostFrequent(validData.map(d => d.eye_movement_analysis));
    const skinAnalysis = pickMostFrequent(validData.map(d => d.skin_overall_health));
    const skinColor = pickMostFrequent(validData.map(d => d.skin_color_analysis));
    let hydrationStatus = pickMostFrequent(validData.map(d => typeof d.hydration_estimate === 'string' ? d.hydration_estimate : undefined));
    if (hydrationStatus === 'N/A') {
      const hydrationNums = validData.map(d => typeof d.hydration_estimate === 'number' ? d.hydration_estimate : null).filter((v): v is number => v !== null);
      if (hydrationNums.length > 0) {
        const avgHydration = hydrationNums.reduce((a, b) => a + b, 0) / hydrationNums.length;
        hydrationStatus = avgHydration >= 0.66 ? 'Good' : avgHydration >= 0.33 ? 'Moderate' : 'Low';
      }
    }

    // HRV average (0-1 scaled placeholder)
    const hrvValues = validData.map(d => typeof d.hrv_score === 'number' ? d.hrv_score : null).filter((v): v is number => v !== null);
    const avgHrv = hrvValues.length > 0 ? (hrvValues.reduce((a, b) => a + b, 0) / hrvValues.length) : null;

    // Blood pressure: pick the most frequent categorical value
    const bloodPressure = pickMostFrequent(validData.map(d => d.blood_pressure));

    // Cognitive load average (0-10)
    const cogValues = validData.map(d => typeof d.cognitive_load === 'number' ? d.cognitive_load : null).filter((v): v is number => v !== null);
    const avgCog = cogValues.length > 0 ? (cogValues.reduce((a, b) => a + b, 0) / cogValues.length) : null;

    // Pain level average (0-10)
    const painValues = validData.map(d => typeof d.pain_level === 'number' ? d.pain_level : null).filter((v): v is number => v !== null);
    const avgPain = painValues.length > 0 ? (painValues.reduce((a, b) => a + b, 0) / painValues.length) : null;

    // Generate health recommendations
    const recommendations = generateRecommendations({
      heartRate: avgHeartRate,
      stressLevel: avgStressNum,
      emotion: dominantEmotion,
      dataQuality: validData.length / dataPoints.length
    });

    // Calculate overall health score
    const healthScore = calculateHealthScore({
      heartRate: avgHeartRate,
      stressLevel: parseFloat(avgStress),
      dataQuality: validData.length / dataPoints.length
    });

    return {
      heartRate: avgHeartRate > 0 ? `${avgHeartRate} BPM` : 'N/A',
      respiratoryRate: avgRespRate > 0 ? `${avgRespRate} /min` : 'N/A',
      stressLevel: parseFloat(avgStress) > 0 ? `${avgStress}/10` : 'N/A',
      emotion: dominantEmotion,
      fatigue,
      facialAsymmetry,
      tremor,
      eyeMovement,
      skinAnalysis,
      skinColor,
      hydrationStatus,
      hrv: (avgHrv !== null && !isNaN(avgHrv)) ? `${Math.round(avgHrv * 100)}/100` : 'N/A',
      bloodPressure: bloodPressure || 'N/A',
      cognitiveLoad: (avgCog !== null && !isNaN(avgCog)) ? `${avgCog.toFixed(1)}/10` : 'N/A',
      painLevel: (avgPain !== null && !isNaN(avgPain)) ? `${avgPain.toFixed(1)}/10` : 'N/A',
      overallHealthScore: `${healthScore}/100`,
      healthStatus: getHealthStatus(healthScore),
      recommendations: recommendations
    };
  };

  const generateRecommendations = (metrics: { heartRate: number; stressLevel: number; emotion: string; dataQuality: number; }) => {
    const recommendations = [];
    
    if (metrics.heartRate > 100) {
      recommendations.push('Consider relaxation techniques - elevated heart rate detected');
    } else if (metrics.heartRate < 60) {
      recommendations.push('Monitor physical activity levels');
    }
    
    if (metrics.stressLevel > 7) {
      recommendations.push('High stress levels - practice deep breathing exercises');
    }
    
    if (metrics.emotion === 'Sad' || metrics.emotion === 'Angry') {
      recommendations.push('Consider mindfulness or stress management activities');
    }
    
    if (metrics.dataQuality < 0.5) {
      recommendations.push('Improve lighting and positioning for better analysis');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('Maintain current healthy habits');
    }
    
    return recommendations;
  };

  const calculateHealthScore = (metrics: { heartRate: number; stressLevel: number; dataQuality: number; }) => {
    let score = 100;
    
    // Heart rate impact
    if (metrics.heartRate > 100 || metrics.heartRate < 60) {
      score -= 20;
    }
    
    // Stress level impact
    score -= metrics.stressLevel * 8; // Max -80 for stress level 10
    
    // Data quality impact
    score -= (1 - metrics.dataQuality) * 30;
    
    return Math.max(0, Math.round(score));
  };

  const getHealthStatus = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Needs Attention';
  };

  if (!permission) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#64FFDA" />
        <Text style={styles.loadingText}>Loading camera permissions...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="camera-outline" size={50} color="#FF6B6B" />
        <Text style={styles.errorText}>Camera Permission Required</Text>
        <Text style={styles.errorDescription}>
          This app needs camera access to perform real-time health detection.
        </Text>
        <TouchableOpacity style={styles.requestPermissionButton} onPress={requestPermission}>
          <Text style={styles.requestPermissionText}>Grant Permission</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.goBackButton} onPress={() => router.back()}>
          <Text style={styles.goBackText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (showResults) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <LinearGradient
          colors={['rgba(10, 25, 47, 0.9)', 'rgba(17, 34, 64, 0.95)']}
          style={styles.gradientBackground}
        >
          <View style={styles.header}>
            <TouchableOpacity onPress={() => setShowResults(false)} style={styles.backButton}>
              <Ionicons name="arrow-back" size={24} color="#E6F1FF" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>Detection Results</Text>
            <TouchableOpacity onPress={() => router.back()} style={styles.closeButton}>
              <Ionicons name="close" size={24} color="#E6F1FF" />
            </TouchableOpacity>
          </View>

          <Animated.ScrollView 
            style={styles.resultsScrollView}
            contentContainerStyle={styles.resultsContent}
            entering={FadeInDown.duration(600)}
          >
            <View style={styles.sessionSummary}>
              <Text style={styles.sectionTitle}>Session Summary</Text>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Duration:</Text>
                <Text style={styles.summaryValue}>{finalResults.sessionDuration}</Text>
              </View>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Data Points:</Text>
                <Text style={styles.summaryValue}>{finalResults.dataPoints}</Text>
              </View>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Overall Score:</Text>
                <Text style={[styles.summaryValue, styles.scoreValue]}>{finalResults.overallHealthScore}</Text>
              </View>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Health Status:</Text>
                <Text style={[styles.summaryValue, styles.statusValue]}>{finalResults.healthStatus}</Text>
              </View>
            </View>

            <Text style={styles.sectionTitle}>Vital Signs</Text>
            <View style={styles.metricsGrid}>
              <View style={styles.metricCard}>
                <Ionicons name="heart" size={24} color="#FF6B6B" />
                <Text style={styles.metricValue}>{finalResults.heartRate}</Text>
                <Text style={styles.metricLabel}>Heart Rate</Text>
              </View>
              
              <View style={styles.metricCard}>
                <Ionicons name="fitness" size={24} color="#4ECDC4" />
                <Text style={styles.metricValue}>{finalResults.respiratoryRate}</Text>
                <Text style={styles.metricLabel}>Respiratory Rate</Text>
              </View>
              
              <View style={styles.metricCard}>
                <Ionicons name="analytics" size={24} color="#FFD93D" />
                <Text style={styles.metricValue}>{finalResults.stressLevel}</Text>
                <Text style={styles.metricLabel}>Stress Level</Text>
              </View>
              
              <View style={styles.metricCard}>
                <Ionicons name="happy" size={24} color="#6BCF7F" />
                <Text style={styles.metricValue}>{finalResults.emotion}</Text>
                <Text style={styles.metricLabel}>Dominant Emotion</Text>
              </View>
            </View>

            {/* Additional Analyses */}
            {(finalResults.fatigue !== 'N/A' ||
              finalResults.facialAsymmetry !== 'N/A' ||
              finalResults.tremor !== 'N/A' ||
              finalResults.eyeMovement !== 'N/A' ||
              finalResults.skinAnalysis !== 'N/A' ||
              finalResults.skinColor !== 'N/A' ||
              finalResults.hydrationStatus !== 'N/A') && (
              <>
                <Text style={styles.sectionTitle}>Additional Analyses</Text>
                <View style={styles.metricsGrid}>
                  {finalResults.hrv !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="pulse" size={24} color="#FFD93D" />
                      <Text style={styles.metricValue}>{finalResults.hrv}</Text>
                      <Text style={styles.metricLabel}>HRV</Text>
                    </View>
                  )}
                  {finalResults.bloodPressure !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="fitness" size={24} color="#FFD93D" />
                      <Text style={styles.metricValue}>{finalResults.bloodPressure}</Text>
                      <Text style={styles.metricLabel}>Blood Pressure</Text>
                    </View>
                  )}
                  {finalResults.cognitiveLoad !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="speedometer" size={24} color="#FFD93D" />
                      <Text style={styles.metricValue}>{finalResults.cognitiveLoad}</Text>
                      <Text style={styles.metricLabel}>Cognitive Load</Text>
                    </View>
                  )}
                  {finalResults.painLevel !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="medkit" size={24} color="#FFD93D" />
                      <Text style={styles.metricValue}>{finalResults.painLevel}</Text>
                      <Text style={styles.metricLabel}>Pain Level</Text>
                    </View>
                  )}
                  {finalResults.fatigue !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="moon" size={24} color="#A3BFFA" />
                      <Text style={styles.metricValue}>{finalResults.fatigue}</Text>
                      <Text style={styles.metricLabel}>Fatigue</Text>
                    </View>
                  )}
                  {finalResults.facialAsymmetry !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="person" size={24} color="#A3BFFA" />
                      <Text style={styles.metricValue}>{finalResults.facialAsymmetry}</Text>
                      <Text style={styles.metricLabel}>Facial Asymmetry</Text>
                    </View>
                  )}
                  {finalResults.tremor !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="hand-left" size={24} color="#A3BFFA" />
                      <Text style={styles.metricValue}>{finalResults.tremor}</Text>
                      <Text style={styles.metricLabel}>Tremor</Text>
                    </View>
                  )}
                  {finalResults.eyeMovement !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="eye" size={24} color="#A3BFFA" />
                      <Text style={styles.metricValue}>{finalResults.eyeMovement}</Text>
                      <Text style={styles.metricLabel}>Eye Movement</Text>
                    </View>
                  )}
                  {finalResults.skinAnalysis !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="color-palette" size={24} color="#A3BFFA" />
                      <Text style={styles.metricValue}>{finalResults.skinAnalysis}</Text>
                      <Text style={styles.metricLabel}>Skin Analysis</Text>
                    </View>
                  )}
                  {finalResults.skinColor !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="brush" size={24} color="#A3BFFA" />
                      <Text style={styles.metricValue}>{finalResults.skinColor}</Text>
                      <Text style={styles.metricLabel}>Skin Color</Text>
                    </View>
                  )}
                  {finalResults.hydrationStatus !== 'N/A' && (
                    <View style={styles.metricCard}>
                      <Ionicons name="water" size={24} color="#A3BFFA" />
                      <Text style={styles.metricValue}>{finalResults.hydrationStatus}</Text>
                      <Text style={styles.metricLabel}>Hydration</Text>
                    </View>
                  )}
                </View>
              </>
            )}

            {finalResults.recommendations.length > 0 && (
              <>
                <Text style={styles.sectionTitle}>Recommendations</Text>
                <View style={styles.recommendationsContainer}>
                  {finalResults.recommendations.map((rec, index) => (
                    <Animated.View
                      key={`rec-${index}`}
                      style={styles.recommendationCard}
                      entering={FadeIn.delay(index * 100)}
                    >
                      <Ionicons name="bulb-outline" size={20} color="#64FFDA" />
                      <Text style={styles.recommendationText}>{rec}</Text>
                    </Animated.View>
                  ))}
                </View>
              </>
            )}

            <View style={styles.actionButtons}>
              <TouchableOpacity
                style={styles.newScanButton}
                onPress={() => {
                  setShowResults(false);
                  handleStartDetection();
                }}
              >
                <Text style={styles.newScanButtonText}>New Scan</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.homeButton}
                onPress={() => router.back()}
              >
                <Text style={styles.homeButtonText}>Back to Home</Text>
              </TouchableOpacity>
            </View>
          </Animated.ScrollView>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <LinearGradient
        colors={['rgba(10, 25, 47, 0.9)', 'rgba(17, 34, 64, 0.95)']}
        style={styles.gradientBackground}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="#E6F1FF" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Real-Time Health Detection</Text>
          <View style={styles.headerStatus}>
            <View style={[styles.statusDot, { backgroundColor: detectionActive ? '#00FF00' : '#FF6B6B' }]} />
            <Text style={styles.statusText}>{detectionActive ? 'LIVE' : 'IDLE'}</Text>
          </View>
        </View>

        <View style={styles.fullScreenCameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.fullScreenCamera}
            facing="front"
            mirror={true}
            onCameraReady={() => {
              console.log('Camera ready for video streaming');
              cameraIsReady.current = true;
            }}
            onMountError={(error) => {
              console.error('Camera Mount Error:', error);
            }}
          />
          
          {/* Real-time detection overlay */}
          <Animated.View style={[styles.detectionOverlay, scanningStyle]}>
            <View style={styles.scanGrid}>
              {[...Array(4)].map((_, i) => (
                [...Array(3)].map((_, j) => (
                  <Animated.View 
                    key={`${i}-${j}`} 
                    style={[styles.scanLine, pulseStyle]} 
                  />
                ))
              ))}
            </View>
          </Animated.View>

          {/* Face detection box */}
          {faceBox && detectionActive && (
            <Animated.View
              style={[styles.faceDetectionBox, {
                left: (faceBox.x / 100) * width,
                top: (faceBox.y / 100) * height,
                width: (faceBox.width / 100) * width,
                height: (faceBox.height / 100) * height,
              }]}
              entering={FadeIn.duration(300)}
            />
          )}

          {/* Facial landmarks */}
          {landmarks.length > 0 && detectionActive && (
            landmarks.map((point, index) => (
              <View
                key={`landmark-${index}`}
                style={[styles.landmarkPoint, {
                  left: (point[0] / 100) * width - 2,
                  top: (point[1] / 100) * height - 2,
                }]}
              />
            ))
          )}

          {/* Real-time metrics overlay */}
          {detectionActive && (
            <Animated.View style={styles.metricsOverlay} entering={FadeIn.duration(500)}>
              <View style={styles.liveMetricsContainer}>
                <Text style={styles.liveMetricsTitle}>Live Metrics</Text>
                
                <View style={styles.liveMetricRow}>
                  <Ionicons name="heart" size={16} color="#FF6B6B" />
                  <Text style={styles.liveMetricLabel}>HR:</Text>
                  <Text style={styles.liveMetricValue}>
                    {currentMetrics.heartRate !== null && currentMetrics.heartRate !== undefined ? `${Math.round(currentMetrics.heartRate)} BPM` : '--'}
                  </Text>
                </View>
                
                <View style={styles.liveMetricRow}>
                  <Ionicons name="fitness" size={16} color="#4ECDC4" />
                  <Text style={styles.liveMetricLabel}>RR:</Text>
                  <Text style={styles.liveMetricValue}>
                    {currentMetrics.respiratoryRate !== null && currentMetrics.respiratoryRate !== undefined ? `${Math.round(currentMetrics.respiratoryRate)}/min` : '--'}
                  </Text>
                </View>
                
                <View style={styles.liveMetricRow}>
                  <Ionicons name="analytics" size={16} color="#FFD93D" />
                  <Text style={styles.liveMetricLabel}>Stress:</Text>
                  <Text style={styles.liveMetricValue}>
                    {currentMetrics.stressLevel !== null && currentMetrics.stressLevel !== undefined ? `${currentMetrics.stressLevel.toFixed(1)}/10` : '--'}
                  </Text>
                </View>
                
                <View style={styles.liveMetricRow}>
                  <Ionicons name="happy" size={16} color="#6BCF7F" />
                  <Text style={styles.liveMetricLabel}>Emotion:</Text>
                  <Text style={styles.liveMetricValue}>
                    {currentMetrics.emotion || 'Detecting...'}
                  </Text>
                </View>
                
                <View style={styles.confidenceBar}>
                  <Text style={styles.confidenceLabel}>Confidence</Text>
                  <View style={styles.confidenceBarContainer}>
                    <View style={[styles.confidenceBarFill, { 
                      width: `${(currentMetrics.confidence || 0) * 100}%`,
                      backgroundColor: currentMetrics.confidence > 0.7 ? '#00FF00' : currentMetrics.confidence > 0.4 ? '#FFD93D' : '#FF6B6B'
                    }]} />
                  </View>
                  <Text style={styles.confidenceValue}>{Math.round((currentMetrics.confidence || 0) * 100)}%</Text>
                </View>
              </View>
            </Animated.View>
          )}

          {/* Control buttons overlay */}
          <View style={styles.controlsOverlay}>
            {!isStreaming ? (
              <TouchableOpacity
                style={styles.startButton}
                onPress={handleStartDetection}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={['#00D4AA', '#00B4A0']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.buttonGradient}
                >
                  <Ionicons name="play-circle" size={32} color="#fff" />
                  <Text style={styles.startButtonText}>Start Live Detection</Text>
                </LinearGradient>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                style={styles.stopButton}
                onPress={handleStopDetection}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={['#FF6B6B', '#FF5252']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.buttonGradient}
                >
                  <Ionicons name="stop-circle" size={32} color="#fff" />
                  <Text style={styles.stopButtonText}>Stop & View Results</Text>
                </LinearGradient>
              </TouchableOpacity>
            )}
          </View>

          {/* Status indicators */}
          <View style={styles.statusIndicators}>
            <View style={styles.detectionQuality}>
              <Text style={styles.qualityLabel}>Quality: </Text>
              <Text style={[styles.qualityValue, {
                color: detectionQuality === 'Good' ? '#00FF00' : 
                       detectionQuality === 'Acceptable' ? '#FFD93D' : '#FF6B6B'
              }]}>
                {detectionQuality}
              </Text>
            </View>
            
            <View style={styles.faceStatus}>
              <Ionicons 
                name={currentMetrics.faceDetected ? "checkmark-circle" : "close-circle"} 
                size={16} 
                color={currentMetrics.faceDetected ? "#00FF00" : "#FF6B6B"} 
              />
              <Text style={styles.faceStatusText}>
                {currentMetrics.faceDetected ? "Face Detected" : "No Face"}
              </Text>
            </View>

            {detectionActive && (
              <View style={styles.timerBadge}>
                <Ionicons name="time" size={14} color="#E6F1FF" />
                <Text style={styles.timerText}>
                  {`${Math.floor(elapsedSeconds / 60)}:${(elapsedSeconds % 60).toString().padStart(2, '0')}`}
                </Text>
              </View>
            )}
          </View>

          {/* Instructions overlay */}
          {detectionActive && detectionQuality === 'Poor' && (
            <Animated.View style={styles.instructionsOverlay} entering={FadeIn.duration(500)}>
              <View style={styles.instructionsContainer}>
                <Ionicons name="information-circle" size={24} color="#FFD93D" />
                <Text style={styles.instructionsTitle}>Improve Detection Quality</Text>
                <Text style={styles.instructionsText}>
                  • Ensure good lighting{'\n'}
                  • Keep face centered{'\n'}
                  • Stay still for better results{'\n'}
                  • Remove glasses if possible
                </Text>
              </View>
            </Animated.View>
          )}
        </View>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0A192F',
  },
  gradientBackground: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: 'rgba(10, 25, 47, 0.9)',
  },
  backButton: {
    padding: 8,
  },
  closeButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#E6F1FF',
  },
  headerStatus: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  statusText: {
    color: '#A3BFFA',
    fontSize: 12,
    fontWeight: 'bold',
  },
  fullScreenCameraContainer: {
    flex: 1,
    position: 'relative',
  },
  fullScreenCamera: {
    width: '100%',
    height: '100%',
  },
  detectionOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
  },
  scanGrid: {
    flex: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    alignItems: 'center',
    padding: 20,
  },
  scanLine: {
    width: 60,
    height: 2,
    backgroundColor: '#00FF00',
    margin: 10,
    opacity: 0.8,
  },
  faceDetectionBox: {
    position: 'absolute',
    borderWidth: 3,
    borderColor: '#00FF00',
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
    borderRadius: 8,
  },
  landmarkPoint: {
    position: 'absolute',
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#FF6B6B',
  },
  metricsOverlay: {
    position: 'absolute',
    top: 60,
    left: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderRadius: 12,
    padding: 16,
    minWidth: 200,
  },
  liveMetricsContainer: {
    alignItems: 'flex-start',
  },
  liveMetricsTitle: {
    color: '#E6F1FF',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 12,
    textAlign: 'center',
    width: '100%',
  },
  liveMetricRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    width: '100%',
  },
  liveMetricLabel: {
    color: '#A3BFFA',
    fontSize: 14,
    marginLeft: 8,
    minWidth: 40,
  },
  liveMetricValue: {
    color: '#E6F1FF',
    fontSize: 14,
    fontWeight: 'bold',
    marginLeft: 8,
    flex: 1,
  },
  confidenceBar: {
    marginTop: 12,
    width: '100%',
  },
  confidenceLabel: {
    color: '#A3BFFA',
    fontSize: 12,
    marginBottom: 4,
  },
  confidenceBarContainer: {
    height: 6,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 4,
  },
  confidenceBarFill: {
    height: '100%',
    borderRadius: 3,
  },
  confidenceValue: {
    color: '#E6F1FF',
    fontSize: 12,
    textAlign: 'right',
  },
  controlsOverlay: {
    position: 'absolute',
    bottom: 50,
    left: 16,
    right: 16,
    alignItems: 'center',
  },
  startButton: {
    borderRadius: 25,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  stopButton: {
    borderRadius: 25,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  buttonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 32,
  },
  startButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 12,
  },
  stopButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 12,
  },
  statusIndicators: {
    position: 'absolute',
    top: 60,
    right: 16,
    alignItems: 'flex-end',
  },
  detectionQuality: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginBottom: 8,
  },
  qualityLabel: {
    color: '#A3BFFA',
    fontSize: 12,
  },
  qualityValue: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  faceStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  faceStatusText: {
    color: '#E6F1FF',
    fontSize: 12,
    marginLeft: 6,
  },
  timerBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
    marginTop: 8,
  },
  timerText: {
    color: '#E6F1FF',
    fontSize: 12,
    fontWeight: 'bold',
    marginLeft: 6,
  },
  instructionsOverlay: {
    position: 'absolute',
    bottom: 150,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderRadius: 12,
    padding: 16,
  },
  instructionsContainer: {
    alignItems: 'center',
  },
  instructionsTitle: {
    color: '#FFD93D',
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 8,
    marginBottom: 8,
  },
  instructionsText: {
    color: '#E6F1FF',
    fontSize: 14,
    textAlign: 'left',
    lineHeight: 20,
  },
  // Results screen styles
  resultsScrollView: {
    flex: 1,
  },
  resultsContent: {
    padding: 16,
    paddingBottom: 32,
  },
  sessionSummary: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  summaryLabel: {
    color: '#A3BFFA',
    fontSize: 16,
  },
  summaryValue: {
    color: '#E6F1FF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  scoreValue: {
    color: '#00D4AA',
    fontSize: 18,
  },
  statusValue: {
    color: '#64FFDA',
    fontSize: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#E6F1FF',
    marginBottom: 16,
    marginTop: 8,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  metricCard: {
    width: '48%',
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#E6F1FF',
    marginTop: 8,
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 14,
    color: '#A3BFFA',
    textAlign: 'center',
  },
  recommendationsContainer: {
    marginBottom: 24,
  },
  recommendationCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'rgba(100, 255, 218, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#64FFDA',
  },
  recommendationText: {
    color: '#E6F1FF',
    fontSize: 14,
    marginLeft: 12,
    flex: 1,
    lineHeight: 20,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  newScanButton: {
    flex: 1,
    backgroundColor: '#00D4AA',
    borderRadius: 12,
    paddingVertical: 16,
    marginRight: 8,
    alignItems: 'center',
  },
  newScanButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  homeButton: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    paddingVertical: 16,
    marginLeft: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  homeButtonText: {
    color: '#E6F1FF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  // Loading and error states
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0A192F',
  },
  loadingText: {
    color: '#E6F1FF',
    fontSize: 16,
    marginTop: 12,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0A192F',
    padding: 16,
  },
  errorText: {
    color: '#FF6B6B',
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 12,
    textAlign: 'center',
  },
  errorDescription: {
    color: '#A3BFFA',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 24,
    lineHeight: 24,
  },
  requestPermissionButton: {
    backgroundColor: '#00D4AA',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    marginBottom: 16,
  },
  requestPermissionText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  goBackButton: {
    padding: 12,
  },
  goBackText: {
    color: '#A3BFFA',
    fontSize: 16,
  },
});