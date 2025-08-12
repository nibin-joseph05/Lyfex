import React, { useState, useRef, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Dimensions, ActivityIndicator, Platform } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import Animated, { FadeIn, FadeInDown, useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming } from 'react-native-reanimated';

const { width, height } = Dimensions.get('window');
const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'http://192.168.220.6:8000';
const WEBSOCKET_URL = BACKEND_URL.replace('http', 'ws') + '/ws/video_stream';
const VIDEO_STREAM_FPS = 15; // Frames per second for video stream
const DETECTION_CONFIDENCE_THRESHOLD = 0.7;

export default function RealTimeDetectionPage() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isStreaming, setIsStreaming] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [detectionActive, setDetectionActive] = useState(false);
  const [sessionData, setSessionData] = useState(null);
  
  // Real-time detection states
  const [currentMetrics, setCurrentMetrics] = useState({
    heartRate: null,
    respiratoryRate: null,
    stressLevel: null,
    emotion: null,
    confidence: 0,
    faceDetected: false
  });

  // Face detection and tracking
  const [faceBox, setFaceBox] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [detectionQuality, setDetectionQuality] = useState('Searching...');
  
  const [finalResults, setFinalResults] = useState({
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
    overallHealthScore: 'N/A', 
    healthStatus: 'N/A', 
    recommendations: [],
    sessionDuration: '0:00',
    dataPoints: 0
  });

  const cameraRef = useRef(null);
  const ws = useRef(null);
  const router = useRouter();
  const streamInterval = useRef(null);
  const sessionStartTime = useRef(null);
  const analysisData = useRef([]);

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
      analysisData.current = [];
      startVideoStream();
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Real-time detection data:', data);

        // Update real-time metrics
        if (data.realtime_metrics) {
          setCurrentMetrics(prev => ({
            ...prev,
            heartRate: data.realtime_metrics.heart_rate,
            respiratoryRate: data.realtime_metrics.respiratory_rate,
            stressLevel: data.realtime_metrics.stress_level,
            emotion: data.realtime_metrics.emotion,
            confidence: data.realtime_metrics.confidence || 0,
            faceDetected: data.realtime_metrics.face_detected || false
          }));
        }

        // Update face detection visualization
        if (data.face_detection) {
          setFaceBox(data.face_detection.bounding_box);
          setLandmarks(data.face_detection.landmarks || []);
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

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setDetectionActive(false);
    };

    ws.current.onclose = () => {
      console.log('Video stream WebSocket disconnected');
      setDetectionActive(false);
      stopVideoStream();
    };

    return () => {
      if (ws.current) {
        ws.current.close();
      }
      stopVideoStream();
    };
  }, [isStreaming]);

  const startVideoStream = useCallback(() => {
    if (!cameraRef.current) return;

    streamInterval.current = setInterval(async () => {
      if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
        console.warn('WebSocket not ready for video streaming');
        return;
      }

      try {
        // Capture frame for video stream
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.6, // Higher quality for better detection
          skipProcessing: true,
          mirror: true,
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

          ws.current.send(JSON.stringify(frameData));
        }
      } catch (error) {
        console.error('Error capturing video frame:', error);
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
    if (!permission.granted) {
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
      ws.current.close(1000, 'Detection stopped by user');
    }
  }, []);

  const processAnalysisData = (dataPoints) => {
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
        overallHealthScore: 'N/A',
        healthStatus: 'Low Confidence',
        recommendations: ['Insufficient data - longer session recommended']
      };
    }

    // Calculate averages and trends from collected data
    const validData = dataPoints.filter(d => d.confidence > DETECTION_CONFIDENCE_THRESHOLD);
    
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
        overallHealthScore: 'N/A',
        healthStatus: 'Poor Detection Quality',
        recommendations: ['Improve lighting conditions', 'Ensure face is clearly visible']
      };
    }

    // Calculate averages
    const avgHeartRate = Math.round(validData.reduce((sum, d) => sum + (d.heart_rate || 0), 0) / validData.length);
    const avgRespRate = Math.round(validData.reduce((sum, d) => sum + (d.respiratory_rate || 0), 0) / validData.length);
    const avgStress = (validData.reduce((sum, d) => sum + (d.stress_level || 0), 0) / validData.length).toFixed(1);
    
    // Most common emotion
    const emotions = validData.map(d => d.emotion).filter(e => e);
    const emotionCounts = emotions.reduce((acc, emotion) => {
      acc[emotion] = (acc[emotion] || 0) + 1;
      return acc;
    }, {});
    const dominantEmotion = Object.keys(emotionCounts).reduce((a, b) => emotionCounts[a] > emotionCounts[b] ? a : b, 'Neutral');

    // Generate health recommendations
    const recommendations = generateRecommendations({
      heartRate: avgHeartRate,
      stressLevel: parseFloat(avgStress),
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
      stressLevel: avgStress > 0 ? `${avgStress}/10` : 'N/A',
      emotion: dominantEmotion,
      fatigue: 'Analyzing...',
      facialAsymmetry: 'Normal',
      tremor: 'Not Detected',
      eyeMovement: 'Normal',
      skinAnalysis: 'Good',
      skinColor: 'Normal',
      hydrationStatus: 'Good',
      overallHealthScore: `${healthScore}/100`,
      healthStatus: getHealthStatus(healthScore),
      recommendations: recommendations
    };
  };

  const generateRecommendations = (metrics) => {
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

  const calculateHealthScore = (metrics) => {
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

  const getHealthStatus = (score) => {
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
            onCameraReady={() => console.log('Camera ready for video streaming')}
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
                    {currentMetrics.heartRate ? `${Math.round(currentMetrics.heartRate)} BPM` : '--'}
                  </Text>
                </View>
                
                <View style={styles.liveMetricRow}>
                  <Ionicons name="fitness" size={16} color="#4ECDC4" />
                  <Text style={styles.liveMetricLabel}>RR:</Text>
                  <Text style={styles.liveMetricValue}>
                    {currentMetrics.respiratoryRate ? `${Math.round(currentMetrics.respiratoryRate)}/min` : '--'}
                  </Text>
                </View>
                
                <View style={styles.liveMetricRow}>
                  <Ionicons name="analytics" size={16} color="#FFD93D" />
                  <Text style={styles.liveMetricLabel}>Stress:</Text>
                  <Text style={styles.liveMetricValue}>
                    {currentMetrics.stressLevel ? `${currentMetrics.stressLevel.toFixed(1)}/10` : '--'}
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
                       detectionQuality === 'Fair' ? '#FFD93D' : '#FF6B6B'
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
          </View>
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