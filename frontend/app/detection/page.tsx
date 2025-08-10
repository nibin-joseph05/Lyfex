import React, { useState, useRef, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Dimensions, ActivityIndicator, ScrollView, Platform } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

const { width } = Dimensions.get('window');
const CAMERA_ASPECT_RATIO = 4 / 3;
const CAMERA_WIDTH = width - 32;
const CAMERA_HEIGHT = CAMERA_WIDTH * CAMERA_ASPECT_RATIO;
const BACKEND_FRAME_WIDTH = 1920;
const BACKEND_FRAME_HEIGHT = 1080;
const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'http://192.168.220.6:8000';
const WEBSOCKET_URL = BACKEND_URL.replace('http', 'ws') + '/ws/analyze_stream';
const FRAME_PROCESSING_INTERVAL_MS = 500;
const KEEP_ALIVE_INTERVAL_MS = 10000;

export default function DetectionPage() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [faceData, setFaceData] = useState(null);
  const [healthMetrics, setHealthMetrics] = useState({
    heartRate: 'N/A', respiratoryRate: 'N/A', stressLevel: 'N/A', emotion: 'N/A',
    fatigue: 'N/A', facialAsymmetry: 'N/A', tremor: 'N/A', eyeMovement: 'N/A',
    skinAnalysis: 'N/A', skinColor: 'N/A', hydrationStatus: 'N/A',
    overallHealthScore: 'N/A', healthStatus: 'N/A', recommendations: [],
  });
  const [alerts, setAlerts] = useState([]);
  const [analysisQuality, setAnalysisQuality] = useState('N/A');
  const [analysisTimestamp, setAnalysisTimestamp] = useState('N/A');
  const [errorMessage, setErrorMessage] = useState(null);
  const [wsStatus, setWsStatus] = useState('Disconnected');
  const [cameraFeedSize, setCameraFeedSize] = useState({ width: CAMERA_WIDTH, height: CAMERA_HEIGHT });
  const cameraRef = useRef(null);
  const ws = useRef(null);
  const router = useRouter();

  // Get actual camera feed size
  const onCameraReady = useCallback(() => {
    console.log('Camera is ready');
    setIsCameraReady(true);
    setErrorMessage(null);
    // Estimate actual camera feed size
    if (cameraRef.current) {
      // Note: expo-camera doesn't provide direct access to feed size, so we rely on container
      // For better accuracy, use device-specific camera feed size if available
      setCameraFeedSize({ width: CAMERA_WIDTH, height: CAMERA_HEIGHT });
    }
  }, []);

  useEffect(() => {
    if (!isStreaming) {
      console.log('Streaming stopped, cleaning up WebSocket');
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.close(1000, 'Streaming stopped');
      }
      setWsStatus('Disconnected');
      return;
    }

    console.log('Starting WebSocket connection:', WEBSOCKET_URL);
    ws.current = new WebSocket(WEBSOCKET_URL);
    setWsStatus('Connecting');

    ws.current.onopen = () => {
      console.log('WebSocket connection opened');
      setWsStatus('Connected');
      setAlerts(prevAlerts => [...prevAlerts, 'Live analysis started']);
      setErrorMessage(null);
      const keepAlive = setInterval(() => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ type: 'ping' }));
          console.log('Sent keep-alive ping');
        }
      }, KEEP_ALIVE_INTERVAL_MS);
      ws.current.keepAlive = keepAlive;
    };

    ws.current.onmessage = (event) => {
      try {
        console.log('Raw WebSocket message:', event.data);
        const receivedData = JSON.parse(event.data);
        console.log('Received from backend:', JSON.stringify(receivedData, null, 2));

        if (receivedData.type === 'pong') {
          console.log('Received keep-alive pong');
          return;
        }

        if (receivedData.error) {
          console.warn('Backend error:', receivedData.error);
          setAlerts(prevAlerts => [...prevAlerts, receivedData.error]);
          setErrorMessage(receivedData.error);
          return;
        }

        setHealthMetrics(prevMetrics => ({
          ...prevMetrics,
          heartRate: receivedData.heartRate || 'N/A',
          respiratoryRate: receivedData.respiratoryRate || 'N/A',
          stressLevel: receivedData.stressLevel || 'N/A',
          emotion: receivedData.emotion || 'N/A',
          fatigue: receivedData.fatigue || 'N/A',
          facialAsymmetry: receivedData.facialAsymmetry || 'N/A',
          treadmill: receivedData.tremor || 'N/A',
          eyeMovement: receivedData.eyeMovement || 'N/A',
          skinAnalysis: receivedData.skinAnalysis || 'N/A',
          skinColor: receivedData.skinColor || 'N/A',
          hydrationStatus: receivedData.hydrationStatus || 'N/A',
          overallHealthScore: receivedData.overallHealthScore || 'N/A',
          healthStatus: receivedData.healthStatus || 'N/A',
          recommendations: receivedData.recommendations || [],
        }));
        setAlerts(receivedData.alerts || []);
        setAnalysisQuality(receivedData.analysis_quality || 'N/A');
        setAnalysisTimestamp(receivedData.analysis_timestamp || 'N/A');
        setFaceData(receivedData.face_data || null);
        if (receivedData.face_data && receivedData.face_data.bounding_box) {
          console.log('Bounding box received:', receivedData.face_data.bounding_box);
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', e.message);
        setAlerts(prevAlerts => [...prevAlerts, 'Error receiving data']);
        setErrorMessage('Failed to process backend data');
      }
    };

    ws.current.onerror = (e) => {
      console.error('WebSocket error:', e);
      setWsStatus('Error');
      setAlerts(prevAlerts => [...prevAlerts, `WebSocket error: ${e.message || 'Unknown'}`]);
      setErrorMessage('WebSocket connection failed');
      setIsStreaming(false);
    };

    ws.current.onclose = (event) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
      setWsStatus('Disconnected');
      setAlerts(prevAlerts => [...prevAlerts, `Live analysis stopped: ${event.reason || 'Unknown'}`]);
      setErrorMessage(`Analysis stopped: ${event.reason || 'Connection closed'}`);
      setIsStreaming(false);
      if (ws.current && ws.current.keepAlive) {
        clearInterval(ws.current.keepAlive);
      }
    };

    return () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        console.log('Closing WebSocket on cleanup');
        ws.current.close(1000, 'Component unmount');
      }
      if (ws.current && ws.current.keepAlive) {
        clearInterval(ws.current.keepAlive);
      }
    };
  }, [isStreaming]);

  const toggleStreaming = async () => {
    if (!permission.granted) {
      console.log('Requesting camera permission');
      const { granted } = await requestPermission();
      if (!granted) {
        console.warn('Camera permission denied');
        setAlerts(prevAlerts => [...prevAlerts, 'Camera permission denied']);
        setErrorMessage('Camera permission required');
        return;
      }
    }

    setIsStreaming(prev => {
      console.log('Toggling streaming state to:', !prev);
      return !prev;
    });
    setAlerts([]);
    setHealthMetrics({
      heartRate: 'N/A', respiratoryRate: 'N/A', stressLevel: 'N/A', emotion: 'N/A',
      fatigue: 'N/A', facialAsymmetry: 'N/A', tremor: 'N/A', eyeMovement: 'N/A',
      skinAnalysis: 'N/A', skinColor: 'N/A', hydrationStatus: 'N/A',
      overallHealthScore: 'N/A', healthStatus: 'N/A', recommendations: [],
    });
    setFaceData(null);
    setErrorMessage(null);
    setWsStatus('Disconnected');
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
          This app needs camera access to perform health scans.
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

  // Calculate scaling factors
  const scaleX = cameraFeedSize.width / BACKEND_FRAME_WIDTH;
  const scaleY = cameraFeedSize.height / BACKEND_FRAME_HEIGHT;

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
          <Text style={styles.headerTitle}>Health Scan</Text>
          <View style={styles.headerRightPlaceholder}>
            <Text style={styles.wsStatusText}>{wsStatus}</Text>
          </View>
        </View>

        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing="front"
            onCameraReady={onCameraReady}
            onMountError={(error) => {
              console.error('Camera Mount Error:', error);
              setErrorMessage('Failed to initialize camera');
            }}
          />
          <FrameCaptureInterval
            cameraRef={cameraRef}
            ws={ws}
            interval={FRAME_PROCESSING_INTERVAL_MS}
            isStreaming={isStreaming}
            setErrorMessage={setErrorMessage}
          />
          <View style={styles.cameraOverlay}>
            {!isCameraReady ? (
              <>
                <ActivityIndicator size="large" color="#64FFDA" />
                <Text style={styles.overlayText}>Initializing Camera...</Text>
              </>
            ) : (
              <>
                <Text style={[styles.overlayText, errorMessage && styles.errorText]}>
                  {errorMessage || 'Position your face clearly in the frame'}
                </Text>
                {isStreaming && (
                  <View style={styles.scanningIndicator}>
                    <ActivityIndicator size="small" color="#00D4AA" />
                    <Text style={styles.scanningText}>Live Scanning...</Text>
                  </View>
                )}
                {faceData && faceData.bounding_box && (
                  <View
                    style={[styles.faceBox, {
                      left: faceData.bounding_box.x * scaleX,
                      top: faceData.bounding_box.y * scaleY,
                      width: faceData.bounding_box.width * scaleX,
                      height: faceData.bounding_box.height * scaleY,
                    }]}
                  />
                )}
                {faceData && faceData.landmarks && faceData.landmarks.length > 0 && (
                  faceData.landmarks.map((point, index) => (
                    <View
                      key={`landmark-${index}`}
                      style={[styles.landmarkPoint, {
                        left: point[0] * scaleX - 2,
                        top: point[1] * scaleY - 2,
                      }]}
                    />
                  ))
                )}
              </>
            )}
          </View>
        </View>

        <View style={styles.controlsContainer}>
          <TouchableOpacity
            style={[styles.scanButton, isStreaming && styles.scanButtonActive]}
            onPress={toggleStreaming}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={isStreaming ? ['#FF6B6B', '#FF5252'] : ['#00D4AA', '#00B4A0']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.scanButtonGradient}
            >
              <Ionicons
                name={isStreaming ? "stop-circle-outline" : "play-circle-outline"}
                size={24}
                color="#fff"
              />
              <Text style={styles.scanButtonText}>
                {isStreaming ? 'Stop Live Scan' : 'Start Live Health Scan'}
              </Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>

        <Animated.ScrollView
          style={styles.metricsScrollView}
          contentContainerStyle={styles.metricsContent}
          entering={FadeInDown.duration(600)}
        >
          <Text style={styles.sectionTitle}>Real-Time Metrics</Text>
          <View style={styles.metricsGrid}>
            {Object.entries(healthMetrics).filter(([key]) => key !== 'recommendations').map(([key, value]) => (
              <View key={key} style={styles.metricItem}>
                <Text style={styles.metricLabel}>
                  {key.replace(/([A-Z])/g, ' $1').trim()}:
                </Text>
                <Text style={styles.metricValue}>
                  {Array.isArray(value) ? value.join(', ') : String(value)}
                </Text>
              </View>
            ))}
          </View>

          {healthMetrics.recommendations.length > 0 && (
            <>
              <Text style={styles.sectionTitle}>Recommendations</Text>
              <View style={styles.alertsContainer}>
                {healthMetrics.recommendations.map((rec, index) => (
                  <Animated.View
                    key={`rec-${index}`}
                    style={styles.alertMessage}
                    entering={FadeIn.delay(index * 100)}
                  >
                    <Ionicons name="bulb-outline" size={16} color="#64FFDA" />
                    <Text style={styles.alertText}>{rec}</Text>
                  </Animated.View>
                ))}
              </View>
            </>
          )}

          {alerts.length > 0 && (
            <>
              <Text style={styles.sectionTitle}>Health Alerts</Text>
              <View style={styles.alertsContainer}>
                {alerts.map((alert, index) => (
                  <Animated.View
                    key={`alert-${index}`}
                    style={styles.alertMessage}
                    entering={FadeIn.delay(index * 100)}
                  >
                    <Ionicons name="warning-outline" size={16} color="#FFD166" />
                    <Text style={styles.alertText}>{alert}</Text>
                  </Animated.View>
                ))}
              </View>
            </>
          )}
        </Animated.ScrollView>
      </LinearGradient>
    </SafeAreaView>
  );
}

const FrameCaptureInterval = ({ cameraRef, ws, interval, isStreaming, setErrorMessage }) => {
  const [isWsReady, setIsWsReady] = useState(false);
  const shouldCapture = useRef(true);

  useEffect(() => {
    if (!isStreaming) {
      setIsWsReady(false);
      shouldCapture.current = false;
      console.log('Streaming stopped, disabling frame capture');
      return;
    }

    shouldCapture.current = true;
    console.log('Starting WebSocket readiness check');
    const checkWsReady = setInterval(() => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        setIsWsReady(true);
        console.log('WebSocket is ready for frame capture');
        clearInterval(checkWsReady);
      } else {
        console.log('Waiting for WebSocket to be ready:', ws.current ? ws.current.readyState : 'No WebSocket');
      }
    }, 100);

    let intervalId;
    if (cameraRef.current && isWsReady && isStreaming) {
      console.log('Starting frame capture interval');
      intervalId = setInterval(async () => {
        if (!shouldCapture.current || !ws.current || ws.current.readyState !== WebSocket.OPEN) {
          console.warn('Cannot capture frame: streaming stopped or WebSocket not open');
          setErrorMessage('WebSocket not connected or streaming stopped');
          return;
        }
        try {
          console.log('Attempting to capture frame');
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.3,
            skipProcessing: true,
            mirror: true, // Ensure front-facing camera feed is mirrored to match backend
          });
          if (photo && photo.base64) {
            console.log('Frame captured, base64 length:', photo.base64.length);
            ws.current.send(JSON.stringify({ image: photo.base64 }));
            console.log('Frame sent to WebSocket at:', new Date().toISOString());
          } else {
            console.warn('No base64 data in captured photo:', photo);
            setErrorMessage('Failed to capture frame data');
          }
        } catch (error) {
          console.error('Frame capture error:', error.message);
          setErrorMessage(`Frame capture failed: ${error.message}`);
        }
      }, interval);
    }

    return () => {
      console.log('Cleaning up frame capture interval');
      shouldCapture.current = false;
      if (intervalId) {
        clearInterval(intervalId);
      }
      clearInterval(checkWsReady);
    };
  }, [cameraRef, ws, interval, isWsReady, isStreaming, setErrorMessage]);

  return null;
};

// Styles (unchanged except for cameraOverlay to ensure proper alignment)
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
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#E6F1FF',
  },
  headerRightPlaceholder: {
    width: 80,
    alignItems: 'flex-end',
  },
  wsStatusText: {
    color: '#A3BFFA',
    fontSize: 12,
  },
  cameraContainer: {
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT,
    backgroundColor: '#000',
    alignSelf: 'center',
    marginVertical: 8,
    overflow: 'hidden', // Prevent overflow of camera feed
  },
  camera: {
    width: '100%',
    height: '100%',
  },
  cameraOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.3)',
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 20,
  },
  overlayText: {
    color: '#E6F1FF',
    fontSize: 16,
    fontWeight: '500',
    textAlign: 'center',
    marginBottom: 10,
  },
  errorText: {
    color: '#FF6B6B',
  },
  scanningIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 8,
    borderRadius: 8,
  },
  scanningText: {
    color: '#00D4AA',
    fontSize: 14,
    marginLeft: 8,
  },
  faceBox: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00FF00',
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
  },
  landmarkPoint: {
    position: 'absolute',
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#FF0000',
  },
  controlsContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: 'rgba(10, 25, 47, 0.9)',
  },
  scanButton: {
    borderRadius: 12,
    overflow: 'hidden',
  },
  scanButtonActive: {
    opacity: 0.9,
  },
  scanButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  metricsScrollView: {
    flex: 1,
  },
  metricsContent: {
    padding: 16,
    paddingBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#E6F1FF',
    marginBottom: 12,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricItem: {
    width: '48%',
    backgroundColor: 'rgba(255,255,255,0.05)',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  metricLabel: {
    fontSize: 14,
    color: '#A3BFFA',
  },
  metricValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#E6F1FF',
    marginTop: 4,
  },
  alertsContainer: {
    marginBottom: 16,
  },
  alertMessage: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.05)',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  alertText: {
    color: '#E6F1FF',
    fontSize: 14,
    marginLeft: 8,
    flex: 1,
  },
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
  },
  errorDescription: {
    color: '#A3BFFA',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 16,
  },
  requestPermissionButton: {
    backgroundColor: '#00D4AA',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    marginBottom: 12,
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