import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Dimensions, ActivityIndicator, ScrollView } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';
import axios from 'axios';

const { width, height } = Dimensions.get('window');

// Use only EXPO_PUBLIC_ environment variables (works with app.json)
const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'http://192.168.220.6:8000';

export default function DetectionPage() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const cameraRef = useRef(null);
  const router = useRouter();

  const [healthMetrics, setHealthMetrics] = useState({
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
    recommendations: [], // Changed to array for recommendations
  });
  const [alerts, setAlerts] = useState([]);
  const [analysisQuality, setAnalysisQuality] = useState('N/A');
  const [analysisTimestamp, setAnalysisTimestamp] = useState('N/A');


  const onCameraReady = () => {
    setIsCameraReady(true);
  };

  const captureImage = async () => {
    if (cameraRef.current && isCameraReady) {
      try {
        const photo = await cameraRef.current.takePictureAsync({ base64: true });
        return photo;
      } catch (error) {
        console.error('Error capturing image:', error);
        setAlerts(prevAlerts => [...prevAlerts, 'Failed to capture image']);
        return null;
      }
    }
    return null;
  };

  const sendToBackend = async (photo) => {
    setIsScanning(true); // Ensure scanning indicator is on when sending to backend
    setAlerts([]); // Clear previous alerts
    setHealthMetrics({ // Reset metrics to N/A while scanning
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
    });

    try {
      console.log('Using backend URL:', BACKEND_URL); // Debug log
      
      const formData = new FormData();
      formData.append('file', {
        uri: photo.uri,
        name: 'photo.jpg',
        type: 'image/jpeg',
      });

      const response = await axios.post(`${BACKEND_URL}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 15000, // Increased timeout to 15 seconds for analysis
      });

      console.log('Backend response:', response.data);

      // Update state with all received metrics and metadata
      setHealthMetrics({
        heartRate: response.data.heartRate || 'N/A',
        respiratoryRate: response.data.respiratoryRate || 'N/A',
        stressLevel: response.data.stressLevel || 'N/A',
        emotion: response.data.emotion || 'N/A',
        fatigue: response.data.fatigue || 'N/A',
        facialAsymmetry: response.data.facialAsymmetry || 'N/A',
        tremor: response.data.tremor || 'N/A',
        eyeMovement: response.data.eyeMovement || 'N/A',
        skinAnalysis: response.data.skinAnalysis || 'N/A',
        skinColor: response.data.skinColor || 'N/A',
        hydrationStatus: response.data.hydrationStatus || 'N/A',
        overallHealthScore: response.data.overallHealthScore || 'N/A',
        healthStatus: response.data.healthStatus || 'N/A',
        recommendations: response.data.recommendations || [],
      });
      setAlerts(response.data.alerts || []);
      setAnalysisQuality(response.data.analysis_quality || 'N/A');
      setAnalysisTimestamp(response.data.analysis_timestamp || 'N/A');

    } catch (error) {
      console.error('Error sending image to backend:', error);
      console.error('Backend URL being used:', BACKEND_URL);
      
      let errorMessage = 'Failed to analyze image';
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Request timeout - check network or increase timeout';
      } else if (error.response) {
        errorMessage = `Server error (${error.response.status}): ${error.response.data.detail || error.response.data.error || 'Unknown error'}`;
      } else if (error.request) {
        errorMessage = 'Network error - unable to reach server. Is backend running?';
      }
      
      setAlerts(prevAlerts => [...prevAlerts, errorMessage]);
    } finally {
      setIsScanning(false); // Stop scanning regardless of success or failure
    }
  };

  const toggleScanning = async () => {
    if (!isScanning) {
      // Start scanning
      const photo = await captureImage();
      if (photo) {
        await sendToBackend(photo);
      } else {
        setIsScanning(false); // Stop scanning if image capture fails
      }
    } else {
      console.log('Stopping health scan');
      setIsScanning(false); // Manually stop scanning
    }
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
        <TouchableOpacity 
          style={styles.requestPermissionButton} 
          onPress={requestPermission}
        >
          <Text style={styles.requestPermissionText}>Grant Permission</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.goBackButton} onPress={() => router.back()}>
          <Text style={styles.goBackText}>Go Back</Text>
        </TouchableOpacity>
      </View>
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
          <Text style={styles.headerTitle}>Health Scan</Text>
          <View style={styles.headerRightPlaceholder} />
        </View>

        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing="front"
            onCameraReady={onCameraReady}
          >
            <View style={styles.cameraOverlay}>
              {!isCameraReady ? (
                <>
                  <ActivityIndicator size="large" color="#64FFDA" />
                  <Text style={styles.overlayText}>Initializing Camera...</Text>
                </>
              ) : (
                <>
                  <Text style={styles.overlayText}>
                    Position your face clearly in the frame
                  </Text>
                  {isScanning && (
                    <View style={styles.scanningIndicator}>
                      <ActivityIndicator size="small" color="#00D4AA" />
                      <Text style={styles.scanningText}>Scanning...</Text>
                    </View>
                  )}
                </>
              )}
            </View>
          </CameraView>
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
                  {Array.isArray(value) ? value.join(', ') : value}
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
                    style={styles.alertMessage} // Reusing alertMessage style for consistency
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

          <TouchableOpacity
            style={[styles.scanButton, isScanning && styles.scanButtonActive]}
            onPress={toggleScanning}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={isScanning ? ['#FF6B6B', '#FF5252'] : ['#00D4AA', '#00B4A0']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.scanButtonGradient}
            >
              <Ionicons 
                name={isScanning ? "stop-circle-outline" : "play-circle-outline"} 
                size={24} 
                color="#fff" 
              />
              <Text style={styles.scanButtonText}>
                {isScanning ? 'Stop Scan' : 'Start Health Scan'}
              </Text>
            </LinearGradient>
          </TouchableOpacity>
        </Animated.ScrollView>
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
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderColor: 'rgba(100, 255, 218, 0.15)',
    backgroundColor: 'rgba(10, 25, 47, 0.9)',
  },
  backButton: {
    padding: 5,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#E6F1FF',
  },
  headerRightPlaceholder: {
    width: 24,
  },
  cameraContainer: {
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: '#000',
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
    textAlign: 'center',
    marginBottom: 10,
  },
  scanningIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 212, 170, 0.2)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  scanningText: {
    color: '#00D4AA',
    marginLeft: 8,
    fontSize: 14,
  },
  metricsScrollView: {
    flex: 1,
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  metricsContent: {
    paddingBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#64FFDA',
    marginBottom: 15,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  metricItem: {
    width: '48%',
    backgroundColor: 'rgba(17, 34, 64, 0.5)',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: 'rgba(100, 255, 218, 0.1)',
  },
  metricLabel: {
    fontSize: 13,
    color: '#8892B0',
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#E6F1FF',
  },
  alertsContainer: {
    backgroundColor: 'rgba(255, 209, 102, 0.1)',
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#FFD166',
  },
  alertMessage: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  alertText: {
    marginLeft: 10,
    color: '#FFD166',
    fontSize: 14,
    flexShrink: 1,
  },
  scanButton: {
    borderRadius: 14,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#00D4AA',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    alignSelf: 'center',
    width: '80%',
  },
  scanButtonActive: {
    shadowColor: '#FF6B6B',
  },
  scanButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    paddingHorizontal: 24,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    letterSpacing: 0.5,
    marginLeft: 12,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0A192F',
  },
  loadingText: {
    marginTop: 10,
    color: '#E6F1FF',
    fontSize: 16,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0A192F',
    padding: 20,
  },
  errorText: {
    color: '#FF6B6B',
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
    textAlign: 'center',
  },
  errorDescription: {
    color: '#8892B0',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 20,
  },
  requestPermissionButton: {
    backgroundColor: '#00D4AA',
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 25,
    marginBottom: 15,
  },
  requestPermissionText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  goBackButton: {
    backgroundColor: 'rgba(100, 255, 218, 0.1)',
    borderColor: '#64FFDA',
    borderWidth: 1,
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 25,
  },
  goBackText: {
    color: '#64FFDA',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
});