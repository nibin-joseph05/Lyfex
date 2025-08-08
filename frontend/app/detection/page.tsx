import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Dimensions, Platform, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import Animated, { FadeIn, FadeInDown, FadeOut } from 'react-native-reanimated';

const { width, height } = Dimensions.get('window');

export default function DetectionPage() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraReady, setIsCameraReady] = useState(false);
  const cameraRef = useRef(null);
  const router = useRouter();

  const [healthMetrics, setHealthMetrics] = useState({
    heartRate: 'N/A',
    respiratoryRate: 'N/A',
    stressLevel: 'N/A',
    emotion: 'N/A',
    facialAsymmetry: 'N/A',
    tremor: 'N/A',
    skinAnalysis: 'N/A',
    fatigue: 'N/A',
  });
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    if (permission && !permission.granted) {
      requestPermission();
    }
  }, []);

  const onCameraReady = () => {
    setIsCameraReady(true);
  };

  if (!permission) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#64FFDA" />
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }
  if (!permission.granted) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="camera-off-outline" size={50} color="#FF6B6B" />
        <Text style={styles.errorText}>No access to camera!</Text>
        <Text style={styles.errorDescription}>Please grant camera permissions in your device settings to use this feature.</Text>
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
                <Text style={styles.overlayText}>
                  Position your face clearly in the frame.
                </Text>
              )}
            </View>
          </CameraView>
        </View>

        <Animated.ScrollView
          style={styles.metricsScrollView}
          contentContainerStyle={styles.metricsContent}
          entering={FadeInDown.duration(800).delay(200)}
        >
          <Text style={styles.sectionTitle}>Real-Time Metrics</Text>
          <View style={styles.metricsGrid}>
            {Object.entries(healthMetrics).map(([key, value]) => (
              <View key={key} style={styles.metricItem}>
                <Text style={styles.metricLabel}>{key.replace(/([A-Z])/g, ' $1').trim()}:</Text>
                <Text style={styles.metricValue}>{value}</Text>
              </View>
            ))}
          </View>

          <Text style={styles.sectionTitle}>Health Alerts</Text>
          {alerts.length > 0 ? (
            <View style={styles.alertsContainer}>
              {alerts.map((alert, index) => (
                <Animated.View key={index} style={styles.alertMessage} entering={FadeIn.delay(index * 100)}>
                  <Ionicons name="warning-outline" size={16} color="#FFD166" />
                  <Text style={styles.alertText}>{alert}</Text>
                </Animated.View>
              ))}
            </View>
          ) : (
            <Text style={styles.noAlertsText}>No immediate alerts. Stay healthy!</Text>
          )}

          <Animated.View entering={FadeInDown.duration(400).delay(700)}>
            <TouchableOpacity
              style={styles.scanButton}
              onPress={() => console.log('Start/Stop Scan Pressed')}
              activeOpacity={0.8}
            >
              <LinearGradient
                colors={['#00D4AA', '#00B4A0']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.scanButtonGradient}
              >
                <Ionicons name="play-circle-outline" size={24} color="#fff" />
                <Text style={styles.scanButtonText}>Start Health Scan</Text>
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>
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
    fontFamily: 'Inter_700Bold',
  },
  headerRightPlaceholder: {
    width: 24,
  },
  cameraContainer: {
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
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
    fontFamily: 'Inter_400Regular',
    textAlign: 'center',
  },
  cameraLoading: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  cameraLoadingText: {
    marginTop: 10,
    color: '#E6F1FF',
    fontFamily: 'Inter_400Regular',
  },
  metricsScrollView: {
    flex: 1,
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#64FFDA',
    marginBottom: 15,
    marginTop: 10,
    fontFamily: 'Inter_600SemiBold',
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
    fontFamily: 'Inter_400Regular',
  },
  metricValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#E6F1FF',
    fontFamily: 'Inter_700Bold',
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
    fontFamily: 'Inter_400Regular',
    flexShrink: 1,
  },
  noAlertsText: {
    color: '#8892B0',
    fontSize: 14,
    textAlign: 'center',
    paddingVertical: 10,
    fontFamily: 'Inter_400Regular',
  },
  scanButton: {
    borderRadius: 14,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#00D4AA',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    marginBottom: 24,
    alignSelf: 'center',
    width: '80%',
  },
  scanButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    paddingHorizontal: 24,
    borderRadius: 14,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    letterSpacing: 0.5,
    marginLeft: 12,
    fontFamily: 'Inter_600SemiBold',
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
    fontFamily: 'Inter_400Regular',
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
    fontFamily: 'Inter_700Bold',
    textAlign: 'center',
  },
  errorDescription: {
    color: '#8892B0',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 30,
    fontFamily: 'Inter_400Regular',
  },
  goBackButton: {
    backgroundColor: '#00D4AA',
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 25,
  },
  goBackText: {
    color: '#fff',
    fontSize: 16,
    fontFamily: 'Inter_600SemiBold',
  },
});