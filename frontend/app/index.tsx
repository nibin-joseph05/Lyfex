import { View, Text, StyleSheet, TouchableOpacity, ImageBackground, ScrollView, SafeAreaView } from 'react-native';
import { useRouter } from 'expo-router';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

export default function HomePage() {
  const router = useRouter();

  const healthMetrics = [
    { icon: 'heart', name: 'Heart Rate', color: '#FF6B6B' },
    { icon: 'body-outline', name: 'Respiration', color: '#4ECDC4' },
    { icon: 'eye', name: 'Neurological', color: '#FFD166' },
    { icon: 'pulse', name: 'Stress Levels', color: '#6A0572' },
  ];

  return (
    <ImageBackground 
      source={require('../assets/bg-pattern.webp')} 
      style={styles.container}
      blurRadius={2}
    >
      <SafeAreaView style={{ flex: 1 }}>
        <LinearGradient
          colors={['rgba(10, 25, 47, 0.9)', 'rgba(17, 34, 64, 0.95)']}
          style={styles.gradientOverlay}
        >
          <Header />
          
          <ScrollView 
            contentContainerStyle={styles.scroll}
          >
            <Animated.View 
              style={styles.content}
              entering={FadeInDown.duration(600).delay(200)}
            >
              <Text style={styles.title}>Comprehensive Health Insights</Text>
              
              <Text style={styles.description}>
                Lyfex uses advanced AI and computer vision to monitor 12+ vital health parameters in real-time. 
                Medical-grade precision from your phone's camera.
              </Text>
              
              <View style={styles.metricsGrid}>
                {healthMetrics.map((metric, index) => (
                  <Animated.View 
                    key={metric.name}
                    style={[styles.metricCard, { backgroundColor: `${metric.color}15` }]}
                    entering={FadeInDown.duration(400).delay(300 + index * 100)}
                  >
                    <Ionicons 
                      name={metric.icon} 
                      size={28} 
                      color={metric.color} 
                    />
                    <Text style={[styles.metricText, { color: metric.color }]}>
                      {metric.name}
                    </Text>
                  </Animated.View>
                ))}
              </View>
              
              <Animated.View entering={FadeInDown.duration(400).delay(700)}>
                <TouchableOpacity
                  style={styles.button}
                  onPress={() => router.push('/detection/page')}
                  activeOpacity={0.8}
                >
                  <LinearGradient
                    colors={['#00D4AA', '#00B4A0']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.buttonGradient}
                  >
                    <Ionicons name="scan" size={24} color="#fff" />
                    <Text style={styles.buttonText}>Start Health Scan</Text>
                    <View style={styles.pulseCircle} />
                  </LinearGradient>
                </TouchableOpacity>
              </Animated.View>
              
              <Animated.View 
                style={styles.featureCard}
                entering={FadeInDown.duration(400).delay(900)}
              >
                <View style={styles.featureIcon}>
                  <Ionicons name="medkit" size={20} color="#64FFDA" />
                </View>
                <View style={styles.featureTextContainer}>
                  <Text style={styles.featureTitle}>Medical-Grade Accuracy</Text>
                  <Text style={styles.featureText}>
                    Validated against clinical equipment with ±3 BPM accuracy
                  </Text>
                </View>
              </Animated.View>
            </Animated.View>
            
            <Footer />
          </ScrollView>
        </LinearGradient>
      </SafeAreaView>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradientOverlay: {
    flex: 1,
  },
  scroll: {
    flexGrow: 1,
    justifyContent: 'space-between',
    paddingBottom: 20,
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
    paddingTop: 20,
    paddingBottom: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#E6F1FF',
    marginBottom: 16,
    letterSpacing: 0.5,
    textAlign: 'center',
    fontFamily: 'Inter_700Bold',
  },
  description: {
    fontSize: 16,
    color: '#8892B0',
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 24,
    fontWeight: '400',
    paddingHorizontal: 10,
    fontFamily: 'Inter_400Regular',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 30,
  },
  metricCard: {
    width: '48%',
    borderRadius: 16,
    padding: 16,
    marginBottom: 15,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(100, 255, 218, 0.1)',
    backgroundColor: 'rgba(17, 34, 64, 0.5)',
  },
  metricText: {
    marginTop: 8,
    fontSize: 14,
    fontWeight: '600',
    fontFamily: 'Inter_600SemiBold',
    textAlign: 'center',
  },
  button: {
    borderRadius: 14,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#00D4AA',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    marginBottom: 24,
  },
  buttonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    paddingHorizontal: 24,
    borderRadius: 14,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    letterSpacing: 0.5,
    marginLeft: 12,
    fontFamily: 'Inter_600SemiBold',
  },
  pulseCircle: {
    position: 'absolute',
    right: -20,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(100, 255, 218, 0.2)',
  },
  featureCard: {
    backgroundColor: 'rgba(17, 34, 64, 0.6)',
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'flex-start',
    borderWidth: 1,
    borderColor: 'rgba(100, 255, 218, 0.15)',
  },
  featureIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: 'rgba(100, 255, 218, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
    marginTop: 2,
  },
  featureTextContainer: {
    flex: 1,
  },
  featureTitle: {
    color: '#64FFDA',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
    fontFamily: 'Inter_600SemiBold',
  },
  featureText: {
    color: '#8892B0',
    fontSize: 14,
    lineHeight: 20,
    fontFamily: 'Inter_400Regular',
  },
});