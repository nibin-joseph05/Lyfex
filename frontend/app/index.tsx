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
    { icon: 'body-outline', name: 'Breathing', color: '#4ECDC4' },
    { icon: 'pulse', name: 'Stress', color: '#A78BFA' },
    { icon: 'eye', name: 'Focus', color: '#FFD166' },
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
              <Text style={styles.title}>Health insights, made simple</Text>
              
              <Text style={styles.description}>
                Check key wellness signals in seconds using your phone’s camera.
                Clear, friendly visuals. No clutter. Just what you need.
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
                    colors={['#34D399', '#10B981']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.buttonGradient}
                  >
                    <Ionicons name="scan" size={24} color="#fff" />
                    <Text style={styles.buttonText}>Start scan</Text>
                    <View style={styles.pulseCircle} />
                  </LinearGradient>
                </TouchableOpacity>
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
    paddingBottom: 28,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 44,
  },
  title: {
    fontSize: 30,
    fontWeight: '700',
    color: '#E6F1FF',
    marginBottom: 12,
    letterSpacing: 0.5,
    textAlign: 'center',
    fontFamily: 'Inter_700Bold',
  },
  description: {
    fontSize: 15,
    color: '#9AA8C7',
    textAlign: 'center',
    marginBottom: 28,
    lineHeight: 22,
    fontWeight: '400',
    paddingHorizontal: 12,
    fontFamily: 'Inter_400Regular',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 26,
  },
  metricCard: {
    width: '48%',
    borderRadius: 14,
    padding: 16,
    marginBottom: 14,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(100, 255, 218, 0.08)',
    backgroundColor: 'rgba(17, 34, 64, 0.55)',
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
    shadowColor: '#10B981',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    marginBottom: 8,
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
    fontSize: 17,
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
    backgroundColor: 'rgba(16, 185, 129, 0.18)',
  },
});