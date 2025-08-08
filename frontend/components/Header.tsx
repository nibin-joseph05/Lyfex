import { View, Text, Image, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { FadeInRight } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export default function Header() {
  const insets = useSafeAreaInsets();

  return (
    <LinearGradient
      colors={['rgba(10, 25, 47, 0.9)', 'rgba(17, 34, 64, 0.8)']}
      style={[styles.header, { paddingTop: insets.top + 25 }]} // Add top padding dynamically
    >
      <Animated.View 
        style={styles.headerContent}
        entering={FadeInRight.duration(800)}
      >
        <Image
          source={require('../assets/logo/logo.png')}
          style={styles.logo}
          resizeMode="contain"
        />
        
        <View style={styles.titleContainer}>
          <Text style={styles.title}>Lyfex</Text>
          <Text style={styles.subtitle}>Advanced Health Intelligence</Text>
        </View>
      </Animated.View>
      
      <View style={styles.waveContainer}>
        <View style={[styles.wave, styles.wave1]} />
        <View style={[styles.wave, styles.wave2]} />
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  header: {
    paddingVertical: 25,
    paddingHorizontal: 24,
    borderBottomWidth: 1,
    borderColor: 'rgba(100, 255, 218, 0.15)',
    overflow: 'hidden',
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    width: 70,
    height: 70,
    marginRight: 16,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#64FFDA',
  },
  titleContainer: {
    alignItems: 'flex-start',
  },
  title: {
    fontSize: 32,
    fontWeight: '700',
    color: '#64FFDA',
    letterSpacing: 1,
    fontFamily: 'Inter_700Bold',
  },
  subtitle: {
    fontSize: 16,
    color: '#8892B0',
    fontWeight: '500',
    letterSpacing: 0.8,
    marginTop: 2,
    fontFamily: 'Inter_500Medium',
  },
  waveContainer: {
    position: 'absolute',
    bottom: -1,
    left: 0,
    right: 0,
    height: 20,
  },
  wave: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 10,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  wave1: {
    backgroundColor: 'rgba(100, 255, 218, 0.15)',
    height: 15,
  },
  wave2: {
    backgroundColor: 'rgba(100, 255, 218, 0.08)',
    height: 20,
  },
});