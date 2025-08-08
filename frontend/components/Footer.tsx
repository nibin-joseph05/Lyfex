// components/Footer.tsx
import { View, Text, StyleSheet } from 'react-native';

export default function Footer() {
  return (
    <View style={styles.footer}>
      <Text style={styles.text}>© 2025 Lyfex Technologies</Text>
      <Text style={styles.text}>Medical-grade health insights</Text>
      <View style={styles.versionBadge}>
        <Text style={styles.version}>v1.0.0</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  footer: {
    paddingVertical: 24,
    alignItems: 'center',
    borderTopWidth: 1,
    borderColor: 'rgba(100, 255, 218, 0.1)',
    backgroundColor: 'rgba(10, 25, 47, 0.7)',
    marginTop: 20,
  },
  text: {
    fontSize: 14,
    color: '#64FFDA',
    fontWeight: '400',
    letterSpacing: 0.5,
    marginBottom: 6,
    fontFamily: 'Inter_400Regular',
  },
  versionBadge: {
    backgroundColor: 'rgba(100, 255, 218, 0.1)',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 20,
    marginTop: 10,
  },
  version: {
    fontSize: 12,
    color: '#64FFDA',
    fontWeight: '500',
    fontFamily: 'Inter_500Medium',
  },
});