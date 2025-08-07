import { View, Text, StyleSheet } from 'react-native';

export default function HomePage() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to Lyfex 👋</Text>
      <Text style={styles.subtitle}>This is your home page</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  title: { fontSize: 28, fontWeight: 'bold' },
  subtitle: { fontSize: 16, marginTop: 10 },
});
