import React, { useEffect, useRef } from 'react';
import { ActivityIndicator, Animated, Easing, StyleSheet, View } from 'react-native';
import { Text } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const COLORS = {
  primary: '#22D3EE',
  gradientStart: '#F0F9FF',
  gradientEnd: '#E0F2FE',
  textPrimary: '#0F172A',
  textSecondary: '#475569',
};

type Props = { message?: string };

export default function LoaderOverlay({ message = 'Please wait...' }: Props) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const rotateAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Fade in with scale
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 300,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        tension: 50,
        friction: 7,
        useNativeDriver: true,
      }),
    ]).start(() => {
      // Start pulse after initial animation
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.05,
            duration: 1000,
            easing: Easing.inOut(Easing.ease),
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 1000,
            easing: Easing.inOut(Easing.ease),
            useNativeDriver: true,
          }),
        ])
      );
      pulse.start();
    });

    // Rotate animation for icon ring
    const rotate = Animated.loop(
      Animated.timing(rotateAnim, {
        toValue: 1,
        duration: 2000,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    );
    rotate.start();

    return () => {
      rotate.stop();
    };
  }, []);

  const rotateValue = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  const combinedScale = Animated.multiply(scaleAnim, pulseAnim);

  return (
    <Animated.View style={[styles.overlay, { opacity: fadeAnim }]}>
      <LinearGradient 
        colors={["rgba(255, 255, 255, 0.95)", "rgba(248, 250, 252, 0.98)", "rgba(255, 255, 255, 0.99)"]} 
        locations={[0, 0.5, 1]}
        style={StyleSheet.absoluteFill} 
      />
      <Animated.View 
        style={[
          styles.content, 
          { 
            transform: [{ scale: combinedScale }] 
          }
        ]}
      >
        <View style={styles.iconContainer}>
          <Animated.View
            style={[
              styles.rotatingRing,
              { transform: [{ rotate: rotateValue }] }
            ]}
          >
            <LinearGradient
              colors={[COLORS.gradientStart, COLORS.gradientEnd]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.ringGradient}
            />
          </Animated.View>
          <View style={styles.iconInner}>
            <MaterialCommunityIcons name="lungs" size={32} color={COLORS.primary} />
          </View>
        </View>
        <ActivityIndicator 
          size="large" 
          color={COLORS.primary} 
          style={styles.spinner}
        />
        <Text variant="titleMedium" style={styles.message}>{message}</Text>
        <Text variant="bodySmall" style={styles.subMessage}>
          Processing audio analysis...
        </Text>
      </Animated.View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  content: {
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 32,
    paddingVertical: 32,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 24,
    elevation: 12,
    minWidth: 240,
    borderWidth: 2,
    borderColor: 'rgba(34, 211, 238, 0.2)',
  },
  iconContainer: {
    width: 80,
    height: 80,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    position: 'relative',
  },
  rotatingRing: {
    position: 'absolute',
    width: 80,
    height: 80,
    borderRadius: 40,
    opacity: 0.2,
  },
  ringGradient: {
    width: '100%',
    height: '100%',
    borderRadius: 40,
  },
  iconInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: `${COLORS.primary}15`,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: `${COLORS.primary}30`,
  },
  spinner: {
    marginVertical: 8,
  },
  message: {
    color: COLORS.textPrimary,
    fontWeight: '700',
    marginTop: 16,
    textAlign: 'center',
    letterSpacing: 0.3,
  },
  subMessage: {
    color: COLORS.textSecondary,
    marginTop: 8,
    textAlign: 'center',
    opacity: 0.8,
  }
});

