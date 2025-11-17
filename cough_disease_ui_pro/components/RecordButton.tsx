import React, { useEffect, useRef } from 'react';
import { Animated, Easing, Pressable, StyleSheet, View } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

const COLORS = {
  primary: '#22D3EE',
  primaryDark: '#06B6D4',
  danger: '#F87171',
  dangerDark: '#EF4444',
  gradientStart: '#22D3EE',
  gradientEnd: '#06B6D4',
  dangerGradientStart: '#F87171',
  dangerGradientEnd: '#EF4444',
};

type Props = {
  recording: boolean;
  onPress: () => void;
};

export default function RecordButton({ recording, onPress }: Props) {
  const pulse = useRef(new Animated.Value(0)).current;
  const glow = useRef(new Animated.Value(0)).current;
  const rotate = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    let anim: Animated.CompositeAnimation | null = null;
    let glowAnim: Animated.CompositeAnimation | null = null;
    let rotateAnim: Animated.CompositeAnimation | null = null;
    
    if (recording) {
      // Pulse animation
      anim = Animated.loop(
        Animated.sequence([
          Animated.timing(pulse, { 
            toValue: 1, 
            duration: 1000, 
            easing: Easing.out(Easing.quad), 
            useNativeDriver: true 
          }),
          Animated.timing(pulse, { 
            toValue: 0, 
            duration: 1000, 
            easing: Easing.in(Easing.quad), 
            useNativeDriver: true 
          })
        ])
      );
      anim.start();

      // Glow animation
      glowAnim = Animated.loop(
        Animated.sequence([
          Animated.timing(glow, { 
            toValue: 1, 
            duration: 1500, 
            easing: Easing.inOut(Easing.ease), 
            useNativeDriver: false 
          }),
          Animated.timing(glow, { 
            toValue: 0, 
            duration: 1500, 
            easing: Easing.inOut(Easing.ease), 
            useNativeDriver: false 
          })
        ])
      );
      glowAnim.start();

      // Subtle rotation
      rotateAnim = Animated.loop(
        Animated.timing(rotate, {
          toValue: 1,
          duration: 2000,
          easing: Easing.linear,
          useNativeDriver: true,
        })
      );
      rotateAnim.start();
    } else {
      pulse.stopAnimation();
      pulse.setValue(0);
      glow.stopAnimation();
      glow.setValue(0);
      rotate.stopAnimation();
      rotate.setValue(0);
    }
    return () => {
      anim?.stop();
      glowAnim?.stop();
      rotateAnim?.stop();
    };
  }, [recording]);

  const scale = pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.15] });
  const glowOpacity = glow.interpolate({ inputRange: [0, 1], outputRange: [0.3, 0.7] });
  const rotateValue = rotate.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  const buttonColors = recording 
    ? [COLORS.dangerGradientStart, COLORS.dangerGradientEnd]
    : [COLORS.gradientStart, COLORS.gradientEnd];

  return (
    <Pressable 
      onPress={onPress} 
      accessibilityRole="button" 
      accessibilityLabel={recording ? 'Stop and analyze' : 'Start recording'}
      style={styles.pressable}
    >
      <Animated.View style={[styles.outerRing, { 
        transform: [{ scale }],
        opacity: glowOpacity 
      }]}>
        <LinearGradient
          colors={buttonColors}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.ringGradient}
        />
      </Animated.View>
      <Animated.View style={[styles.circle, { transform: [{ scale }] }]}> 
        <LinearGradient
          colors={buttonColors}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.gradient}
        >
          <Animated.View 
            style={[
              styles.innerCircle,
              { transform: [{ rotate: recording ? rotateValue : '0deg' }] }
            ]}
          >
            <MaterialCommunityIcons 
              name={recording ? 'stop' : 'microphone'} 
              size={40} 
              color="#fff" 
            />
          </Animated.View>
        </LinearGradient>
      </Animated.View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  pressable: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  outerRing: {
    position: 'absolute',
    width: 120,
    height: 120,
    borderRadius: 60,
    shadowColor: COLORS.danger,
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 30,
    shadowOpacity: 0.6,
    elevation: 20,
  },
  ringGradient: {
    width: '100%',
    height: '100%',
    borderRadius: 60,
    opacity: 0.3,
  },
  circle: {
    width: 110,
    height: 110,
    borderRadius: 55,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 12 },
    shadowRadius: 24,
    shadowOpacity: 0.4,
    elevation: 12,
  },
  gradient: {
    width: '100%',
    height: '100%',
    borderRadius: 55,
    alignItems: 'center',
    justifyContent: 'center',
  },
  innerCircle: {
    width: 90,
    height: 90,
    borderRadius: 45,
    borderWidth: 3,
    borderColor: 'rgba(255,255,255,0.4)',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.1)',
  }
});
