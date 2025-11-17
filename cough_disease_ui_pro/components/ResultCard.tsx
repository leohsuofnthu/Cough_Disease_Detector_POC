import React, { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import { Card, Text, Button, Badge } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import ProbabilityChart from './ProbabilityChart';
import { formatDuration, formatTimestamp } from '../utils/time';

const COLORS = {
  primary: '#22D3EE',
  accent: '#10B981',
  accentDark: '#059669',
  danger: '#EF4444',
  warning: '#F59E0B',
  textPrimary: '#0F172A',
  textSecondary: '#475569',
  textMuted: '#64748B',
  successBg: '#D1FAE5',
  successBorder: '#10B981',
  cardBg: '#FFFFFF',
  cardBgSecondary: '#F1F5F9',
  border: '#E2E8F0',
};

const getDiseaseIcon = (label: string): string => {
  const lower = label.toLowerCase();
  if (lower.includes('normal') || lower.includes('healthy')) return 'check-circle';
  if (lower.includes('copd')) return 'lung';
  if (lower.includes('heart')) return 'heart-pulse';
  if (lower.includes('pneumonia')) return 'virus';
  if (lower.includes('bronchiectasis')) return 'lungs';
  if (lower.includes('infection')) return 'bug';
  return 'medical-bag';
};

const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.7) return COLORS.accent;
  if (confidence >= 0.5) return COLORS.warning;
  return COLORS.danger;
};

type Props = {
  result: { label: string; confidence: number; all?: Record<string, number> };
  durationMs: number;
  timestamp: Date;
  onRecordAgain?: () => void;
};

export default function ResultCard({ result, durationMs, timestamp, onRecordAgain }: Props) {
  const chartData = useMemo(() => {
    const entries = Object.entries(result.all ?? {})
      .sort((a, b) => b[1] - a[1])
      .map(([label, value]) => ({ label, value }));
    return entries;
  }, [result]);

  const confidenceColor = getConfidenceColor(result.confidence);
  const iconName = getDiseaseIcon(result.label);

  return (
    <Card style={styles.card} mode="elevated" elevation={8}>
      <LinearGradient
        colors={[COLORS.cardBg, '#F8FAFC']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.gradient}
      >
        <Card.Content style={styles.content}>
          <View style={styles.header}>
            <View style={styles.titleContainer}>
              <MaterialCommunityIcons name="chart-line-variant" size={24} color={COLORS.primary} />
              <Text variant="titleLarge" style={styles.title}>Analysis Result</Text>
            </View>
            <Text variant="bodySmall" style={styles.subtitle}>
              Top prediction and probability distribution
            </Text>
          </View>

          <View style={[styles.resultContainer, { borderColor: `${confidenceColor}40` }]}>
            <View style={styles.resultHeader}>
              <View style={[styles.iconContainer, { backgroundColor: `${confidenceColor}20`, borderColor: `${confidenceColor}60` }]}>
                <MaterialCommunityIcons name={iconName} size={40} color={confidenceColor} />
              </View>
              <View style={styles.resultText}>
                <Text variant="displaySmall" style={styles.diseaseName}>
                  {result.label}
                </Text>
                <View style={styles.confidenceBadge}>
                  <View style={[styles.badgeContainer, { backgroundColor: confidenceColor }]}>
                    <MaterialCommunityIcons name="shield-check" size={16} color="#FFFFFF" />
                    <Text style={styles.badgeText}>
                      {(result.confidence * 100).toFixed(1)}% Confidence
                    </Text>
                  </View>
                </View>
              </View>
            </View>

            <View style={styles.metaContainer}>
              <View style={styles.metaItem}>
                <MaterialCommunityIcons name="clock-outline" size={18} color={COLORS.textSecondary} />
                <Text variant="bodyMedium" style={styles.metaText}>
                  {formatTimestamp(timestamp)}
                </Text>
              </View>
              <View style={styles.metaItem}>
                <MaterialCommunityIcons name="timer-outline" size={18} color={COLORS.textSecondary} />
                <Text variant="bodyMedium" style={styles.metaText}>
                  {formatDuration(durationMs)}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.chartContainer}>
            <Text variant="titleMedium" style={styles.chartTitle}>
              Probability Distribution
            </Text>
            <ProbabilityChart data={chartData} />
          </View>

          {onRecordAgain && (
            <Button 
              mode="outlined" 
              onPress={onRecordAgain} 
              style={styles.recordAgainButton}
              labelStyle={styles.buttonLabel}
              textColor={COLORS.primary}
              icon="microphone"
            >
              Record Again
            </Button>
          )}
        </Card.Content>
      </LinearGradient>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 24,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 16,
    elevation: 8,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  gradient: {
    borderRadius: 24,
  },
  content: {
    paddingVertical: 24,
    paddingHorizontal: 20,
  },
  header: {
    marginBottom: 24,
  },
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  title: {
    color: COLORS.textPrimary,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  subtitle: {
    color: COLORS.textSecondary,
    marginLeft: 32,
    lineHeight: 20,
  },
  resultContainer: {
    backgroundColor: COLORS.cardBgSecondary,
    borderRadius: 20,
    padding: 24,
    marginBottom: 24,
    borderWidth: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 4,
    borderColor: COLORS.border,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 20,
    marginBottom: 20,
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 3,
  },
  resultText: {
    flex: 1,
    gap: 12,
  },
  diseaseName: {
    color: COLORS.textPrimary,
    fontWeight: '800',
    lineHeight: 48,
    fontSize: 36,
    letterSpacing: -0.5,
  },
  confidenceBadge: {
    alignSelf: 'flex-start',
  },
  badgeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },
  badgeText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  badge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    fontSize: 13,
  },
  metaContainer: {
    flexDirection: 'row',
    gap: 24,
    paddingTop: 20,
    borderTopWidth: 2,
    borderTopColor: COLORS.border,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  metaText: {
    color: COLORS.textSecondary,
    fontSize: 14,
    fontWeight: '500',
  },
  chartContainer: {
    marginBottom: 20,
    backgroundColor: '#F8FAFC',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  chartTitle: {
    color: COLORS.textPrimary,
    fontWeight: '700',
    marginBottom: 20,
    letterSpacing: 0.3,
    fontSize: 18,
  },
  recordAgainButton: {
    borderRadius: 12,
    borderWidth: 2,
    borderColor: COLORS.primary,
    marginTop: 8,
  },
  buttonLabel: {
    fontSize: 15,
    fontWeight: '600',
    paddingVertical: 4,
  }
});
