import React from 'react';
import { Platform, StyleSheet, View } from 'react-native';
import { Text } from 'react-native-paper';
import Svg, { Rect } from 'react-native-svg';

const COLORS = {
  squareFilled: '#0F172A',
  squareOutline: '#64748B',
  text: '#0F172A',
  border: '#E2E8F0',
  background: '#FFFFFF',
};

type Props = {
  data: { label: string; value: number }[];
  barColor?: string;
};

// Component to render a single confidence square
const ConfidenceSquare = ({ filled, isLast }: { filled: boolean; isLast?: boolean }) => {
  const size = 12;
  const strokeWidth = 1;
  
  if (Platform.OS === 'web') {
    return (
      <View
        style={[
          styles.square,
          filled ? styles.squareFilled : styles.squareUnfilled,
          { width: size, height: size },
          !isLast && styles.squareMargin
        ]}
      />
    );
  }

  // Native rendering using SVG
  return (
    <View style={!isLast && styles.squareMargin}>
      <Svg width={size} height={size} style={styles.squareSvg}>
        <Rect
          x={strokeWidth / 2}
          y={strokeWidth / 2}
          width={size - strokeWidth}
          height={size - strokeWidth}
          fill={filled ? COLORS.squareFilled : 'none'}
          stroke={COLORS.squareOutline}
          strokeWidth={strokeWidth}
        />
      </Svg>
    </View>
  );
};

// Component to render the heatmap-style confidence block (10 squares)
const ConfidenceHeatmap = ({ value }: { value: number }) => {
  // Calculate how many squares should be filled (0-10)
  // Round to nearest integer, but ensure at least 1 square if value > 0.05
  let filledCount = Math.round(Math.min(10, Math.max(0, value * 10)));
  if (value > 0.05 && filledCount === 0) {
    filledCount = 1;
  }
  
  return (
    <View style={styles.heatmapContainer}>
      {Array.from({ length: 10 }, (_, index) => (
        <ConfidenceSquare key={index} filled={index < filledCount} isLast={index === 9} />
      ))}
    </View>
  );
};

export default function ProbabilityChart({ data }: Props) {
  return (
    <View style={styles.container}>
      {/* Table Header */}
      <View style={styles.tableHeader}>
        <View style={styles.classColumn}>
          <Text style={styles.headerText}>Class</Text>
        </View>
        <View style={styles.confidenceColumn}>
          <Text style={styles.headerText}>Confidence</Text>
        </View>
      </View>

      {/* Table Rows */}
      {data.map((item, index) => (
        <View
          key={item.label}
          style={[
            styles.tableRow,
            index < data.length - 1 && styles.tableRowBorder
          ]}
        >
          <View style={styles.classColumn}>
            <Text style={styles.classText} numberOfLines={1}>
              {item.label}
            </Text>
          </View>
          <View style={styles.confidenceColumn}>
            <ConfidenceHeatmap value={item.value} />
          </View>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    backgroundColor: COLORS.background,
    borderRadius: 8,
    overflow: 'hidden',
  },
  tableHeader: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: '#F8FAFC',
  },
  tableRow: {
    flexDirection: 'row',
    paddingVertical: 14,
    paddingHorizontal: 16,
    alignItems: 'center',
    minHeight: 44,
  },
  tableRowBorder: {
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  classColumn: {
    flex: 1,
    minWidth: 100,
    paddingRight: 16,
  },
  confidenceColumn: {
    flex: 1,
    alignItems: 'flex-start',
    justifyContent: 'center',
    minWidth: 120,
  },
  headerText: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
  classText: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '500',
    letterSpacing: 0.1,
  },
  heatmapContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-start',
  },
  square: {
    borderWidth: 1,
    borderColor: COLORS.squareOutline,
  },
  squareFilled: {
    backgroundColor: COLORS.squareFilled,
    borderColor: COLORS.squareOutline,
  },
  squareUnfilled: {
    backgroundColor: '#F8FAFC',
    borderColor: COLORS.squareOutline,
  },
  squareMargin: {
    marginRight: 4,
  },
  squareSvg: {
    margin: 0,
  },
});
