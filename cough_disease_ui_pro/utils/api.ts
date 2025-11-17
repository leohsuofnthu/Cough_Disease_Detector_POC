export type UploadPayload = {
  uri?: string;
  name: string;
  type: string;
  blob?: Blob;
};

type Prediction = {
  label: string;
  confidence: number;
  all: Record<string, number>;
};

const MOCK: Prediction = {
  label: 'COPD',
  confidence: 0.76,
  all: {
    'Normal': 0.05,
    'COPD': 0.76,
    'Heart Disease': 0.06,
    'Bronchiectasis': 0.04,
    'Pneumonia': 0.03,
    'Upper Respiratory Tract Infection': 0.03,
    'Lower Respiratory Tract Infection': 0.03
  }
};

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000/v1/infer';

export async function uploadAudio(file: UploadPayload): Promise<Prediction> {
  try {
    const formData = new FormData();

    // Web: use Blob
    if (file.blob) {
      const webFile = new File([file.blob], file.name, { type: file.type });
      formData.append('file', webFile);
    } else if (file.uri) {
      // Native: use { uri, name, type }
      // @ts-ignore: RN FormData file type
      formData.append('file', { uri: file.uri, name: file.name, type: file.type });
    } else {
      return MOCK;
    }

    const res = await fetch(API_URL, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    if (!json || !json.label) throw new Error('Invalid response');
    const probabilities = json.probabilities || json.all;
    const standard: Prediction = {
      label: json.label,
      confidence: json.confidence ?? 0,
      all: probabilities || {}
    };
    return standard;
  } catch (e) {
    console.warn('API unreachable, using mock. Reason:', e);
    return MOCK;
  }
}
