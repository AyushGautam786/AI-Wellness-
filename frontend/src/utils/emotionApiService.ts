/**
 * Emotion Detection API Service
 * Handles communication with the FastAPI emotion detection service
 */

export interface EmotionPrediction {
  predicted_emotion: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  success: boolean;
  error?: string;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: number;
}

class EmotionApiService {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = 'http://127.0.0.1:8000', timeout: number = 10000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  /**
   * Check if the emotion detection API is available
   */
  async checkHealth(): Promise<ApiResponse<{ status: string; model_loaded: boolean }>> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        return {
          error: `API health check failed: ${response.status} ${response.statusText}`,
          status: response.status,
        };
      }

      const data = await response.json();
      return {
        data,
        status: response.status,
      };
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          return {
            error: 'Request timeout - emotion API is not responding',
            status: 408,
          };
        }
        return {
          error: `Network error: ${error.message}`,
          status: 0,
        };
      }
      return {
        error: 'Unknown error occurred',
        status: 0,
      };
    }
  }

  /**
   * Predict emotion from text
   */
  async predictEmotion(text: string): Promise<ApiResponse<EmotionPrediction>> {
    try {
      if (!text || !text.trim()) {
        return {
          error: 'Text input cannot be empty',
          status: 400,
        };
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text.trim() }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return {
          error: errorData.detail || `API request failed: ${response.status} ${response.statusText}`,
          status: response.status,
        };
      }

      const data = await response.json();
      return {
        data,
        status: response.status,
      };
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          return {
            error: 'Request timeout - emotion API is not responding',
            status: 408,
          };
        }
        return {
          error: `Network error: ${error.message}`,
          status: 0,
        };
      }
      return {
        error: 'Unknown error occurred',
        status: 0,
      };
    }
  }

  /**
   * Get list of supported emotions
   */
  async getSupportedEmotions(): Promise<ApiResponse<{ emotions: string[]; total_count: number }>> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.baseUrl}/emotions`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        return {
          error: `Failed to get supported emotions: ${response.status} ${response.statusText}`,
          status: response.status,
        };
      }

      const data = await response.json();
      return {
        data,
        status: response.status,
      };
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          return {
            error: 'Request timeout - emotion API is not responding',
            status: 408,
          };
        }
        return {
          error: `Network error: ${error.message}`,
          status: 0,
        };
      }
      return {
        error: 'Unknown error occurred',
        status: 0,
      };
    }
  }

  /**
   * Set a new base URL for the API
   */
  setBaseUrl(url: string): void {
    this.baseUrl = url;
  }

  /**
   * Set request timeout
   */
  setTimeout(timeout: number): void {
    this.timeout = timeout;
  }
}

// Create a singleton instance
export const emotionApiService = new EmotionApiService();

// Helper functions for common use cases
export const emotionApi = {
  /**
   * Quick emotion prediction with error handling
   */
  async predictEmotion(text: string): Promise<EmotionPrediction | null> {
    const response = await emotionApiService.predictEmotion(text);
    if (response.error) {
      console.error('Emotion prediction error:', response.error);
      return null;
    }
    return response.data || null;
  },

  /**
   * Check if API is available
   */
  async isAvailable(): Promise<boolean> {
    const response = await emotionApiService.checkHealth();
    return !response.error && response.data?.model_loaded === true;
  },

  /**
   * Get emotion with fallback
   */
  async getEmotionOrFallback(text: string, fallback: string = 'neutral'): Promise<string> {
    const prediction = await this.predictEmotion(text);
    return prediction?.predicted_emotion || fallback;
  },

  /**
   * Get emotion confidence
   */
  async getEmotionConfidence(text: string): Promise<number> {
    const prediction = await this.predictEmotion(text);
    return prediction?.confidence || 0;
  },
};

export default emotionApiService;