// Sampler types and configurations for text generation
// Used for backend integration with the Pie runtime

/**
 * Internal sampler type discriminated union (used by the runtime)
 */
export type SamplerType =
  | { type: 'Custom'; temperature: number; sampler: object }
  | { type: 'Multinomial'; temperature: number }
  | { type: 'TopP'; temperature: number; top_p: number }
  | { type: 'TopK'; temperature: number; top_k: number }
  | { type: 'MinP'; temperature: number; min_p: number }
  | { type: 'TopKTopP'; temperature: number; top_k: number; top_p: number };

/**
 * User-friendly sampling configuration object.
 * All properties are optional - unspecified values use defaults.
 *
 * @example
 * { topP: 0.95, temperature: 0.6 }
 * { topK: 50, temperature: 0.8 }
 * { topK: 40, topP: 0.92, temperature: 0.9 }
 */
export interface SamplingConfig {
  /** Controls randomness (0.0 = greedy, higher = more random). Default: 1.0 */
  temperature?: number;
  /** Top-p (nucleus) sampling threshold (typically 0.9-0.95) */
  topP?: number;
  /** Top-k sampling - number of top tokens to consider */
  topK?: number;
  /** Min-p sampling - minimum probability relative to top token */
  minP?: number;
}

/**
 * Sampler presets for common use cases
 */
export const Sampler = {
  /**
   * Greedy sampling - always picks the most probable token.
   */
  greedy(): SamplingConfig {
    return { temperature: 0 };
  },

  /**
   * Balanced default settings for general text generation.
   */
  default(): SamplingConfig {
    return { temperature: 0.7, topP: 0.95 };
  },

  /**
   * Creative/diverse sampling with higher temperature.
   */
  creative(): SamplingConfig {
    return { temperature: 1.0, topP: 0.95 };
  },

  /**
   * Optimized for reasoning tasks.
   * Uses conservative sampling settings (topK=20, topP=0.95, temperature=0.6)
   * to produce coherent, focused outputs suitable for chain-of-thought reasoning.
   */
  reasoning(): SamplingConfig {
    return { temperature: 0.6, topK: 20, topP: 0.95 };
  },

  /**
   * Top-p (nucleus) sampling preset.
   */
  topP(p: number = 0.95, temperature: number = 0.7): SamplingConfig {
    return { topP: p, temperature };
  },

  /**
   * Top-k sampling preset.
   */
  topK(k: number = 40, temperature: number = 0.7): SamplingConfig {
    return { topK: k, temperature };
  },

  /**
   * Min-p sampling preset.
   * Filters out tokens with probability below minP * max_prob.
   */
  minP(minP: number = 0.1, temperature: number = 0.7): SamplingConfig {
    return { minP, temperature };
  },
} as const;

/**
 * Convert a SamplingConfig to the internal SamplerType for backend integration
 */
export function toSamplerType(config: SamplingConfig): SamplerType {
  const temp = config.temperature ?? 1.0;

  // Determine which sampler type to use based on provided options
  if (config.topK !== undefined && config.topP !== undefined) {
    return { type: 'TopKTopP', temperature: temp, top_k: config.topK, top_p: config.topP };
  }
  if (config.topP !== undefined) {
    return { type: 'TopP', temperature: temp, top_p: config.topP };
  }
  if (config.topK !== undefined) {
    return { type: 'TopK', temperature: temp, top_k: config.topK };
  }
  if (config.minP !== undefined) {
    return { type: 'MinP', temperature: temp, min_p: config.minP };
  }

  // Default: multinomial (greedy if temp is 0)
  return { type: 'Multinomial', temperature: temp };
}
