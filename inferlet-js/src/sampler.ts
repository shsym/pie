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
