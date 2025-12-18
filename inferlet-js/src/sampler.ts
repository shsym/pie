// Sampler types and configurations for text generation
// Mirrors the Rust Sampler enum from inferlet/src/sampler.rs

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

export class Sampler {
  private constructor(private readonly config: SamplerType) {}

  /**
   * Creates a greedy sampler (temperature = 0.0)
   * Always selects the most likely token
   */
  static greedy(): Sampler {
    return new Sampler({ type: 'Multinomial', temperature: 0.0 });
  }

  /**
   * Creates a multinomial sampler with the given temperature
   * Samples from the probability distribution with no filtering
   *
   * @param temperature - Controls randomness (0.0 = greedy, higher = more random)
   */
  static multinomial(temperature: number): Sampler {
    return new Sampler({ type: 'Multinomial', temperature });
  }

  /**
   * Creates a top-p (nucleus) sampler
   * Samples from the smallest set of tokens whose cumulative probability >= top_p
   *
   * @param temperature - Controls randomness (0.0 = greedy, higher = more random)
   * @param top_p - Cumulative probability threshold (typically 0.9-0.95)
   */
  static topP(temperature: number, top_p: number): Sampler {
    return new Sampler({ type: 'TopP', temperature, top_p });
  }

  /**
   * Creates a top-k sampler
   * Samples from the k most likely tokens
   *
   * @param temperature - Controls randomness (0.0 = greedy, higher = more random)
   * @param top_k - Number of top tokens to consider
   */
  static topK(temperature: number, top_k: number): Sampler {
    return new Sampler({ type: 'TopK', temperature, top_k });
  }

  /**
   * Creates a min-p sampler
   * Filters out tokens with probability < (min_p * max_probability)
   *
   * @param temperature - Controls randomness (0.0 = greedy, higher = more random)
   * @param min_p - Minimum probability threshold relative to the top token
   */
  static minP(temperature: number, min_p: number): Sampler {
    return new Sampler({ type: 'MinP', temperature, min_p });
  }

  /**
   * Creates a combined top-k and top-p sampler
   * First applies top-k filtering, then top-p filtering
   *
   * @param temperature - Controls randomness (0.0 = greedy, higher = more random)
   * @param top_k - Number of top tokens to consider
   * @param top_p - Cumulative probability threshold
   */
  static topKTopP(temperature: number, top_k: number, top_p: number): Sampler {
    return new Sampler({ type: 'TopKTopP', temperature, top_k, top_p });
  }

  /**
   * Creates a sampler optimized for reasoning tasks
   * Uses top_k_top_p(0.6, 20, 0.95)
   */
  static reasoning(): Sampler {
    return Sampler.topKTopP(0.6, 20, 0.95);
  }

  /**
   * Gets the sampler configuration
   */
  getConfig(): SamplerType {
    return this.config;
  }

  /**
   * Converts the sampler to a plain object for serialization
   */
  toJSON(): SamplerType {
    return this.config;
  }
}
