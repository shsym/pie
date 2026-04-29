// Sampler configuration — maps to pie:core/inference.Sampler WIT variant.

import type { Sampler as WitSampler } from 'pie:core/inference';

/**
 * Sampler constructors for controlling token selection during generation.
 *
 * Each method returns a WIT-compatible discriminated union value
 * that can be passed to `ForwardPass.sampler()`.
 */
export const Sampler = {
  /** Greedy / argmax decoding (temperature=0, top-k=1).
   *
   * Recommended default for grammar-constrained generation. */
  greedy(): WitSampler {
    return { tag: 'multinomial', val: [0.0, 1] };
  },

  /** Argmax sampling. Alias for {@link Sampler.greedy}. */
  argmax(): WitSampler {
    return { tag: 'multinomial', val: [0.0, 1] };
  },

  /** Multinomial sampling with temperature and top-k. */
  multinomial(temperature: number, topK: number): WitSampler {
    return { tag: 'multinomial', val: [temperature, topK] };
  },

  /** Top-k sampling. */
  topK(temperature: number, topK: number): WitSampler {
    return { tag: 'top-k', val: [temperature, topK] };
  },

  /** Top-p (nucleus) sampling. */
  topP(temperature: number, topP: number): WitSampler {
    return { tag: 'top-p', val: [temperature, topP] };
  },

  /** Min-p sampling. */
  minP(temperature: number, minP: number): WitSampler {
    return { tag: 'min-p', val: [temperature, minP] };
  },

  /** Combined top-k + top-p sampling. */
  topKTopP(temperature: number, topK: number, topP: number): WitSampler {
    return { tag: 'top-k-top-p', val: [temperature, topK, topP] };
  },

  /** Embedding extraction (no sampling). */
  embedding(): WitSampler {
    return { tag: 'embedding' };
  },

  /** Distribution output mode.
   *
   * Returns the top-`topK` token IDs with their probabilities instead of a
   * sampled token. Useful for tree search, best-of-n, or external samplers.
   * `topK = 0` returns the full distribution. */
  distribution(temperature: number, topK: number): WitSampler {
    return { tag: 'dist', val: [temperature, topK] };
  },

  /** Raw logits output mode.
   *
   * Returns the model's pre-softmax, untemperatured logits as a packed
   * little-endian f32 byte buffer (length = `vocab_size * 4`) per requested
   * position. Decode via `new Float32Array(buf.buffer, buf.byteOffset,
   * buf.byteLength / 4)` (or similar). */
  rawLogits(): WitSampler {
    return { tag: 'raw-logits' };
  },

  /** Single-label logprob at this position.
   *
   * Returns `log p(tokenId | context)` via `Output.logprobs` (length-1 inner
   * list per slot). Computed via log-softmax with no temperature scaling. */
  logprob(tokenId: number): WitSampler {
    return { tag: 'logprob', val: tokenId };
  },

  /** Multi-label logprobs at one position.
   *
   * Returns `log p(t | context)` for each `t` in `tokenIds`, in order, via
   * `Output.logprobs` (length-K inner list per slot). */
  logprobs(tokenIds: number[]): WitSampler {
    return { tag: 'logprobs', val: tokenIds };
  },

  /** Shannon entropy `H(p) = -sum(p log p)` of the unscaled distribution at
   * this position, via `Output.entropies`. */
  entropy(): WitSampler {
    return { tag: 'entropy' };
  },
} as const;

/** Re-export the WIT sampler type for external use. */
export type { WitSampler as SamplerType };
