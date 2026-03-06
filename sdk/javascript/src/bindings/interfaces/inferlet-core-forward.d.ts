/** @module Interface inferlet:core/forward **/
export function createForwardPass(queue: Queue): ForwardPass;
export function attentionMask(pass: ForwardPass, mask: Array<Uint32Array>): void;
export function kvCache(pass: ForwardPass, kvPagePtrs: Uint32Array, lastKvPageLen: number): void;
export function inputEmbeddings(pass: ForwardPass, embPtrs: Uint32Array, positions: Uint32Array): void;
export function inputTokens(pass: ForwardPass, inputTokens: Uint32Array, positions: Uint32Array): void;
export function outputEmbeddings(pass: ForwardPass, embPtrs: Uint32Array, indices: Uint32Array): void;
export function outputDistributions(pass: ForwardPass, indices: Uint32Array, temperature: number, topK: number | undefined): void;
export function outputTokens(pass: ForwardPass, indices: Uint32Array, temperature: number): void;
export function outputTokensTopK(pass: ForwardPass, indices: Uint32Array, temperature: number, topK: number): void;
export function outputTokensTopP(pass: ForwardPass, indices: Uint32Array, temperature: number, topP: number): void;
export function outputTokensMinP(pass: ForwardPass, indices: Uint32Array, temperature: number, minP: number): void;
export function outputTokensTopKTopP(pass: ForwardPass, indices: Uint32Array, temperature: number, topK: number, topP: number): void;
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Queue = import('./inferlet-core-common.js').Queue;
export type Pointer = import('./inferlet-core-common.js').Pointer;

export class ForwardPass {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  execute(): ForwardPassResult | undefined;
}

export class ForwardPassResult {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Returns a pollable object to check when the result is ready
  */
  pollable(): Pollable;
  /**
  * Retrieves the result if ready; None if still pending
  * Each tuple: (token IDs, associated probabilities)
  */
  getDistributions(): Array<[Uint32Array, Float32Array]> | undefined;
  getTokens(): Uint32Array | undefined;
}
