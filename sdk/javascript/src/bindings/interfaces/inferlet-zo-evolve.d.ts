/** @module Interface inferlet:zo/evolve **/
export function setAdapterSeed(pass: ForwardPass, seed: bigint): void;
export function initializeAdapter(queue: Queue, adapterPtr: Pointer, rank: number, alpha: number, populationSize: number, muFraction: number, initialSigma: number): void;
export function updateAdapter(queue: Queue, adapterPtr: Pointer, scores: Float32Array, seeds: BigInt64Array, maxSigma: number): void;
export type Queue = import('./inferlet-core-common.js').Queue;
export type Pointer = import('./inferlet-core-common.js').Pointer;
export type Blob = import('./inferlet-core-common.js').Blob;
export type BlobResult = import('./inferlet-core-common.js').BlobResult;
export type ForwardPass = import('./inferlet-core-forward.js').ForwardPass;
