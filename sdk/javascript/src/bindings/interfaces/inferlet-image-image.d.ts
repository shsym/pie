/** @module Interface inferlet:image/image **/
/**
 * Embeds an image blob into model-compatible embeddings
 */
export function embedImage(queue: Queue, embPtrs: Uint32Array, imageBlob: Uint8Array, positionOffset: number): void;
/**
 * Computes the number of embeddings required for an image of given dimensions
 */
export function calculateEmbedSize(queue: Queue, imageWidth: number, imageHeight: number): number;
export type Queue = import('./inferlet-core-common.js').Queue;
export type Pointer = import('./inferlet-core-common.js').Pointer;
export type Blob = import('./inferlet-core-common.js').Blob;
export type BlobResult = import('./inferlet-core-common.js').BlobResult;
