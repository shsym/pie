/** @module Interface inferlet:adapter/common **/
export function setAdapter(pass: ForwardPass, adapterPtr: Pointer): void;
export function downloadAdapter(queue: Queue, adapterPtr: Pointer, name: string): BlobResult;
export function uploadAdapter(queue: Queue, adapterPtr: Pointer, name: string, blob: Blob): void;
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Queue = import('./inferlet-core-common.js').Queue;
export type Pointer = import('./inferlet-core-common.js').Pointer;
export type Blob = import('./inferlet-core-common.js').Blob;
export type BlobResult = import('./inferlet-core-common.js').BlobResult;
export type ForwardPass = import('./inferlet-core-forward.js').ForwardPass;
