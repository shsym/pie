// Messaging functions for communicating with the remote user client.
// Mirrors the Rust messaging functions from inferlet/src/lib.rs

import * as message from 'inferlet:core/message';
import type { Pollable } from 'wasi:io/poll';

/**
 * Represents a binary blob that can be sent/received.
 */
export class Blob {
  private inner: message.Blob;

  private constructor(inner: message.Blob) {
    this.inner = inner;
  }

  /**
   * Create a new Blob from binary data
   */
  static new(data: Uint8Array): Blob {
    return new Blob(new message.Blob(data));
  }

  /**
   * Create a Blob from a WIT message.Blob resource (internal use)
   * @internal
   */
  static fromWit(inner: message.Blob): Blob {
    return new Blob(inner);
  }

  /**
   * Get the inner blob resource (for internal use)
   */
  getInner(): message.Blob {
    return this.inner;
  }

  /**
   * Get the data from the blob
   */
  getData(): Uint8Array {
    // Note: The actual method depends on the WIT binding
    // This may need adjustment based on the actual binding API
    return this.inner.data;
  }
}

/**
 * Sends a message to the remote user client.
 * @param msg The message to send
 */
export function send(msg: string): void {
  message.send(msg);
}

/**
 * Receives an incoming message from the remote user client.
 * This is an asynchronous operation.
 * @returns A promise that resolves to the received message
 */
export async function receive(): Promise<string> {
  const future = message.receive();
  const pollable: Pollable = future.pollable();

  // Use pollable.block() which is the WASI way to wait
  pollable.block();

  const result = future.get();
  if (result === undefined) {
    throw new Error('receive() returned undefined');
  }
  return result;
}

/**
 * Sends a blob to the remote user client.
 * @param blob The blob to send
 */
export function sendBlob(blob: Blob): void {
  message.sendBlob(blob.getInner());
}

/**
 * Receives an incoming blob from the remote user client.
 * This is an asynchronous operation.
 * @returns A promise that resolves to the received blob
 */
export async function receiveBlob(): Promise<Blob> {
  const future = message.receiveBlob();
  const pollable: Pollable = future.pollable();

  // Use pollable.block() which is the WASI way to wait
  pollable.block();

  const result = future.get();
  if (result === undefined) {
    throw new Error('receiveBlob() returned undefined');
  }
  return Blob.fromWit(result);
}

/**
 * Publishes a message to a topic, broadcasting it to all subscribers.
 * @param topic The topic to broadcast to
 * @param msg The message to broadcast
 */
export function broadcast(topic: string, msg: string): void {
  message.broadcast(topic, msg);
}

/**
 * Subscribes to a topic and waits for a message.
 * This is an asynchronous operation.
 * @param topic The topic to subscribe to
 * @returns A promise that resolves to the received message
 */
export async function subscribe(topic: string): Promise<string> {
  const future = message.subscribe(topic);
  const pollable: Pollable = future.pollable();

  // Use pollable.block() which is the WASI way to wait
  pollable.block();

  const result = future.get();
  if (result === undefined) {
    throw new Error('subscribe() returned undefined');
  }
  return result;
}
