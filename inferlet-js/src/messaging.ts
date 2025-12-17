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
 *
 * IMPLEMENTATION NOTE - Workaround for componentize-js limitation:
 *
 * The WIT interface (inferlet:core/message) defines a proper send() function:
 *   send: func(message: string);
 *
 * However, this function does not work correctly when JavaScript inferlets are
 * compiled with componentize-js. The exact root cause is unknown, but the WIT
 * binding for message.send() fails to deliver messages to the client.
 *
 * CURRENT WORKAROUND:
 * We use console.log() instead, which works because:
 * 1. componentize-js compiles console.log to WASI stdout (wasi:cli/stdout)
 * 2. The Pie runtime captures WASI stdout from the WebAssembly component
 * 3. Stdout is delivered to clients via the StreamingOutput::Stdout message type
 * 4. The server dispatches this as an InstanceEvent::StreamingOutput event
 *
 * This approach is functionally equivalent for text output, but bypasses the
 * proper message.send() channel defined in the WIT interface.
 *
 * FUTURE WORK:
 * - Investigate why componentize-js bindings for message.send() don't work
 * - Test if this is a general componentize-js issue or specific to our WIT setup
 * - Consider filing an issue with componentize-js if this is a binding bug
 * - Once resolved, replace console.log with: message.send(msg);
 */
export function send(msg: string): void {
  // Workaround: Use console.log which works via WASI stdout capture
  // See documentation comment above for full explanation
  console.log(msg);
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
  return new (Blob as any)(result);
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
