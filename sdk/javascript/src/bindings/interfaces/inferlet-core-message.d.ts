/** @module Interface inferlet:core/message **/
/**
 * Sends a message to the remote user client
 */
export function send(message: string): void;
/**
 * Receives an incoming message from the remote user client
 */
export function receive(): ReceiveResult;
export function sendBlob(blob: Blob): void;
export function receiveBlob(): BlobResult;
/**
 * Publishes a message to a topic (broadcast to all subscribers)
 */
export function broadcast(topic: string, message: string): void;
/**
 * Subscribes to a topic and returns a subscription handle
 */
export function subscribe(topic: string): Subscription;
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Blob = import('./inferlet-core-common.js').Blob;
export type BlobResult = import('./inferlet-core-common.js').BlobResult;

export class ReceiveResult {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Pollable to check readiness
  */
  pollable(): Pollable;
  /**
  * Retrieves the message if available; None if not ready
  */
  get(): string | undefined;
}

export class Subscription {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Pollable to check for new messages on the topic
  */
  pollable(): Pollable;
  /**
  * Retrieves a new message from the topic, if available
  */
  get(): string | undefined;
  /**
  * Cancels the subscription
  */
  unsubscribe(): void;
}
