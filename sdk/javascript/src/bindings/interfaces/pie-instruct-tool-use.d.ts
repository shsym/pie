/** @module Interface pie:instruct/tool-use **/
/**
 * Register available tools (list of JSON schema strings)
 */
export function equip(model: Model, tools: Array<string>): Uint32Array;
/**
 * Provide a tool result (after a tool-call reply)
 */
export function answer(model: Model, name: string, value: string): Uint32Array;
/**
 * Create a decoder to detect tool calls in generated tokens
 */
export function createDecoder(model: Model): Decoder;
/**
 * Create a grammar matcher to force-generate tool calls
 */
export function createMatcher(model: Model, tools: Array<string>): Matcher;
export type Model = import('./pie-core-model.js').Model;
export type Matcher = import('./pie-core-inference.js').Matcher;
export type Error = import('./pie-core-types.js').Error;
export type Event = EventStart | EventCall;
/**
 * Tool call detected
 */
export interface EventStart {
  tag: 'start',
}
/**
 * Complete tool call: (name, arguments-json)
 */
export interface EventCall {
  tag: 'call',
  val: [string, string],
}

export class Decoder {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  feed(tokens: Uint32Array): Event;
  reset(): void;
}
