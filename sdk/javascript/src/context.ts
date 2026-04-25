// Context, TokenStream, EventStream, and Decoder — wraps pie:core/context
// and pie:instruct/* WIT interfaces.

import {
  Context as _Context,
} from 'pie:core/context';
import {
  ForwardPass as _ForwardPass,
} from 'pie:core/inference';
import type { Sampler, Brle, Output } from 'pie:core/inference';
import type { Model as _Model } from 'pie:core/model';
import type { Adapter as _Adapter } from 'pie:core/adapter';

import * as _chat from 'pie:instruct/chat';
import * as _reasoning from 'pie:instruct/reasoning';
import * as _toolUse from 'pie:instruct/tool-use';

import { awaitFuture } from './_async.js';
import { Model } from './model.js';
import type { Adapter } from './adapter.js';
import type { Constraint, Schema } from './grammar.js';
import { StaticMaskConstraint, buildSchema } from './grammar.js';


// ─── Event Types ────────────────────────────────────────────────────────

/** A text chunk was generated. */
export interface EventText {
  readonly type: 'text';
  readonly text: string;
}

/** A reasoning/thinking text chunk. */
export interface EventThinking {
  readonly type: 'thinking';
  readonly text: string;
}

/** Reasoning is complete (full reasoning text). */
export interface EventThinkingDone {
  readonly type: 'thinking-done';
  readonly text: string;
}

/** A tool call was detected. */
export interface EventToolCallStart {
  readonly type: 'tool-call-start';
}

/** A complete tool call: name + JSON arguments. */
export interface EventToolCall {
  readonly type: 'tool-call';
  readonly name: string;
  readonly arguments: string;
}

/** Generation is done (full accumulated text). */
export interface EventDone {
  readonly type: 'done';
  readonly text: string;
}

export type Event =
  | EventText
  | EventThinking
  | EventThinkingDone
  | EventToolCallStart
  | EventToolCall
  | EventDone;

// ─── Decoder ────────────────────────────────────────────────────────────

/** Options for creating a `Decoder`. */
export interface DecoderOptions {
  /** Enable reasoning detection. Default: false. */
  reasoning?: boolean;
  /** Enable tool-use detection. Default: false. */
  toolUse?: boolean;
}

/**
 * Unified event decoder that multiplexes chat, reasoning, and tool-use
 * decoders from the instruct WIT interfaces.
 *
 * Event priority: reasoning > tool-use > chat.
 */
export class Decoder {
  private readonly _chatDecoder: _chat.Decoder;
  private readonly _reasoningDecoder?: _reasoning.Decoder;
  private readonly _toolUseDecoder?: _toolUse.Decoder;
  private _inReasoning = false;

  constructor(model: _Model, options: DecoderOptions = {}) {
    this._chatDecoder = _chat.createDecoder(model);
    if (options.reasoning) {
      this._reasoningDecoder = _reasoning.createDecoder(model);
    }
    if (options.toolUse) {
      this._toolUseDecoder = _toolUse.createDecoder(model);
    }
  }

  /**
   * Feed token IDs to all active decoders and return the highest-priority
   * event. Priority: reasoning > tool-use > chat.
   *
   * All sub-decoders are always fed to maintain their internal state.
   */
  feed(tokens: Uint32Array): Event {
    // Always feed the chat decoder
    const chatEvent = this._chatDecoder.feed(tokens);

    // Reasoning decoder (highest priority)
    if (this._reasoningDecoder) {
      const ev = this._reasoningDecoder.feed(tokens);
      switch (ev.tag) {
        case 'start':
          this._inReasoning = true;
          return { type: 'thinking', text: '' };
        case 'delta':
          return { type: 'thinking', text: ev.val };
        case 'complete':
          this._inReasoning = false;
          return { type: 'thinking-done', text: ev.val };
      }
    }

    // Tool-use decoder (second priority)
    if (this._toolUseDecoder) {
      const ev = this._toolUseDecoder.feed(tokens);
      switch (ev.tag) {
        case 'start':
          return { type: 'tool-call-start' };
        case 'call':
          return { type: 'tool-call', name: ev.val[0], arguments: ev.val[1] };
      }
    }

    // Chat decoder (always active, lowest priority)
    switch (chatEvent.tag) {
      case 'delta':
        return { type: 'text', text: chatEvent.val };
      case 'done':
        return { type: 'done', text: chatEvent.val };
      case 'interrupt':
        // Align with Python SDK: interrupts are mapped to Done("")
        return { type: 'done', text: '' };
    }
  }

  /** Reset all decoders to initial state. */
  reset(): void {
    this._chatDecoder.reset();
    this._reasoningDecoder?.reset();
    this._toolUseDecoder?.reset();
    this._inReasoning = false;
  }
}

/**
 * Reserve pages, run a forward pass, and commit pages.
 *
 * Uses `workingPageTokenCount()` and `committedPageCount()` to derive
 * sequence positions.
 *
 * @internal
 */
async function _reserveAndRun(
  ctxHandle: _Context,
  tokens: Uint32Array,
  sampler?: Sampler,
  logitMask?: Brle,
  adapter?: _Adapter,
): Promise<Output> {
  const numTokens = tokens.length;
  const pageSize = ctxHandle.tokensPerPage();
  const wpt = ctxHandle.workingPageTokenCount();
  const seqStart = ctxHandle.committedPageCount() * pageSize + wpt;

  // Reserve additional pages
  const currentWorking = ctxHandle.workingPageCount();
  const totalTokens = wpt + numTokens;
  const pagesNeeded = Math.ceil(totalTokens / pageSize);
  const additional = Math.max(0, pagesNeeded - currentWorking);
  if (additional > 0) {
    ctxHandle.reserveWorkingPages(additional);
  }

  // Build forward pass
  const pass = new _ForwardPass(ctxHandle.model());
  pass.context(ctxHandle);

  const positions = new Uint32Array(numTokens);
  for (let i = 0; i < numTokens; i++) {
    positions[i] = seqStart + i;
  }
  pass.inputTokens(tokens, positions);

  if (sampler !== undefined) {
    pass.sampler(new Uint32Array([numTokens - 1]), sampler);
  }
  if (logitMask !== undefined) {
    pass.logitMask(logitMask);
  }
  if (adapter !== undefined) {
    pass.adapter(adapter);
  }

  // Execute
  const output = await awaitFuture(pass.execute(), 'Forward pass failed');

  // Commit pages
  const newWpt = wpt + numTokens;
  const pagesToCommit = Math.floor(newWpt / pageSize);
  if (pagesToCommit > 0) {
    ctxHandle.commitWorkingPages(pagesToCommit);
  }

  return output;
}

// ─── TokenStream ────────────────────────────────────────────────────────

/** Options for `generate()`. */
export interface GenerateOptions {
  /** Sampler configuration. */
  sampler: Sampler;
  /** Maximum number of tokens to generate. */
  maxTokens: number;
  /** Stop token IDs. If omitted, uses the model's default stop tokens. */
  stopTokens?: Set<number> | number[];
  /** LoRA adapter to use. */
  adapter?: Adapter;
  /** Declarative {@link Schema} constraint or custom {@link Constraint}.
   *  Mutually exclusive with `logitMask`. */
  constrain?: Schema | Constraint;
  /** Static BRLE logit mask applied every step (for stateless constraints).
   *  Mutually exclusive with `constrain`. */
  logitMask?: Brle;
  /** Decode events instead of raw tokens. */
  decode?: DecoderOptions;
}

/**
 * An async iterator that generates tokens one step at a time.
 *
 * Each iteration returns a `Uint32Array` of accepted token IDs.
 *
 * ```js
 * for await (const tokens of stream) { ... }
 * const allText = await stream.text();
 * const allTokens = await stream.tokens();
 * ```
 */
export class TokenStream implements AsyncIterable<Uint32Array> {
  private readonly _ctx: Context;
  private readonly _sampler: Sampler;
  private readonly _maxTokens: number;
  private readonly _stopTokens: Set<number>;
  private readonly _adapter?: _Adapter;
  private readonly _constraint?: Constraint;
  private _constraintPending: Uint32Array = new Uint32Array();
  private _generated: number = 0;
  private _done: boolean = false;
  private _pendingTokens: Uint32Array;

  /** @internal */
  constructor(
    ctx: Context,
    options: GenerateOptions,
    constraint: Constraint | undefined,
    pendingTokens?: Uint32Array,
  ) {
    this._ctx = ctx;
    this._sampler = options.sampler;
    this._maxTokens = options.maxTokens;
    this._adapter = options.adapter?._handle;
    this._constraint = constraint;
    this._pendingTokens = pendingTokens ?? new Uint32Array();

    // Auto-detect stop tokens from model if not provided
    if (options.stopTokens !== undefined) {
      this._stopTokens = new Set(options.stopTokens);
    } else {
      this._stopTokens = new Set(_chat.stopTokens(ctx._handle.model()));
    }
  }

  /** Execute one forward pass and return generated tokens. */
  async step(): Promise<Uint32Array | undefined> {
    if (this._done || this._generated >= this._maxTokens) {
      this._done = true;
      return undefined;
    }

    if (this._pendingTokens.length === 0) {
      this._done = true;
      return undefined;
    }

    // Advance the constraint and compute this step's mask.
    let mask: Brle | undefined = undefined;
    if (this._constraint !== undefined) {
      const pending = this._constraintPending;
      this._constraintPending = new Uint32Array();
      const m = this._constraint.step(pending);
      if (m.length > 0) mask = m;
    }

    const output = await _reserveAndRun(
      this._ctx._handle,
      this._pendingTokens,
      this._sampler,
      mask,
      this._adapter,
    );

    // Process output
    if (output.tag === 'tokens') {
      let tokens = output.val;

      // Truncate at the first stop token (exclude it)
      for (let i = 0; i < tokens.length; i++) {
        if (this._stopTokens.has(tokens[i])) {
          tokens = tokens.subarray(0, i);
          this._done = true;
          break;
        }
      }

      if (tokens.length === 0) {
        this._done = true;
        return undefined;
      }

      this._generated += tokens.length;

      // Stash accepted tokens for the next step's constraint advance.
      if (this._constraint !== undefined) {
        const merged = new Uint32Array(this._constraintPending.length + tokens.length);
        merged.set(this._constraintPending);
        merged.set(tokens, this._constraintPending.length);
        this._constraintPending = merged;
      }

      // Seed pending_tokens with only the LAST generated token
      this._pendingTokens = new Uint32Array([tokens[tokens.length - 1]]);
      return tokens;
    }

    this._done = true;
    return undefined;
  }

  /** Consume the stream and return the decoded text. */
  async text(): Promise<string> {
    const tokenizer = this._ctx._handle.model().tokenizer();
    const chunks: string[] = [];
    for await (const tokens of this) {
      chunks.push(tokenizer.decode(tokens));
    }
    return chunks.join('');
  }

  /** Consume the stream and return all token batches. */
  async tokens(): Promise<Uint32Array[]> {
    const result: Uint32Array[] = [];
    for await (const tokens of this) {
      result.push(tokens);
    }
    return result;
  }

  async *[Symbol.asyncIterator](): AsyncIterableIterator<Uint32Array> {
    while (!this._done) {
      const tokens = await this.step();
      if (tokens !== undefined) {
        yield tokens;
      }
    }
  }
}

// ─── EventStream ────────────────────────────────────────────────────────

/** Listener signature per event tag — keeps the callback API strictly typed. */
type EventListeners = {
  'text':            (text: string) => void;
  'thinking':        (text: string) => void;
  'thinking-done':   (text: string) => void;
  'tool-call-start': () => void;
  'tool-call':       (name: string, args: string) => void;
  'done':            (text: string) => void;
};
type EventTag = keyof EventListeners;
type ListenerStore = { [K in EventTag]?: EventListeners[K][] };

/**
 * An async iterator that yields decoded `Event` objects.
 *
 * Supports both `for await...of` iteration and callback-based `.on().run()`,
 * plus `.text()` / `.events()` shorthands for one-shot consumption.
 *
 * ```typescript
 * // One-shot:
 * const text = await stream.text();
 * const events = await stream.events();
 *
 * // Iterator style:
 * for await (const event of stream) { ... }
 *
 * // Callback style:
 * await stream
 *   .on('text', text => session.send(text))
 *   .on('done', text => console.log('finished:', text))
 *   .run();
 * ```
 */
export class EventStream implements AsyncIterable<Event> {
  private readonly _stream: TokenStream;
  private readonly _decoder: Decoder;
  private readonly _listeners: ListenerStore = {};

  /** @internal */
  constructor(stream: TokenStream, decoder: Decoder) {
    this._stream = stream;
    this._decoder = decoder;
  }

  /** Register a callback for a specific event type. Returns `this` for chaining. */
  on<K extends EventTag>(type: K, cb: EventListeners[K]): this {
    const list = (this._listeners[type] ??= []) as EventListeners[K][];
    list.push(cb);
    return this;
  }

  /** Drive the stream to completion, invoking registered callbacks. */
  async run(): Promise<void> {
    for await (const event of this) {
      switch (event.type) {
        case 'text':
        case 'thinking':
        case 'thinking-done':
        case 'done':
          this._listeners[event.type]?.forEach(cb => cb(event.text));
          break;
        case 'tool-call-start':
          this._listeners[event.type]?.forEach(cb => cb());
          break;
        case 'tool-call':
          this._listeners[event.type]?.forEach(cb => cb(event.name, event.arguments));
          break;
      }
    }
  }

  /** Consume the stream and return the concatenated text from `text` / `done` events. */
  async text(): Promise<string> {
    const parts: string[] = [];
    for await (const event of this) {
      if (event.type === 'text') {
        parts.push(event.text);
      } else if (event.type === 'done') {
        return parts.join('') || event.text;
      }
    }
    return parts.join('');
  }

  /** Consume the stream and return every event as a flat array. */
  async events(): Promise<Event[]> {
    const events: Event[] = [];
    for await (const event of this) events.push(event);
    return events;
  }

  async *[Symbol.asyncIterator](): AsyncIterableIterator<Event> {
    for await (const tokens of this._stream) {
      yield this._decoder.feed(tokens);
    }
  }
}

// ─── Context ────────────────────────────────────────────────────────────

/**
 * A conversation / generation context.
 *
 * Wraps the `pie:core/context.Context` WIT resource and provides
 * high-level methods for chat formatting, generation, and forking.
 *
 * ```js
 * const ctx = Context.create(model);
 * ctx.system('You are helpful.');
 * ctx.user('Hello!');
 * const text = ctx.generateText({ sampler: Sampler.topP(0.6, 0.95), maxTokens: 128 });
 * ```
 */
export class Context implements Disposable {
  /** @internal */
  readonly _handle: _Context;
  /** @internal Wrapped model — needed to build schema constraints. */
  readonly _model: Model;
  /** SDK-local pending tokens */
  private _pendingTokens: Uint32Array = new Uint32Array();

  private constructor(handle: _Context, model: Model) {
    this._handle = handle;
    this._model = model;
  }

  /** Create a new anonymous context. Name is NOT needed. */
  static create(model: Model): Context {
    return new Context(_Context.create(model._handle), model);
  }

  /** Open a saved (named) context. */
  static open(model: Model, name: string): Context | undefined {
    const handle = _Context.open(model._handle, name);
    return handle !== undefined ? new Context(handle, model) : undefined;
  }

  /** Look up an existing context by name. Alias for open(). */
  static lookup(model: Model, name: string): Context | undefined {
    return Context.open(model, name);
  }

  /** Fork this context into a new anonymous one. */
  fork(): Context {
    return new Context(this._handle.fork(), this._model);
  }

  /** Save this context with a name, making it persistent. */
  save(name: string): void {
    this._handle.save(name);
  }

  /** Destroy the context and release its KV resources. */
  destroy(): void {
    this._handle.destroy();
  }

  /** Disposable protocol — calls `destroy()`. */
  [Symbol.dispose](): void {
    this.destroy();
  }

  // ── Text-level buffer fill ──
  //
  // Filler methods return `this` so they can be chained:
  //
  //     ctx.system("...").user("...").cue()

  /** Fill the context buffer with text (encodes via the model's tokenizer). */
  fill(text: string): this {
    const tokenizer = this._handle.model().tokenizer();
    const tokens = tokenizer.encode(text);
    this._appendPending(tokens);
    return this;
  }

  /** Fill the context buffer with raw token IDs. */
  fillTokens(tokens: Uint32Array): this {
    this._appendPending(tokens);
    return this;
  }

  // ── Chat formatting (pie:instruct/chat) ──

  /** Fill a system message. */
  system(message: string): this {
    this._appendPending(_chat.system(this._handle.model(), message));
    return this;
  }

  /** Fill a user message. */
  user(message: string): this {
    this._appendPending(_chat.user(this._handle.model(), message));
    return this;
  }

  /** Fill an assistant message (for history replay). */
  assistant(message: string): this {
    this._appendPending(_chat.assistant(this._handle.model(), message));
    return this;
  }

  /** Cue the model to generate (fills generation header). */
  cue(): this {
    this._appendPending(_chat.cue(this._handle.model()));
    return this;
  }

  /** Seal the current turn (inserts stop token). */
  seal(): this {
    this._appendPending(_chat.seal(this._handle.model()));
    return this;
  }

  /** Returns the stop token IDs for this context's model. */
  stopTokens(): Uint32Array {
    return _chat.stopTokens(this._handle.model());
  }

  // ── Tool use (pie:instruct/tool-use) ──

  /** Register available tools.
   *
   * Each tool may be either a JSON-schema string or a structured object
   * (auto-stringified). */
  equipTools(tools: Array<string | object>): this {
    const strs = tools.map(t => typeof t === 'string' ? t : JSON.stringify(t));
    this._appendPending(_toolUse.equip(this._handle.model(), strs));
    return this;
  }

  /** Provide a tool result after a tool call.
   *
   * `value` may be a string or a JSON-serializable object (auto-stringified). */
  answerTool(name: string, value: string | object): this {
    const s = typeof value === 'string' ? value : JSON.stringify(value);
    this._appendPending(_toolUse.answer(this._handle.model(), name, s));
    return this;
  }

  // ── Low-level context operations ──

  /** Number of tokens per KV page. */
  tokensPerPage(): number {
    return this._handle.tokensPerPage();
  }

  /** Returns the wrapped {@link Model} this context was created with. */
  model(): Model {
    return this._model;
  }

  /** Flush pending tokens by executing a fill forward pass (no sampling).
   *  Keeps the last token as seed for the next generation step. */
  async flush(): Promise<void> {
    if (this._pendingTokens.length <= 1) return;

    // Flush all but the last token
    const tokensToFlush = this._pendingTokens.slice(0, -1);
    const lastToken = this._pendingTokens[this._pendingTokens.length - 1];

    await _reserveAndRun(this._handle, tokensToFlush);

    // Keep last token as seed
    this._pendingTokens = new Uint32Array([lastToken]);
  }

  // ── Generation ──

  /**
   * Start token generation, returning a `TokenStream` or `EventStream`.
   *
   * **Auto-flush behavior:** every call to `generate()` first invokes
   * `flush()` and then `cue()`, so any pending instruct fills land in the
   * KV cache and the model is cued to begin generating. There is no
   * `autoFlush: false` escape — if you need to inspect the buffer before
   * generation, call `flush()` yourself, then `generate()` will be a no-op
   * flush.
   *
   * Stop tokens default to the model's chat stop tokens.
   */
  async generate(options: GenerateOptions & { decode: DecoderOptions }): Promise<EventStream>;
  async generate(options: GenerateOptions): Promise<TokenStream>;
  async generate(options: GenerateOptions): Promise<TokenStream | EventStream> {
    if (options.constrain !== undefined && options.logitMask !== undefined) {
      throw new Error(
        'constrain and logitMask are mutually exclusive — wrap the static ' +
          'mask in a Constraint implementation if you need to combine both',
      );
    }

    await this.flush();
    this.cue();

    let constraint: Constraint | undefined = undefined;
    if (options.constrain !== undefined) {
      const c = options.constrain;
      // Discriminate on `step` — Constraint has it, Schema doesn't. Using
      // `kind` as the discriminator would misroute custom Constraints that
      // happen to carry a `kind` field.
      constraint = (typeof (c as Constraint).step === 'function')
        ? (c as Constraint)
        : buildSchema(c as Schema, this._model);
    } else if (options.logitMask !== undefined) {
      constraint = new StaticMaskConstraint(options.logitMask);
    }

    const stream = new TokenStream(this, options, constraint, this._pendingTokens);
    this._pendingTokens = new Uint32Array();

    if (options.decode) {
      const decoder = new Decoder(this._handle.model(), options.decode);
      return new EventStream(stream, decoder);
    }

    return stream;
  }

  /**
   * Generate and return the full response text.
   *
   * Convenience method that calls `generate()` + `text()` internally.
   *
   * ```js
   * const reply = ctx.generateText({
   *   sampler: Sampler.topP(0.6, 0.95),
   *   maxTokens: 256,
   * });
   * ```
   */
  async generateText(options: Omit<GenerateOptions, 'decode'>): Promise<string> {
    return (await this.generate(options)).text();
  }

  /**
   * Generate JSON-constrained output and parse it.
   *
   * If `schema` is provided, output is constrained to JSON valid against
   * that JSON Schema. Otherwise, output is constrained to any valid JSON.
   *
   * The optional `parse` hook lets you plug in a runtime validator (Zod,
   * arktype, etc.). When provided, the parsed JSON is passed through it
   * and the result is returned typed as `T`. Without it, the result is
   * returned as `unknown`.
   *
   * ```typescript
   * // untyped:
   * const data = await ctx.generateJson({
   *   sampler: Sampler.argmax(),
   *   maxTokens: 512,
   *   schema: PERSON_SCHEMA,
   * });
   *
   * // typed via Zod:
   * import { z } from 'zod';
   * import { zodToJsonSchema } from 'zod-to-json-schema';
   *
   * const Person = z.object({ name: z.string(), age: z.number() });
   * const person = await ctx.generateJson({
   *   sampler: Sampler.argmax(),
   *   maxTokens: 512,
   *   schema: JSON.stringify(zodToJsonSchema(Person)),
   *   parse: Person.parse,
   * });
   * // person: z.infer<typeof Person>
   * ```
   */
  async generateJson<T = unknown>(
    options: Omit<GenerateOptions, 'decode' | 'constrain' | 'logitMask'> & {
      /** JSON Schema string. Omit to allow any valid JSON. */
      schema?: string;
      /** Runtime validator. Receives the parsed JSON, returns the typed value. */
      parse?: (value: unknown) => T;
    },
  ): Promise<T> {
    const { schema, parse, ...rest } = options;
    const constrain: Schema = schema !== undefined
      ? { kind: 'json-schema', value: schema }
      : { kind: 'json' };
    const text = await this.generateText({ ...rest, constrain });
    const value = JSON.parse(text);
    return parse ? parse(value) : (value as T);
  }

  /** @internal Append tokens to the pending buffer. */
  private _appendPending(tokens: Uint32Array): void {
    if (tokens.length === 0) return;
    const merged = new Uint32Array(this._pendingTokens.length + tokens.length);
    merged.set(this._pendingTokens);
    merged.set(tokens, this._pendingTokens.length);
    this._pendingTokens = merged;
  }
}
