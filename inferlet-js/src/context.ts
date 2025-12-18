// Context class for managing conversation state and generation.
// Mirrors the Rust Context from inferlet/src/context.rs

import { Model, Queue } from './model.js';
import { Tokenizer } from './tokenizer.js';
import { ChatFormatter } from './chat.js';
import { Brle } from './brle.js';
import { KvPage, ForwardPass, type Distribution } from './forward.js';
import { Sampler, type SamplerType, type SamplingConfig } from './sampler.js';
import { MaxLen, AnyEndsWith, type StopCondition } from './stop-condition.js';
import { KvPageManager } from './kv-page-manager.js';
import { ImmutableArray } from './immutable-array.js';


/**
 * Message role for chat messages
 */
export type MessageRole = 'system' | 'user' | 'assistant';

/**
 * Chat message format
 */
export interface ChatMessage {
  role: MessageRole;
  content: string;
}

/**
 * Stop condition configuration
 */
export interface StopConfig {
  /** Maximum number of tokens to generate */
  maxTokens?: number;
  /** Stop sequences as tokenized arrays */
  sequences?: number[][];
}

/**
 * Options for the generate() method
 */
export interface GenerateOptions {
  /** Sampling configuration - either a config object or a Sampler instance */
  sampling: SamplingConfig | Sampler;
  /** Stop conditions */
  stop: StopConfig;
}

/**
 * Context manages the state for text generation, including:
 * - Token history and pending tokens
 * - Attention masks
 * - KV cache pages
 * - Chat formatting
 */
export class Context {
  queue: Queue;
  model: Model;
  tokenizer: Tokenizer;
  formatter: ChatFormatter;

  private _tokenIds: ImmutableArray<number> = ImmutableArray.empty();
  tokenIdsPending: number[] = [];

  tokenMaskPending: Brle[] = [];
  tokenMaskCurrent: Brle;

  private _positionIds: ImmutableArray<number> = ImmutableArray.empty();

  private kvPageManager: KvPageManager;
  kvPageSize: number;

  adapterPtr?: number;
  adapterRandomSeed?: number;

  beginOfSequence: boolean = true;

  constructor(model: Model) {
    this.queue = model.createQueue();
    this.kvPageSize = model.kvPageSize;
    this.kvPageManager = new KvPageManager(this.queue, this.kvPageSize);
    this.tokenizer = model.tokenizer;
    this.model = model;
    this.formatter = new ChatFormatter();
    this.tokenMaskCurrent = Brle.new(0);
  }

  /**
   * The committed token IDs (as regular array for backward compatibility)
   */
  get tokenIds(): number[] {
    return this._tokenIds.toArray();
  }

  set tokenIds(value: number[]) {
    this._tokenIds = ImmutableArray.from(value);
  }

  /**
   * The position IDs (as regular array for backward compatibility)
   */
  get positionIds(): number[] {
    return this._positionIds.toArray();
  }

  set positionIds(value: number[]) {
    this._positionIds = ImmutableArray.from(value);
  }

  /**
   * The KV pages array (for backward compatibility)
   */
  get kvPages(): KvPage[] {
    return this.kvPageManager.allPages;
  }

  /**
   * The last page length
   */
  get kvPageLastLen(): number {
    return this.kvPageManager.lastPageLen;
  }

  /**
   * The unique IDs of the KV cache pages currently in use
   */
  get kvPagePtrs(): number[] {
    return this.kvPageManager.ptrs;
  }

  /**
   * Creates a new Context from previously exported and now imported KV pages.
   * This is used to restore a context's state from a cache.
   */
  static fromImportedState(
    model: Model,
    kvPages: KvPage[],
    prefixTokens: number[],
    kvPageLastLen: number
  ): Context {
    const ctx = new Context(model);
    const kvPageSize = model.kvPageSize;

    // Verify the token count matches the KV page state
    const expectedTokens = (kvPages.length - 1) * kvPageSize + kvPageLastLen;
    if (prefixTokens.length !== expectedTokens) {
      throw new Error(
        `Token count mismatch: expected ${expectedTokens}, got ${prefixTokens.length}`
      );
    }

    ctx._tokenIds = ImmutableArray.from([...prefixTokens]);
    ctx._positionIds = ImmutableArray.from(Array.from({ length: prefixTokens.length }, (_, i) => i));

    // Import pages into manager
    ctx.kvPageManager.importPages(kvPages, kvPageLastLen);

    ctx.tokenMaskCurrent = Brle.new(prefixTokens.length);
    ctx.beginOfSequence = false;

    return ctx;
  }

  /**
   * The text representation of all tokens (computed on access)
   */
  get text(): string {
    return this.tokenizer.detokenize(new Uint32Array(this._tokenIds.toArray()));
  }

  /**
   * Set the adapter pointer for LoRA inference
   */
  setAdapter(adapterPtr: number): void {
    this.adapterPtr = adapterPtr;
  }

  /**
   * Remove the adapter
   */
  removeAdapter(): void {
    this.adapterPtr = undefined;
  }

  /**
   * Set the random seed for adapter
   */
  setAdapterRandomSeed(seed: number): void {
    this.adapterRandomSeed = seed;
  }

  /**
   * Creates a safe, copy-on-write fork of the context.
   *
   * This method creates a new context that shares only FULL KV cache pages.
   * Partial pages are dropped and their tokens moved to pending buffer for
   * recomputation. This ensures state isolation - shared pages are read-only
   * (full pages won't be written to), and writable pages are unique per context.
   */
  fork(): Context {
    const forked = new Context(this.model);

    // Always use forkPartial behavior to ensure no writable pages are shared.
    // This drops partial pages and moves their tokens to pending for recomputation.
    const { manager: forkedKvManager, droppedTokenCount } = this.kvPageManager.fork();
    forked.kvPageManager = forkedKvManager;

    const keptTokensLen = forkedKvManager.totalTokens;

    forked._tokenIds = this._tokenIds.slice(0, keptTokensLen);
    forked._positionIds = this._positionIds.slice(0, keptTokensLen);

    // Combine uncommitted tokens from last page with pending tokens
    forked.tokenIdsPending = [
      ...this._tokenIds.toArray().slice(keptTokensLen),
      ...this.tokenIdsPending,
    ];

    // Rebuild the mask for pending tokens
    // Optimization: Build final mask first, then truncate (avoids O(n²) clone+append)
    let maskBuilder = this.tokenMaskCurrent.clone();
    const parentTotalMaskLen = this.tokenIds.length + this.tokenIdsPending.length;
    maskBuilder.removeRange(keptTokensLen, parentTotalMaskLen);

    const pendingCount = forked.tokenIdsPending.length;
    const baseLen = maskBuilder.len();

    // Append all positions at once
    for (let i = 0; i < pendingCount; i++) {
      maskBuilder.append(false);
    }

    // Build masks by truncating from final (cheaper than repeated clone+append)
    forked.tokenMaskPending = [];
    for (let i = 0; i < pendingCount; i++) {
      forked.tokenMaskPending.push(maskBuilder.truncate(baseLen + i + 1));
    }

    // tokenMaskCurrent includes both committed and pending tokens
    forked.tokenMaskCurrent = maskBuilder;

    forked.adapterPtr = this.adapterPtr;
    forked.adapterRandomSeed = this.adapterRandomSeed;
    forked.beginOfSequence = this.beginOfSequence;

    return forked;
  }

  /**
   * Release all KV pages held by this context.
   * Call this when the context is no longer needed to free resources.
   * Safe to call multiple times.
   */
  release(): void {
    this.kvPageManager.release();
  }

  /**
   * Fill with raw text, tokenizing it first
   */
  fill(text: string): void {
    const newTokenIds = this.tokenizer.tokenize(text);
    this.fillTokens([...newTokenIds]);
  }

  /**
   * Fill with an array of token IDs
   */
  fillTokens(newTokenIds: number[]): void {
    const n = newTokenIds.length;
    this.tokenIdsPending.push(...newTokenIds);

    for (let i = 0; i < n; i++) {
      this.tokenMaskCurrent.append(false);
      this.tokenMaskPending.push(this.tokenMaskCurrent.clone());
    }
    this.beginOfSequence = false;
  }

  /**
   * Fill with a single token ID
   */
  fillToken(newTokenId: number): void {
    this.tokenIdsPending.push(newTokenId);
    this.tokenMaskCurrent.append(false);
    this.tokenMaskPending.push(this.tokenMaskCurrent.clone());
    this.beginOfSequence = false;
  }

  /**
   * Fill a system message using the chat formatter
   */
  fillSystem(text: string): void {
    this.formatter.system(text);
    this.flushChatMessages(false);
  }

  /**
   * Fill a user message using the chat formatter (with generation prompt)
   */
  fillUser(text: string): void {
    this.formatter.user(text);
    this.flushChatMessages(true);
  }

  /**
   * Fill a user message without generation prompt
   */
  fillUserOnly(text: string): void {
    this.formatter.user(text);
    this.flushChatMessages(false);
  }

  /**
   * Fill an assistant message using the chat formatter
   */
  fillAssistant(text: string): void {
    this.formatter.assistant(text);
    this.flushChatMessages(false);
  }

  /**
   * Mask specific token indices
   */
  maskTokens(indices: number[], mask: boolean): void {
    this.tokenMaskCurrent.mask(indices, mask);
  }

  /**
   * Mask a range of tokens
   */
  maskTokenRange(start: number, end: number, mask: boolean): void {
    this.tokenMaskCurrent.maskRange(start, end, mask);
  }

  /**
   * Mask a single token
   */
  maskToken(index: number, mask: boolean): void {
    this.tokenMaskCurrent.mask([index], mask);
  }

  /**
   * Grow the KV cache by the specified number of tokens
   */
  growKvPages(numTokens: number): void {
    this.kvPageManager.grow(numTokens);
  }

  /**
   * Shrink the KV cache by the specified number of tokens
   */
  shrinkKvPages(numTokens: number): void {
    this.kvPageManager.shrink(numTokens);
  }

  /**
   * Flush chat messages to tokens
   */
  private flushChatMessages(addGenerationPrompt: boolean): void {
    if (this.formatter.hasMessages()) {
      const prompt = this.formatter.render(
        this.model.promptTemplate,
        addGenerationPrompt,
        this.beginOfSequence
      );
      this.beginOfSequence = false;
      this.formatter.clear();
      this.fill(prompt);
    }
  }

  /**
   * Processes a batch of pending tokens to update the model's internal state.
   */
  async flush(): Promise<void> {
    if (this.tokenIdsPending.length === 0) {
      return;
    }

    const processCount = this.tokenIdsPending.length;

    // Process all pending tokens
    const pendingTokenIds = this.tokenIdsPending.splice(0, processCount);
    const mask = this.tokenMaskPending
      .splice(0, processCount)
      .map((b) => b.buffer);

    const lastPos =
      this.positionIds.length > 0
        ? this.positionIds[this.positionIds.length - 1] + 1
        : 0;

    const positionIds = Array.from(
      { length: pendingTokenIds.length },
      (_, i) => lastPos + i
    );

    this.growKvPages(pendingTokenIds.length);

    const p = this.queue.createForwardPass();
    p.inputTokens(pendingTokenIds, positionIds);
    p.kvCachePtrs(this.kvPagePtrs, this.kvPageLastLen);
    p.attentionMask(mask);

    await p.execute();

    this._tokenIds = this._tokenIds.pushAll(pendingTokenIds);
    this._positionIds = this._positionIds.pushAll(positionIds);
  }

  /**
   * Performs a single autoregressive decoding step.
   * Returns the sampled token ID.
   */
  async decodeStep(sampler: Sampler): Promise<number> {
    if (this.tokenIdsPending.length === 0) {
      throw new Error('Must have at least one seed token');
    }

    const pendingTokenIds = this.tokenIdsPending.splice(0);
    const lastPosId =
      this.positionIds.length > 0
        ? this.positionIds[this.positionIds.length - 1] + 1
        : 0;

    const positionIds = Array.from(
      { length: pendingTokenIds.length },
      (_, i) => lastPosId + i
    );

    this.growKvPages(pendingTokenIds.length);

    const mask = this.tokenMaskPending.splice(0).map((b) => b.buffer);

    const p = this.queue.createForwardPass();
    p.inputTokens(pendingTokenIds, positionIds);
    p.kvCachePtrs(this.kvPagePtrs, this.kvPageLastLen);
    p.attentionMask(mask);

    const outputIdx = pendingTokenIds.length - 1;
    const config = sampler.getConfig();

    // Configure output based on sampler type
    switch (config.type) {
      case 'Multinomial':
        p.outputTokens([outputIdx], config.temperature);
        break;
      case 'TopP':
        p.outputTokensTopP([outputIdx], config.temperature, config.top_p);
        break;
      case 'TopK':
        p.outputTokensTopK([outputIdx], config.temperature, config.top_k);
        break;
      case 'MinP':
        p.outputTokensMinP([outputIdx], config.temperature, config.min_p);
        break;
      case 'TopKTopP':
        p.outputTokensTopKTopP(
          [outputIdx],
          config.temperature,
          config.top_k,
          config.top_p
        );
        break;
      case 'Custom':
        p.outputDistributions([outputIdx], config.temperature, undefined);
        break;
    }

    const res = await p.execute();

    let sampled: number;
    if (config.type === 'Custom' && res.distributions) {
      // Custom sampler needs to sample from distribution
      const dist = res.distributions[0];
      // For now, just take the highest probability token
      sampled = dist.ids[0];
    } else if (res.tokens) {
      sampled = res.tokens[0];
    } else {
      throw new Error('No token generated');
    }

    this._tokenIds = this._tokenIds.pushAll(pendingTokenIds);
    this._positionIds = this._positionIds.pushAll(positionIds);

    return sampled;
  }

  /**
   * Performs a single decoding step and returns the distribution.
   */
  async decodeStepDist(): Promise<Distribution> {
    if (this.tokenIdsPending.length === 0) {
      throw new Error('Must have at least one seed token');
    }

    const pendingTokenIds = this.tokenIdsPending.splice(0);
    const lastPosId =
      this.positionIds.length > 0
        ? this.positionIds[this.positionIds.length - 1] + 1
        : 0;

    const positionIds = Array.from(
      { length: pendingTokenIds.length },
      (_, i) => lastPosId + i
    );

    this.growKvPages(pendingTokenIds.length);

    const mask = this.tokenMaskPending.splice(0).map((b) => b.buffer);

    const p = this.queue.createForwardPass();
    p.inputTokens(pendingTokenIds, positionIds);
    p.kvCachePtrs(this.kvPagePtrs, this.kvPageLastLen);
    p.attentionMask(mask);

    const outputIdx = pendingTokenIds.length - 1;
    p.outputDistributions([outputIdx], 1.0, undefined);

    const res = await p.execute();

    if (!res.distributions || res.distributions.length === 0) {
      throw new Error('No distribution returned');
    }

    const dist = res.distributions[0];

    this._tokenIds = this._tokenIds.pushAll(pendingTokenIds);
    this._positionIds = this._positionIds.pushAll(positionIds);

    return dist;
  }

  /**
   * Generates text autoregressively until a stop condition is met.
   *
   * Fill context with messages using fillSystem(), fillUser(), fillAssistant()
   * before calling generate().
   *
   * Can be called with either:
   * - New API: generate(options: GenerateOptions)
   * - Legacy API: generate(sampler: Sampler, stopCondition: StopCondition)
   *
   * @example New API with stateful context
   * ```ts
   * ctx.fillSystem('You are helpful.');
   * ctx.fillUser('Hello!');
   *
   * const result = await ctx.generate({
   *   sampling: Sampler.topP(0.6, 0.95),
   *   stop: { maxTokens: 256, sequences: model.eosTokens }
   * });
   * ```
   *
   * @example New API with sampling config
   * ```ts
   * const result = await ctx.generate({
   *   sampling: { topP: 0.95, temperature: 0.6 },
   *   stop: { maxTokens: 256, sequences: model.eosTokens }
   * });
   * ```
   *
   * @example Legacy API (deprecated)
   * ```ts
   * const result = await ctx.generate(sampler, stopCondition);
   * ```
   */
  async generate(optionsOrSampler: GenerateOptions | Sampler, stopCondition?: StopCondition): Promise<string> {
    let sampler: Sampler;
    let stop: StopCondition;

    if (optionsOrSampler instanceof Sampler) {
      // Legacy API: generate(sampler, stopCondition)
      if (!stopCondition) {
        throw new Error('stopCondition is required when using legacy generate API');
      }
      sampler = optionsOrSampler;
      stop = stopCondition;
    } else {
      // New API: generate(options)
      const options = optionsOrSampler;

      // Convert sampling config to Sampler
      sampler = this.toSampler(options.sampling);

      // Convert stop config to StopCondition
      stop = this.toStopCondition(options.stop);
    }

    const generatedTokenIds: number[] = [];

    // The autoregressive generation loop
    while (true) {
      const nextTokenId = await this.decodeStep(sampler);
      this.fillToken(nextTokenId);
      generatedTokenIds.push(nextTokenId);

      if (stop.check(generatedTokenIds)) {
        break;
      }
    }

    return this.tokenizer.detokenize(new Uint32Array(generatedTokenIds));
  }

  /**
   * Convert a SamplingConfig or Sampler to a Sampler instance
   */
  private toSampler(sampling: SamplingConfig | Sampler): Sampler {
    // If it's already a Sampler instance, use it directly
    if (sampling instanceof Sampler) {
      return sampling;
    }

    // It's a user-friendly SamplingConfig
    const config = sampling as SamplingConfig;
    const temp = config.temperature ?? 1.0;

    // Determine which sampler type to use based on provided options
    if (config.topK !== undefined && config.topP !== undefined) {
      return Sampler.topKTopP(temp, config.topK, config.topP);
    }
    if (config.topP !== undefined) {
      return Sampler.topP(temp, config.topP);
    }
    if (config.topK !== undefined) {
      return Sampler.topK(temp, config.topK);
    }
    if (config.minP !== undefined) {
      return Sampler.minP(temp, config.minP);
    }

    // Default: greedy if temp is 0, otherwise basic multinomial
    if (temp === 0) {
      return Sampler.greedy();
    }

    // Return multinomial sampler with the given temperature
    return Sampler.multinomial(temp);
  }

  /**
   * Convert a StopConfig to a StopCondition
   */
  private toStopCondition(stopConfig: StopConfig): StopCondition {
    const conditions: StopCondition[] = [];

    if (stopConfig.maxTokens !== undefined) {
      conditions.push(new MaxLen(stopConfig.maxTokens));
    }

    if (stopConfig.sequences !== undefined && stopConfig.sequences.length > 0) {
      conditions.push(new AnyEndsWith(stopConfig.sequences));
    }

    if (conditions.length === 0) {
      throw new Error('At least one stop condition (maxTokens or sequences) must be specified');
    }

    // Combine all conditions with OR
    let result = conditions[0];
    for (let i = 1; i < conditions.length; i++) {
      result = result.or(conditions[i]);
    }

    return result;
  }

  /**
   * Generates text using beam search decoding until a stop condition is met.
   */
  async generateWithBeam(
    stopCondition: StopCondition,
    beamSize: number
  ): Promise<string> {
    type BeamState = [Context, number[], number]; // [context, generated_tokens, score]

    let beams: BeamState[] = [[this.fork(), [], 0.0]];

    while (true) {
      // Check if any beam satisfies the stop condition
      const completedBeam = beams.find(([, generated]) =>
        stopCondition.check(generated)
      );

      if (completedBeam) {
        const [winningBeam, generatedTokens] = completedBeam;

        // Adopt the state from the winning beam
        this.kvPageManager.adopt(winningBeam.kvPageManager);
        this._tokenIds = winningBeam._tokenIds.fork();
        this._positionIds = winningBeam._positionIds.fork();
        this.tokenIdsPending = [...winningBeam.tokenIdsPending];

        // Release all beams
        for (const [beam] of beams) {
          beam.release();
        }

        return this.tokenizer.detokenize(new Uint32Array(generatedTokens));
      }

      // Progress all beams in parallel
      const nextDists = await Promise.all(
        beams.map(([beam]) => beam.decodeStepDist())
      );

      // Expand beams
      const nextBeams: BeamState[] = [];
      for (let i = 0; i < beams.length; i++) {
        const [beam, generated, score] = beams[i];
        const nextDist = nextDists[i];

        // Expand with top candidates
        const expandCount = Math.min(beamSize, nextDist.ids.length);
        for (let j = 0; j < expandCount; j++) {
          const nextBeam = beam.fork();
          const tokenId = nextDist.ids[j];
          nextBeam.fillToken(tokenId);

          const nextGenerated = [...generated, tokenId];
          const nextScore = score + Math.log(nextDist.probs[j]);

          nextBeams.push([nextBeam, nextGenerated, nextScore]);
        }

        // Release the old beam after forking
        beam.release();
      }

      // Prune: Sort by score (descending) and keep top beamSize
      nextBeams.sort((a, b) => b[2] - a[2]);

      // Release pruned beams
      const prunedBeams = nextBeams.slice(beamSize);
      for (const [prunedCtx] of prunedBeams) {
        prunedCtx.release();
      }

      beams = nextBeams.slice(0, beamSize);
    }
  }
}
