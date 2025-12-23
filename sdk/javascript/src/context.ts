// Context class for managing conversation state and generation.
// Mirrors the Rust Context from inferlet/src/context.rs

import { Model, Queue } from './model.js';
import { Tokenizer } from './tokenizer.js';
import { ChatFormatter } from './chat.js';
import { Brle } from './brle.js';
import { KvPage, ForwardPass, type Distribution } from './forward.js';
import { toSamplerType, type SamplerType, type SamplingConfig } from './sampler.js';
import { KvPageManager } from './kv-page-manager.js';



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
  /** Sampling configuration */
  sampling: SamplingConfig;
  /** Stop conditions */
  stop: StopConfig;
}

/**
 * Options for the generateWithBeam() method
 */
export interface GenerateBeamOptions {
  /** Number of beams to use */
  beamSize: number;
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

  private _tokenIds: number[] = [];
  tokenIdsPending: number[] = [];

  tokenMaskPending: Brle[] = [];
  tokenMaskCurrent: Brle;

  private _positionIds: number[] = [];

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
    return this._tokenIds;
  }

  set tokenIds(value: number[]) {
    this._tokenIds = [...value];
  }

  /**
   * The position IDs (as regular array for backward compatibility)
   */
  get positionIds(): number[] {
    return this._positionIds;
  }

  set positionIds(value: number[]) {
    this._positionIds = [...value];
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

    // Validate kvPageLastLen is within valid range
    if (kvPageLastLen < 0 || kvPageLastLen > kvPageSize) {
      throw new Error(
        `kvPageLastLen out of range: expected 0..${kvPageSize}, got ${kvPageLastLen}`
      );
    }

    // Handle empty state and validate kvPages/kvPageLastLen consistency
    let expectedTokens: number;
    if (kvPages.length === 0) {
      if (kvPageLastLen !== 0) {
        throw new Error(
          `Invalid state: kvPages is empty but kvPageLastLen is ${kvPageLastLen} (must be 0 when kvPages is empty)`
        );
      }
      expectedTokens = 0;
    } else {
      expectedTokens = (kvPages.length - 1) * kvPageSize + kvPageLastLen;
    }

    // Verify the token count matches the KV page state
    if (prefixTokens.length !== expectedTokens) {
      throw new Error(
        `Token count mismatch: expected ${expectedTokens}, got ${prefixTokens.length} (kvPages.length=${kvPages.length}, kvPageLastLen=${kvPageLastLen}, kvPageSize=${kvPageSize})`
      );
    }

    ctx._tokenIds = [...prefixTokens];
    ctx._positionIds = Array.from({ length: prefixTokens.length }, (_, i) => i);

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
    return this.tokenizer.detokenize(new Uint32Array(this._tokenIds));
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
      ...this._tokenIds.slice(keptTokensLen),
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

    this._tokenIds.push(...pendingTokenIds);
    this._positionIds.push(...positionIds);
  }

  /**
   * Performs a single autoregressive decoding step.
   * Returns the sampled token ID.
   */
  private async decodeStep(samplerType: SamplerType): Promise<number> {
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

    // Configure output based on sampler type
    switch (samplerType.type) {
      case 'Multinomial':
        p.outputTokens([outputIdx], samplerType.temperature);
        break;
      case 'TopP':
        p.outputTokensTopP([outputIdx], samplerType.temperature, samplerType.top_p);
        break;
      case 'TopK':
        p.outputTokensTopK([outputIdx], samplerType.temperature, samplerType.top_k);
        break;
      case 'MinP':
        p.outputTokensMinP([outputIdx], samplerType.temperature, samplerType.min_p);
        break;
      case 'TopKTopP':
        p.outputTokensTopKTopP(
          [outputIdx],
          samplerType.temperature,
          samplerType.top_k,
          samplerType.top_p
        );
        break;
      case 'Custom':
        p.outputDistributions([outputIdx], samplerType.temperature, undefined);
        break;
    }

    const res = await p.execute();

    let sampled: number;
    if (samplerType.type === 'Custom' && res.distributions) {
      // Custom sampler needs to sample from distribution
      const dist = res.distributions[0];
      // For now, just take the highest probability token
      sampled = dist.ids[0];
    } else if (res.tokens) {
      sampled = res.tokens[0];
    } else {
      throw new Error('No token generated');
    }

    this._tokenIds.push(...pendingTokenIds);
    this._positionIds.push(...positionIds);

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

    this._tokenIds.push(...pendingTokenIds);
    this._positionIds.push(...positionIds);

    return dist;
  }

  /**
   * Generates text autoregressively until a stop condition is met.
   *
   * Fill context with messages using fillSystem(), fillUser(), fillAssistant()
   * before calling generate().
   *
   * @example
   * ```ts
   * ctx.fillSystem('You are helpful.');
   * ctx.fillUser('Hello!');
   *
   * const result = await ctx.generate({
   *   sampling: { topP: 0.95, temperature: 0.6 },
   *   stop: { maxTokens: 256, sequences: model.eosTokens }
   * });
   * ```
   */
  async generate(options: GenerateOptions): Promise<string> {
    const samplerType = toSamplerType(options.sampling);
    const { maxTokens, sequences } = options.stop;

    if (maxTokens === undefined && (sequences === undefined || sequences.length === 0)) {
      throw new Error('At least one stop condition (maxTokens or sequences) must be specified');
    }

    const generatedTokenIds: number[] = [];

    // The autoregressive generation loop
    while (true) {
      const nextTokenId = await this.decodeStep(samplerType);
      this.fillToken(nextTokenId);
      generatedTokenIds.push(nextTokenId);

      // Check stop conditions
      if (maxTokens !== undefined && generatedTokenIds.length >= maxTokens) {
        break;
      }
      if (sequences !== undefined && this.endsWithAny(generatedTokenIds, sequences)) {
        break;
      }
    }

    return this.tokenizer.detokenize(new Uint32Array(generatedTokenIds));
  }

  /**
   * Check if tokenIds ends with any of the given sequences
   */
  private endsWithAny(tokenIds: number[], sequences: number[][]): boolean {
    for (const seq of sequences) {
      if (seq.length === 0) continue;
      if (tokenIds.length < seq.length) continue;

      let matches = true;
      const start = tokenIds.length - seq.length;
      for (let i = 0; i < seq.length; i++) {
        if (tokenIds[start + i] !== seq[i]) {
          matches = false;
          break;
        }
      }
      if (matches) return true;
    }
    return false;
  }

  /**
   * Generates text using beam search decoding until a stop condition is met.
   *
   * @example
   * ```ts
   * const result = await ctx.generateWithBeam({
   *   beamSize: 4,
   *   stop: { maxTokens: 128, sequences: model.eosTokens }
   * });
   * ```
   */
  async generateWithBeam(options: GenerateBeamOptions): Promise<string> {
    const { beamSize, stop } = options;
    const { maxTokens, sequences } = stop;

    if (maxTokens === undefined && (sequences === undefined || sequences.length === 0)) {
      throw new Error('At least one stop condition (maxTokens or sequences) must be specified');
    }

    type BeamState = [Context, number[], number]; // [context, generated_tokens, score]

    let beams: BeamState[] = [[this.fork(), [], 0.0]];

    const checkStop = (generated: number[]): boolean => {
      if (maxTokens !== undefined && generated.length >= maxTokens) {
        return true;
      }
      if (sequences !== undefined && this.endsWithAny(generated, sequences)) {
        return true;
      }
      return false;
    };

    while (true) {
      // Check if any beam satisfies the stop condition
      const completedBeam = beams.find(([, generated]) => checkStop(generated));

      if (completedBeam) {
        const [winningBeam, generatedTokens] = completedBeam;

        // Adopt the state from the winning beam
        this.kvPageManager.adopt(winningBeam.kvPageManager);
        this._tokenIds = [...winningBeam._tokenIds];
        this._positionIds = [...winningBeam._positionIds];
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
