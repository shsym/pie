// Context class for managing conversation state and generation.
// Mirrors the Rust Context from inferlet/src/context.rs

import { Model, Queue } from './model.js';
import { Tokenizer } from './tokenizer.js';
import { ChatFormatter } from './chat.js';
import { Brle } from './brle.js';
import { KvPage, ForwardPass, type Distribution } from './forward.js';
import { Sampler, type SamplerType } from './sampler.js';
import type { StopCondition } from './stop-condition.js';

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

  tokenIds: number[] = [];
  tokenIdsPending: number[] = [];

  tokenMaskPending: Brle[] = [];
  tokenMaskCurrent: Brle;

  positionIds: number[] = [];

  kvPages: KvPage[] = [];
  kvPageLastLen: number = 0;
  kvPageSize: number;

  adapterPtr?: number;
  adapterRandomSeed?: number;

  beginOfSequence: boolean = true;

  constructor(model: Model) {
    this.queue = model.createQueue();
    this.kvPageSize = model.getKvPageSize();
    this.tokenizer = model.getTokenizer();
    this.model = model;
    this.formatter = new ChatFormatter();
    this.tokenMaskCurrent = Brle.new(0);
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
    const kvPageSize = model.getKvPageSize();

    // Verify the token count matches the KV page state
    const expectedTokens = (kvPages.length - 1) * kvPageSize + kvPageLastLen;
    if (prefixTokens.length !== expectedTokens) {
      throw new Error(
        `Token count mismatch: expected ${expectedTokens}, got ${prefixTokens.length}`
      );
    }

    ctx.tokenIds = [...prefixTokens];
    ctx.positionIds = Array.from({ length: prefixTokens.length }, (_, i) => i);
    ctx.kvPages = kvPages;
    ctx.kvPageLastLen = kvPageLastLen;
    ctx.tokenMaskCurrent = Brle.new(prefixTokens.length);
    ctx.beginOfSequence = false;

    return ctx;
  }

  /**
   * Get the model associated with this context
   */
  getModel(): Model {
    return this.model;
  }

  /**
   * Get the queue associated with this context
   */
  getQueue(): Queue {
    return this.queue;
  }

  /**
   * Get all token IDs in the context
   */
  getTokenIds(): number[] {
    return this.tokenIds;
  }

  /**
   * Get the text representation of all tokens
   */
  getText(): string {
    return this.tokenizer.detokenize(new Uint32Array(this.tokenIds));
  }

  /**
   * Returns the unique IDs of the KV cache pages currently in use
   */
  getKvPagePtrs(): number[] {
    return this.kvPages.map((p) => p.ptr);
  }

  /**
   * Returns the number of tokens stored in the last KV cache page
   */
  getKvPageLastLen(): number {
    return this.kvPageLastLen;
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
   * This method creates a new context that shares the immutable history of the current
   * one. If the last KV-cache page is not full, its tokens are moved to the
   * `tokenIdsPending` buffer of the new context to be recomputed, ensuring state isolation.
   */
  fork(): Context {
    const forked = new Context(this.model);

    if (this.kvPageLastLen === this.kvPageSize) {
      // Easy case: the last page is full, we can share everything.
      forked.tokenIds = [...this.tokenIds];
      forked.tokenIdsPending = [...this.tokenIdsPending];
      forked.kvPages = [...this.kvPages];
      forked.kvPageLastLen = this.kvPageLastLen;
      forked.positionIds = [...this.positionIds];
      forked.tokenMaskPending = this.tokenMaskPending.map((m) => m.clone());
      forked.tokenMaskCurrent = this.tokenMaskCurrent.clone();
    } else {
      // Hard case: the last page is partially full and must be recomputed.
      const keptKvPageLen = Math.max(0, this.kvPages.length - 1);
      const keptTokensLen = keptKvPageLen * this.kvPageSize;

      forked.tokenIds = this.tokenIds.slice(0, keptTokensLen);
      forked.kvPages = this.kvPages.slice(0, keptKvPageLen);
      forked.positionIds = this.positionIds.slice(0, keptTokensLen);

      // Combine uncommitted tokens from last page with pending tokens
      forked.tokenIdsPending = [
        ...this.tokenIds.slice(keptTokensLen),
        ...this.tokenIdsPending,
      ];

      forked.kvPageLastLen =
        forked.kvPages.length > 0 ? this.kvPageSize : 0;

      // Rebuild the mask for pending tokens
      let maskBuilder = this.tokenMaskCurrent.clone();
      const parentTotalMaskLen =
        this.tokenIds.length + this.tokenIdsPending.length;
      // Remove the range that's being moved to pending
      maskBuilder.removeRange(keptTokensLen, parentTotalMaskLen);

      forked.tokenMaskPending = [];
      for (let i = 0; i < forked.tokenIdsPending.length; i++) {
        maskBuilder.append(false);
        forked.tokenMaskPending.push(maskBuilder.clone());
      }

      // tokenMaskCurrent includes both committed and pending tokens
      forked.tokenMaskCurrent = maskBuilder;
    }

    forked.adapterPtr = this.adapterPtr;
    forked.adapterRandomSeed = this.adapterRandomSeed;
    forked.beginOfSequence = this.beginOfSequence;

    return forked;
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
   * Adjusts the number of KV pages to match the required number of tokens.
   */
  private adjustKvPages(numTokens: number): void {
    if (numTokens === 0) return;

    const currentTokens =
      this.kvPages.length === 0
        ? this.kvPageLastLen
        : (this.kvPages.length - 1) * this.kvPageSize + this.kvPageLastLen;

    const newTotalTokens = currentTokens + numTokens;
    if (newTotalTokens < 0) {
      throw new Error('Token count adjustment resulted in underflow');
    }

    const currentPages = this.kvPages.length;
    const requiredPages = Math.ceil(newTotalTokens / this.kvPageSize);

    if (requiredPages > currentPages) {
      // Grow: Allocate new pages
      const newPagesNeeded = requiredPages - currentPages;
      const newKvPages = this.queue.newKvPages(newPagesNeeded);
      this.kvPages.push(...newKvPages);
    } else if (requiredPages < currentPages) {
      // Shrink: Release excess pages
      const pagesToRelease = this.kvPages.splice(requiredPages);
      for (const page of pagesToRelease) {
        page.release();
      }
    }

    // Update the length of the last page
    const lastPageLen = newTotalTokens % this.kvPageSize;
    this.kvPageLastLen =
      lastPageLen === 0 && newTotalTokens > 0 ? this.kvPageSize : lastPageLen;
  }

  /**
   * Grow the KV cache by the specified number of tokens
   */
  growKvPages(numTokens: number): void {
    this.adjustKvPages(numTokens);
  }

  /**
   * Shrink the KV cache by the specified number of tokens
   */
  shrinkKvPages(numTokens: number): void {
    this.adjustKvPages(-numTokens);
  }

  /**
   * Flush chat messages to tokens
   */
  private flushChatMessages(addGenerationPrompt: boolean): void {
    if (this.formatter.hasMessages()) {
      const prompt = this.formatter.render(
        this.model.getPromptTemplate(),
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
    p.kvCachePtrs(this.getKvPagePtrs(), this.kvPageLastLen);
    p.attentionMask(mask);

    await p.execute();

    this.tokenIds.push(...pendingTokenIds);
    this.positionIds.push(...positionIds);
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
    p.kvCachePtrs(this.getKvPagePtrs(), this.kvPageLastLen);
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

    this.tokenIds.push(...pendingTokenIds);
    this.positionIds.push(...positionIds);

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
    p.kvCachePtrs(this.getKvPagePtrs(), this.kvPageLastLen);
    p.attentionMask(mask);

    const outputIdx = pendingTokenIds.length - 1;
    p.outputDistributions([outputIdx], 1.0, undefined);

    const res = await p.execute();

    if (!res.distributions || res.distributions.length === 0) {
      throw new Error('No distribution returned');
    }

    const dist = res.distributions[0];

    this.tokenIds.push(...pendingTokenIds);
    this.positionIds.push(...positionIds);

    return dist;
  }

  /**
   * Generates text autoregressively until a stop condition is met.
   */
  async generate(sampler: Sampler, stopCondition: StopCondition): Promise<string> {
    const generatedTokenIds: number[] = [];

    // The autoregressive generation loop
    while (true) {
      const nextTokenId = await this.decodeStep(sampler);
      this.fillToken(nextTokenId);
      generatedTokenIds.push(nextTokenId);

      if (stopCondition.check(generatedTokenIds)) {
        break;
      }
    }

    return this.tokenizer.detokenize(new Uint32Array(generatedTokenIds));
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
        const [beam, generatedTokens] = completedBeam;

        // Adopt the state from the winning beam
        this.kvPageLastLen = beam.kvPageLastLen;
        this.tokenIds = [...beam.tokenIds];
        this.tokenIdsPending = [...beam.tokenIdsPending];
        this.kvPages = [...beam.kvPages];

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
        for (let j = 0; j < Math.min(beamSize, nextDist.ids.length); j++) {
          const nextBeam = beam.fork();
          nextBeam.fillToken(nextDist.ids[j]);

          const nextGenerated = [...generated, nextDist.ids[j]];
          const nextScore = score + Math.log(nextDist.probs[j]);

          nextBeams.push([nextBeam, nextGenerated, nextScore]);
        }
      }

      // Prune: Sort by score (descending) and keep top beamSize
      nextBeams.sort((a, b) => b[2] - a[2]);
      beams = nextBeams.slice(0, beamSize);
    }
  }
}
