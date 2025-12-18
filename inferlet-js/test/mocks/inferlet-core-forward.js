// inferlet-js/test/mocks/inferlet-core-forward.js
// Mock implementation of inferlet:core/forward

import { Pollable } from './wasi-io-poll.js';

let nextTokenId = 100;

export class ForwardPassResult {
  #tokens;
  #distributions;

  constructor(tokens, distributions = null) {
    this.#tokens = tokens;
    this.#distributions = distributions;
  }

  pollable() {
    return new Pollable(true);
  }

  getDistributions() {
    return this.#distributions;
  }

  getTokens() {
    return this.#tokens;
  }
}

export class ForwardPass {
  #inputTokens = [];
  #inputPositions = [];
  #outputIndices = [];
  #outputMode = 'tokens'; // 'tokens' | 'distributions'
  #temperature = 0.0;
  #topK = null;
  #topP = null;

  execute() {
    // Generate mock output tokens based on input
    const numOutputs = this.#outputIndices.length || 1;
    const tokens = new Uint32Array(numOutputs);

    for (let i = 0; i < numOutputs; i++) {
      // Deterministic mock: output token based on input
      tokens[i] = nextTokenId++;
    }

    if (this.#outputMode === 'distributions') {
      // Return distributions instead of tokens
      const distributions = [];
      for (let i = 0; i < numOutputs; i++) {
        const ids = new Uint32Array([tokens[i], tokens[i] + 1, tokens[i] + 2]);
        const probs = new Float32Array([0.7, 0.2, 0.1]);
        distributions.push([ids, probs]);
      }
      return new ForwardPassResult(null, distributions);
    }

    return new ForwardPassResult(tokens, null);
  }

  // Internal setters used by module functions
  _setInputTokens(tokens, positions) {
    this.#inputTokens = Array.from(tokens);
    this.#inputPositions = Array.from(positions);
  }

  _setOutputIndices(indices) {
    this.#outputIndices = Array.from(indices);
  }

  _setOutputMode(mode) {
    this.#outputMode = mode;
  }

  _setTemperature(temp) {
    this.#temperature = temp;
  }

  _setTopK(k) {
    this.#topK = k;
  }

  _setTopP(p) {
    this.#topP = p;
  }
}

// Module functions
export function createForwardPass(queue) {
  return new ForwardPass();
}

export function attentionMask(pass, mask) {
  // No-op in mock
}

export function kvCache(pass, kvPagePtrs, lastKvPageLen) {
  // No-op in mock
}

export function inputEmbeddings(pass, embPtrs, positions) {
  // No-op in mock
}

export function inputTokens(pass, tokens, positions) {
  pass._setInputTokens(tokens, positions);
}

export function outputEmbeddings(pass, embPtrs, indices) {
  // No-op in mock
}

export function outputDistributions(pass, indices, temperature, topK) {
  pass._setOutputIndices(indices);
  pass._setOutputMode('distributions');
  pass._setTemperature(temperature);
  if (topK !== undefined) pass._setTopK(topK);
}

export function outputTokens(pass, indices, temperature) {
  pass._setOutputIndices(indices);
  pass._setOutputMode('tokens');
  pass._setTemperature(temperature);
}

export function outputTokensTopK(pass, indices, temperature, topK) {
  pass._setOutputIndices(indices);
  pass._setOutputMode('tokens');
  pass._setTemperature(temperature);
  pass._setTopK(topK);
}

export function outputTokensTopP(pass, indices, temperature, topP) {
  pass._setOutputIndices(indices);
  pass._setOutputMode('tokens');
  pass._setTemperature(temperature);
  pass._setTopP(topP);
}

export function outputTokensMinP(pass, indices, temperature, minP) {
  pass._setOutputIndices(indices);
  pass._setOutputMode('tokens');
  pass._setTemperature(temperature);
}

export function outputTokensTopKTopP(pass, indices, temperature, topK, topP) {
  pass._setOutputIndices(indices);
  pass._setOutputMode('tokens');
  pass._setTemperature(temperature);
  pass._setTopK(topK);
  pass._setTopP(topP);
}

// Test helper
export function __resetMockState() {
  nextTokenId = 100;
}
