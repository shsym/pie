// StopCondition - defines stopping conditions during token generation
// This mirrors the Rust inferlet stop_condition module API

/**
 * A trait-like interface for defining stopping conditions during token generation.
 */
export interface StopCondition {
  /**
   * Checks if the generation should stop based on the sequence of token IDs.
   */
  check(tokenIds: number[]): boolean;

  /**
   * Combines this condition with another using a logical OR.
   * This allows for creating complex conditions by chaining calls.
   * Example: maxLen(100).or(endsWith([50256]))
   */
  or(other: StopCondition): Or;
}

// --- Concrete Conditions ---

/**
 * Stops generation if the sequence ends with a specific sub-sequence of tokens.
 */
export class EndsWith implements StopCondition {
  constructor(private readonly tokenIds: number[]) {}

  check(tokenIds: number[]): boolean {
    if (this.tokenIds.length === 0) {
      return true;
    }
    if (tokenIds.length < this.tokenIds.length) {
      return false;
    }

    const start = tokenIds.length - this.tokenIds.length;
    for (let i = 0; i < this.tokenIds.length; i++) {
      if (tokenIds[start + i] !== this.tokenIds[i]) {
        return false;
      }
    }
    return true;
  }

  or(other: StopCondition): Or {
    return new Or(this, other);
  }
}

/**
 * Stops generation if the sequence reaches a maximum length.
 */
export class MaxLen implements StopCondition {
  constructor(private readonly maxTokens: number) {}

  check(tokenIds: number[]): boolean {
    return tokenIds.length >= this.maxTokens;
  }

  or(other: StopCondition): Or {
    return new Or(this, other);
  }
}

// --- Combinators ---

/**
 * A combinator that stops if *any* of its inner conditions are met.
 * This version is specialized for `EndsWith` to handle a dynamic list efficiently.
 */
export class AnyEndsWith implements StopCondition {
  private readonly conditions: EndsWith[];

  constructor(stopSequences: number[][]) {
    this.conditions = stopSequences.map((tokenIds) => new EndsWith(tokenIds));
  }

  check(tokenIds: number[]): boolean {
    return this.conditions.some((c) => c.check(tokenIds));
  }

  or(other: StopCondition): Or {
    return new Or(this, other);
  }
}

/**
 * A generic combinator that stops if either of its two conditions (A or B) is met.
 * This is the backbone of the `.or()` chaining method.
 */
export class Or implements StopCondition {
  constructor(
    private readonly first: StopCondition,
    private readonly second: StopCondition
  ) {}

  check(tokenIds: number[]): boolean {
    return this.first.check(tokenIds) || this.second.check(tokenIds);
  }

  or(other: StopCondition): Or {
    return new Or(this, other);
  }
}

// --- Constructor Functions ---

/**
 * Creates a condition that stops when the generated sequence reaches `maxTokens`.
 */
export function maxLen(maxTokens: number): MaxLen {
  return new MaxLen(maxTokens);
}

/**
 * Creates a condition that stops if the sequence ends with any of the provided token sequences.
 */
export function endsWithAny(stopSequences: number[][]): AnyEndsWith {
  return new AnyEndsWith(stopSequences);
}

/**
 * Creates a condition that stops if the sequence ends with a single provided token sequence.
 */
export function endsWith(tokenIds: number[]): EndsWith {
  return new EndsWith(tokenIds);
}
