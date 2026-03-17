/**
 * Drafter interface for speculative decoding.
 *
 * Speculative decoding accelerates text generation by using a small, fast "drafter"
 * model to propose candidate tokens. These draft tokens are then verified in a single,
 * parallel forward pass by the main model, allowing multiple tokens to be generated
 * for the cost of one forward pass when the drafter's predictions are correct.
 *
 * The drafter produces a **forest of Tries** (prefix trees) representing multiple
 * speculative token sequences.
 *
 * Mirrors the Rust Drafter trait from inferlet/src/drafter.rs
 */

/**
 * Interface for implementing draft token generation in speculative decoding.
 */
export interface Drafter {
  /**
   * Updates the drafter's internal state with new context tokens.
   *
   * This method is called to keep the drafter synchronized with the main model's
   * token history. It is invoked:
   * - Once at the start of generation with the full token history.
   * - After each verification step with the tokens that were accepted.
   *
   * @param context - Token IDs to append to the drafter's internal context.
   *   These are tokens that have been confirmed by the main model.
   */
  update(context: number[]): void;

  /**
   * Generates a forest of draft token sequences for speculative verification.
   *
   * Returns a Trie forest encoded as two parallel arrays in **depth-first search (DFS)
   * order**. This encoding allows the verifier to efficiently traverse and verify
   * multiple speculative paths.
   *
   * @returns A tuple `[draftTokens, draftPosIds]` where:
   *   - `draftTokens`: The proposed token IDs in DFS traversal order.
   *   - `draftPosIds`: The depth of each token in the trie, starting at 1 for root nodes.
   *     These positions are **relative to the last context token**.
   *
   * The returned arrays must satisfy the following invariants:
   * 1. **Equal length**: Both arrays must have the same length.
   * 2. **Valid depths**: All position IDs must >= 1 (depth 1 = immediate continuation
   *    of the context).
   * 3. **Valid DFS order**: The sequence must represent a valid DFS traversal of a Trie
   *    forest.
   */
  draft(): [number[], number[]];
}

/**
 * Empty drafter that produces no drafts (disables speculative decoding).
 *
 * Use this when you want to disable speculative decoding or as a placeholder.
 */
export class EmptyDrafter implements Drafter {
  update(_context: number[]): void {
    // No-op
  }

  draft(): [number[], number[]] {
    return [[], []];
  }
}
