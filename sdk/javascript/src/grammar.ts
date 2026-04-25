// Grammar, Matcher, Constraint, GrammarConstraint, and Schema for
// constrained decoding.
//
// For most use cases, build a `Schema` and pass it as `constrain` to
// `ctx.generate({ ... })` — the SDK compiles it into a stateful matcher
// and drives it per generated token.
//
//     const text = await ctx.generateText({
//         sampler: Sampler.argmax(),
//         maxTokens: 512,
//         constrain: Schema.ebnf(grammarStr),
//     });
//
// For custom logic (banned tokens, learned constraints, etc.), implement
// the `Constraint` interface and pass an instance directly.

import {
    Grammar as _Grammar,
    Matcher as _Matcher,
} from 'pie:core/inference';
import type { Brle } from 'pie:core/inference';
import type { Tokenizer, Model } from './model.js';

// ─── Grammar / Matcher (raw resource wrappers) ──────────────────────────

/**
 * Describes the structure that LLM output must conform to.
 *
 * Wraps the `pie:core/inference.Grammar` WIT resource.
 */
export class Grammar {
    /** @internal */
    readonly _handle: _Grammar;

    private constructor(handle: _Grammar) {
        this._handle = handle;
    }

    /** Construct from a JSON Schema string. */
    static fromJsonSchema(schema: string): Grammar {
        return new Grammar(_Grammar.fromJsonSchema(schema));
    }

    /** Construct a built-in free-form JSON grammar (any valid JSON). */
    static json(): Grammar {
        return new Grammar(_Grammar.json());
    }

    /** Construct from a regular expression pattern. */
    static fromRegex(pattern: string): Grammar {
        return new Grammar(_Grammar.fromRegex(pattern));
    }

    /** Construct from an EBNF grammar string. */
    static fromEbnf(ebnf: string): Grammar {
        return new Grammar(_Grammar.fromEbnf(ebnf));
    }
}

/**
 * Stateful matcher that walks a grammar automaton as tokens are accepted.
 *
 * Most callers should reach for {@link Schema} or {@link GrammarConstraint}
 * instead — Matcher is the lower-level resource wrapper.
 */
export class Matcher {
    /** @internal */
    readonly _handle: _Matcher;

    constructor(grammar: Grammar, tokenizer: Tokenizer) {
        this._handle = new _Matcher(grammar._handle, tokenizer._handle);
    }

    acceptTokens(tokenIds: Uint32Array): void {
        this._handle.acceptTokens(tokenIds);
    }

    nextTokenLogitMask(): Brle {
        return this._handle.nextTokenLogitMask();
    }

    isTerminated(): boolean {
        return this._handle.isTerminated();
    }

    reset(): void {
        this._handle.reset();
    }
}

// ─── Constraint protocol ────────────────────────────────────────────────

/**
 * Stateful sampling constraint.
 *
 * On each generation step, the {@link TokenStream} passes any newly
 * accepted tokens (or empty on the first step) and gets back the
 * BRLE-encoded logit mask for the next position.
 *
 * Returning an empty mask means "no restriction".
 */
export interface Constraint {
    /** Advance internal state with `accepted` tokens, then return the mask
     *  for the next position. */
    step(accepted: Uint32Array): Brle;
}

/**
 * Grammar-driven {@link Constraint} backed by a host {@link Matcher}.
 *
 * Most callers should reach for {@link Schema} instead — `GrammarConstraint`
 * is the lower-level type for callers that want to keep a constraint
 * instance around (e.g., to compose with custom constraints).
 */
export class GrammarConstraint implements Constraint {
    private readonly _matcher: Matcher;

    constructor(matcher: Matcher) {
        this._matcher = matcher;
    }

    /** Build from a pre-compiled grammar (compile once, reuse). */
    static fromGrammar(grammar: Grammar, model: Model): GrammarConstraint {
        return new GrammarConstraint(new Matcher(grammar, model.tokenizer()));
    }

    /** Build from a JSON Schema string. */
    static fromJsonSchema(schema: string, model: Model): GrammarConstraint {
        return GrammarConstraint.fromGrammar(Grammar.fromJsonSchema(schema), model);
    }

    /** Build a constraint that accepts any valid JSON. */
    static json(model: Model): GrammarConstraint {
        return GrammarConstraint.fromGrammar(Grammar.json(), model);
    }

    /** Build from a regular expression pattern. */
    static fromRegex(pattern: string, model: Model): GrammarConstraint {
        return GrammarConstraint.fromGrammar(Grammar.fromRegex(pattern), model);
    }

    /** Build from an EBNF grammar string. */
    static fromEbnf(ebnf: string, model: Model): GrammarConstraint {
        return GrammarConstraint.fromGrammar(Grammar.fromEbnf(ebnf), model);
    }

    step(accepted: Uint32Array): Brle {
        if (accepted.length > 0) {
            this._matcher.acceptTokens(accepted);
        }
        return this._matcher.nextTokenLogitMask();
    }
}

/** @internal Wraps a static BRLE mask as a {@link Constraint}. */
export class StaticMaskConstraint implements Constraint {
    private readonly _mask: Brle;
    constructor(mask: Brle) { this._mask = mask; }
    step(_accepted: Uint32Array): Brle { return this._mask; }
}

// ─── Schema — declarative constraint description ────────────────────────

/**
 * Declarative description of a constraint.
 *
 * Pass to `ctx.generate({ ..., constrain: Schema.* })`. The SDK compiles
 * the schema into a {@link GrammarConstraint} internally.
 *
 * ```typescript
 * Schema.jsonSchema('{"type":"object","properties":{...}}')
 * Schema.json()                          // any valid JSON
 * Schema.regex('\\d{3}-\\d{4}')
 * Schema.ebnf('root ::= "hello" | "world"')
 * Schema.grammar(precompiledGrammar)
 * ```
 */
export type Schema =
    | { kind: 'json-schema'; value: string }
    | { kind: 'json' }
    | { kind: 'regex'; value: string }
    | { kind: 'ebnf'; value: string }
    | { kind: 'grammar'; value: Grammar };

export const Schema = {
    /** Constrain to JSON valid against the given JSON Schema string. */
    jsonSchema: (schema: string): Schema => ({ kind: 'json-schema', value: schema }),
    /** Constrain to any valid JSON. */
    json: (): Schema => ({ kind: 'json' }),
    /** Constrain to strings matching the regular expression. */
    regex: (pattern: string): Schema => ({ kind: 'regex', value: pattern }),
    /** Constrain to a custom EBNF grammar. */
    ebnf: (ebnf: string): Schema => ({ kind: 'ebnf', value: ebnf }),
    /** Constrain to a pre-compiled {@link Grammar} (compile once, reuse). */
    grammar: (grammar: Grammar): Schema => ({ kind: 'grammar', value: grammar }),
} as const;

/** @internal Build a Schema into a runtime {@link GrammarConstraint}. */
export function buildSchema(schema: Schema, model: Model): GrammarConstraint {
    switch (schema.kind) {
        case 'json-schema': return GrammarConstraint.fromJsonSchema(schema.value, model);
        case 'json':         return GrammarConstraint.json(model);
        case 'regex':        return GrammarConstraint.fromRegex(schema.value, model);
        case 'ebnf':         return GrammarConstraint.fromEbnf(schema.value, model);
        case 'grammar':      return GrammarConstraint.fromGrammar(schema.value, model);
    }
}
