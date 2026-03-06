//! Unified decoder that muxes chat, reasoning, and tool-use WIT decoders.

use crate::model::Model;
use crate::Result;

// Re-export the instruct sub-interface types for convenience.
pub use crate::pie::instruct::chat::{
    Decoder as ChatDecoder,
    Event as ChatEvent,
};
pub use crate::pie::instruct::tool_use::{
    Decoder as ToolDecoder,
    Event as ToolEvent,
};
pub use crate::pie::instruct::reasoning::{
    Decoder as ReasoningDecoder,
    Event as ReasoningEvent,
};

// Re-export the matcher type from core inference.
pub use crate::pie::core::inference::Matcher;

use crate::pie::instruct::{chat, tool_use, reasoning};

// =============================================================================
// Unified Event
// =============================================================================

/// Unified event emitted by [`Decoder`].
///
/// Merges chat, reasoning, and tool-use decoder outputs into a single enum
/// so callers only need one `match` arm per token.
pub enum Event {
    /// Raw token with no semantic significance (yet).
    Token,
    /// Generated text chunk (outside of reasoning blocks).
    Text(String),
    /// Reasoning/thinking text chunk.
    Thinking(String),
    /// Reasoning block complete (full accumulated thinking text).
    ThinkingDone(String),
    /// A complete tool call was detected: (name, arguments_json).
    ToolCall(String, String),
    /// Generation complete (full accumulated response text).
    Done(String),
}

// =============================================================================
// Decoder
// =============================================================================

/// Tracks whether we are inside a reasoning (thinking) block.
enum DecoderState {
    /// Normal text generation.
    Normal,
    /// Inside a reasoning block — chat deltas are suppressed,
    /// reasoning deltas are forwarded as `Event::Thinking`.
    Reasoning,
}

/// Unified decoder that internally muxes the chat, reasoning, and tool-use
/// WIT decoder resources.
///
/// Created via [`Decoder::new`], then configured with builder methods and
/// fed tokens one batch at a time.
///
/// Event priority:
/// 1. Reasoning transitions (start / complete) override chat deltas.
/// 2. Tool calls override chat deltas.
/// 3. Chat done is always forwarded.
/// 4. Otherwise, chat deltas are forwarded as `Event::Text`.
pub struct Decoder {
    chat: ChatDecoder,
    reasoning: Option<ReasoningDecoder>,
    tools: Option<ToolDecoder>,
    state: DecoderState,
}

impl Decoder {
    /// Create a decoder (chat-only by default).
    ///
    /// Reasoning and tool-use decoders are created lazily when enabled
    /// via [`with_reasoning`](Self::with_reasoning) or
    /// [`with_tool_use`](Self::with_tool_use).
    pub fn new(model: &Model) -> Self {
        Self {
            chat: chat::create_decoder(model),
            reasoning: None,
            tools: None,
            state: DecoderState::Normal,
        }
    }

    /// Enable reasoning (thinking block) decoding.
    pub fn with_reasoning(mut self, model: &Model) -> Self {
        self.reasoning = Some(reasoning::create_decoder(model));
        self
    }

    /// Enable tool-use decoding.
    pub fn with_tool_use(mut self, model: &Model) -> Self {
        self.tools = Some(tool_use::create_decoder(model));
        self
    }

    /// Feed a batch of token IDs and get back a single unified event.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Event> {
        // 1. Reasoning decoder (highest priority for state transitions)
        if let Some(ref mut reasoning) = self.reasoning {
            match reasoning.feed(tokens)? {
                ReasoningEvent::Start => {
                    self.state = DecoderState::Reasoning;
                    let _ = self.chat.feed(tokens);
                    if let Some(ref mut tools) = self.tools {
                        let _ = tools.feed(tokens);
                    }
                    return Ok(Event::Thinking(String::new()));
                }
                ReasoningEvent::Complete(s) => {
                    self.state = DecoderState::Normal;
                    let _ = self.chat.feed(tokens);
                    if let Some(ref mut tools) = self.tools {
                        let _ = tools.feed(tokens);
                    }
                    return Ok(Event::ThinkingDone(s));
                }
                ReasoningEvent::Delta(s) => {
                    if matches!(self.state, DecoderState::Reasoning) && !s.is_empty() {
                        let _ = self.chat.feed(tokens);
                        if let Some(ref mut tools) = self.tools {
                            let _ = tools.feed(tokens);
                        }
                        return Ok(Event::Thinking(s));
                    }
                }
            }
        }

        // 2. Tool decoder
        if let Some(ref mut tools) = self.tools {
            match tools.feed(tokens)? {
                ToolEvent::Call(name_args) => {
                    let _ = self.chat.feed(tokens);
                    return Ok(Event::ToolCall(name_args.0, name_args.1));
                }
                ToolEvent::Start => {}
            }
        }

        // 3. Chat decoder (lowest priority)
        match self.chat.feed(tokens)? {
            ChatEvent::Done(s) => Ok(Event::Done(s)),
            ChatEvent::Delta(s) => {
                if matches!(self.state, DecoderState::Reasoning) {
                    Ok(Event::Token)
                } else {
                    Ok(Event::Text(s))
                }
            }
            ChatEvent::Interrupt(_) => Ok(Event::Token),
        }
    }

    /// Reset all sub-decoders to their initial state.
    pub fn reset(&mut self) {
        self.chat.reset();
        if let Some(ref mut reasoning) = self.reasoning {
            reasoning.reset();
        }
        if let Some(ref mut tools) = self.tools {
            tools.reset();
        }
        self.state = DecoderState::Normal;
    }
}
