import nunjucks from 'nunjucks';

// Configure nunjucks to not autoescape (similar to minijinja behavior)
const env = new nunjucks.Environment(null, {
  autoescape: false,
});

/**
 * Represents a single tool call.
 */
export interface ToolCall {
  name: string;
  arguments: any; // JSON value, similar to serde_json::Value
}

/**
 * Represents a single message in the conversation history.
 * Internal structure, not exported.
 */
interface Message {
  role: string;
  content: string;
  reasoning_content?: string;
  tool_calls?: ToolCall[];
}

/**
 * ChatFormatter - A class for formatting chat messages using templates.
 * This mirrors the Rust ChatFormatter API.
 */
export class ChatFormatter {
  private messages: Message[] = [];

  /**
   * Creates a new ChatFormatter instance.
   */
  constructor() {
    this.messages = [];
  }

  /**
   * Adds a system message to the conversation.
   * @param content The content of the system message
   */
  system(content: string): void {
    this.messages.push({
      role: 'system',
      content: content,
    });
  }

  /**
   * Adds a user message to the conversation.
   * @param content The content of the user message
   */
  user(content: string): void {
    this.messages.push({
      role: 'user',
      content: content,
    });
  }

  /**
   * Adds a simple assistant message with only text content.
   * @param content The content of the assistant message
   */
  assistant(content: string): void {
    this.assistantResponse(content, undefined, undefined);
  }

  /**
   * Adds a comprehensive assistant response, optionally including reasoning and tool calls.
   * @param content The main content of the assistant message
   * @param reasoning Optional reasoning content
   * @param toolCalls Optional array of tool calls
   */
  assistantResponse(
    content: string,
    reasoning?: string,
    toolCalls?: ToolCall[]
  ): void {
    const message: Message = {
      role: 'assistant',
      content: content,
    };

    if (reasoning !== undefined) {
      message.reasoning_content = reasoning;
    }

    if (toolCalls !== undefined) {
      message.tool_calls = toolCalls;
    }

    this.messages.push(message);
  }

  /**
   * Adds a tool response message to the conversation.
   * @param content The content of the tool response
   */
  tool(content: string): void {
    this.messages.push({
      role: 'tool',
      content: content,
    });
  }

  /**
   * Checks if there are any messages in the conversation.
   * @returns true if there are messages, false otherwise
   */
  hasMessages(): boolean {
    return this.messages.length > 0;
  }

  /**
   * Clears all messages from the conversation.
   */
  clear(): void {
    this.messages = [];
  }

  /**
   * Renders the entire conversation into a single formatted string prompt.
   * @param template The Nunjucks template string
   * @param addGenerationPrompt Whether to add a generation prompt
   * @param beginOfSequence Whether to include begin of sequence token
   * @returns The rendered conversation string
   */
  render(
    template: string,
    addGenerationPrompt: boolean,
    beginOfSequence: boolean
  ): string {
    return env.renderString(template, {
      messages: this.messages,
      add_generation_prompt: addGenerationPrompt,
      begin_of_sequence: beginOfSequence,
    });
  }
}
