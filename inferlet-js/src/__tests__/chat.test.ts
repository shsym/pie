import { describe, it, expect, beforeEach } from 'vitest';
import { ChatFormatter, ToolCall } from '../chat.js';

describe('ChatFormatter', () => {
  let formatter: ChatFormatter;

  beforeEach(() => {
    formatter = new ChatFormatter();
  });

  describe('constructor', () => {
    it('should create an empty formatter', () => {
      expect(formatter.hasMessages()).toBe(false);
    });
  });

  describe('system()', () => {
    it('should add a system message', () => {
      formatter.system('You are a helpful assistant.');
      expect(formatter.hasMessages()).toBe(true);
    });
  });

  describe('user()', () => {
    it('should add a user message', () => {
      formatter.user('Hello, how are you?');
      expect(formatter.hasMessages()).toBe(true);
    });
  });

  describe('assistant()', () => {
    it('should add a simple assistant message', () => {
      formatter.assistant('I am doing well, thank you!');
      expect(formatter.hasMessages()).toBe(true);
    });
  });

  describe('assistantResponse()', () => {
    it('should add an assistant message with content only', () => {
      formatter.assistantResponse('Simple response', undefined, undefined);
      expect(formatter.hasMessages()).toBe(true);
    });

    it('should add an assistant message with reasoning', () => {
      formatter.assistantResponse(
        'Let me help you with that.',
        'First, I need to analyze the request.',
        undefined
      );
      expect(formatter.hasMessages()).toBe(true);
    });

    it('should add an assistant message with tool calls', () => {
      const toolCalls: ToolCall[] = [
        {
          name: 'get_weather',
          arguments: { location: 'San Francisco' },
        },
      ];
      formatter.assistantResponse('Checking the weather...', undefined, toolCalls);
      expect(formatter.hasMessages()).toBe(true);
    });

    it('should add an assistant message with both reasoning and tool calls', () => {
      const toolCalls: ToolCall[] = [
        {
          name: 'search',
          arguments: { query: 'weather forecast' },
        },
      ];
      formatter.assistantResponse(
        'Let me search for that.',
        'I need to use the search tool',
        toolCalls
      );
      expect(formatter.hasMessages()).toBe(true);
    });
  });

  describe('tool()', () => {
    it('should add a tool response message', () => {
      formatter.tool('{"temperature": 72, "conditions": "sunny"}');
      expect(formatter.hasMessages()).toBe(true);
    });
  });

  describe('hasMessages()', () => {
    it('should return false for empty formatter', () => {
      expect(formatter.hasMessages()).toBe(false);
    });

    it('should return true after adding a message', () => {
      formatter.user('Hello');
      expect(formatter.hasMessages()).toBe(true);
    });
  });

  describe('clear()', () => {
    it('should clear all messages', () => {
      formatter.system('System message');
      formatter.user('User message');
      formatter.assistant('Assistant message');
      expect(formatter.hasMessages()).toBe(true);

      formatter.clear();
      expect(formatter.hasMessages()).toBe(false);
    });
  });

  describe('render()', () => {
    it('should render a simple conversation', () => {
      formatter.system('You are a helpful assistant.');
      formatter.user('Hello!');
      formatter.assistant('Hi there!');

      const template = '{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}';
      const result = formatter.render(template, false, false);

      expect(result).toContain('system: You are a helpful assistant.');
      expect(result).toContain('user: Hello!');
      expect(result).toContain('assistant: Hi there!');
    });

    it('should pass add_generation_prompt to template', () => {
      formatter.user('Hello');

      const template = '{% if add_generation_prompt %}[GEN]{% endif %}{{ messages[0].content }}';
      const resultWithPrompt = formatter.render(template, true, false);
      const resultWithoutPrompt = formatter.render(template, false, false);

      expect(resultWithPrompt).toBe('[GEN]Hello');
      expect(resultWithoutPrompt).toBe('Hello');
    });

    it('should pass begin_of_sequence to template', () => {
      formatter.user('Hello');

      const template = '{% if begin_of_sequence %}[BOS]{% endif %}{{ messages[0].content }}';
      const resultWithBOS = formatter.render(template, false, true);
      const resultWithoutBOS = formatter.render(template, false, false);

      expect(resultWithBOS).toBe('[BOS]Hello');
      expect(resultWithoutBOS).toBe('Hello');
    });

    it('should handle messages with reasoning_content', () => {
      formatter.assistantResponse('Answer', 'My reasoning', undefined);

      const template = '{% for msg in messages %}{% if msg.reasoning_content %}[Reasoning: {{ msg.reasoning_content }}] {% endif %}{{ msg.content }}{% endfor %}';
      const result = formatter.render(template, false, false);

      expect(result).toContain('[Reasoning: My reasoning]');
      expect(result).toContain('Answer');
    });

    it('should handle messages with tool_calls', () => {
      const toolCalls: ToolCall[] = [
        {
          name: 'calculator',
          arguments: { operation: 'add', a: 5, b: 3 },
        },
      ];
      formatter.assistantResponse('Calculating...', undefined, toolCalls);

      const template = '{% for msg in messages %}{% if msg.tool_calls %}TOOLS: {{ msg.tool_calls | length }}{% endif %}{% endfor %}';
      const result = formatter.render(template, false, false);

      expect(result).toContain('TOOLS: 1');
    });

    it('should render complex multi-turn conversation', () => {
      formatter.system('You are a helpful AI assistant.');
      formatter.user('What is 2+2?');
      formatter.assistantResponse(
        'Let me calculate that.',
        'I need to use the calculator',
        [{ name: 'add', arguments: { a: 2, b: 2 } }]
      );
      formatter.tool('{"result": 4}');
      formatter.assistant('The answer is 4.');

      const template = `{% for msg in messages %}{{ msg.role }}{% if msg.reasoning_content %} (reasoning: {{ msg.reasoning_content }}){% endif %}: {{ msg.content }}
{% endfor %}`;
      const result = formatter.render(template, false, false);

      expect(result).toContain('system: You are a helpful AI assistant.');
      expect(result).toContain('user: What is 2+2?');
      expect(result).toContain('assistant (reasoning: I need to use the calculator): Let me calculate that.');
      expect(result).toContain('tool: {"result": 4}');
      expect(result).toContain('assistant: The answer is 4.');
    });

    it('should handle empty messages list', () => {
      const template = '{% if messages | length > 0 %}HAS MESSAGES{% else %}NO MESSAGES{% endif %}';
      const result = formatter.render(template, false, false);

      expect(result).toBe('NO MESSAGES');
    });

    it('should render with both flags enabled', () => {
      formatter.user('Test');

      const template = '{% if begin_of_sequence %}[BOS]{% endif %}{{ messages[0].content }}{% if add_generation_prompt %}[GEN]{% endif %}';
      const result = formatter.render(template, true, true);

      expect(result).toBe('[BOS]Test[GEN]');
    });
  });

  describe('integration tests', () => {
    it('should support building and clearing conversation history', () => {
      // Build conversation
      formatter.system('System prompt');
      formatter.user('Question 1');
      formatter.assistant('Answer 1');
      expect(formatter.hasMessages()).toBe(true);

      // Clear and rebuild
      formatter.clear();
      expect(formatter.hasMessages()).toBe(false);

      formatter.user('Question 2');
      formatter.assistant('Answer 2');
      expect(formatter.hasMessages()).toBe(true);

      const template = '{% for msg in messages %}{{ msg.role }}\n{% endfor %}';
      const result = formatter.render(template, false, false);

      // Should only contain the second conversation
      expect(result).not.toContain('system');
      expect(result).toContain('user\n');
      expect(result).toContain('assistant\n');
    });

    it('should handle special characters in content', () => {
      formatter.user('Hello "world" with <special> & characters');

      const template = '{{ messages[0].content }}';
      const result = formatter.render(template, false, false);

      // Nunjucks with autoescape: false should preserve special characters
      expect(result).toBe('Hello "world" with <special> & characters');
    });

    it('should support multiple tool calls in one message', () => {
      const toolCalls: ToolCall[] = [
        { name: 'tool1', arguments: { arg: 'value1' } },
        { name: 'tool2', arguments: { arg: 'value2' } },
        { name: 'tool3', arguments: { arg: 'value3' } },
      ];

      formatter.assistantResponse('Using multiple tools', undefined, toolCalls);

      const template = `{% for msg in messages %}{% if msg.tool_calls %}{% for call in msg.tool_calls %}{{ call.name }}:{{ call.arguments.arg }},{% endfor %}{% endif %}{% endfor %}`;
      const result = formatter.render(template, false, false);

      expect(result).toBe('tool1:value1,tool2:value2,tool3:value3,');
    });
  });
});
