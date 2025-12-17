import { describe, it, expect } from 'vitest';
import { Sampler } from '../sampler.js';

describe('Sampler', () => {
  describe('greedy()', () => {
    it('should create a Multinomial sampler with temperature 0.0', () => {
      const sampler = Sampler.greedy();
      const config = sampler.getConfig();

      expect(config.type).toBe('Multinomial');
      expect(config.temperature).toBe(0.0);
    });
  });

  describe('topP()', () => {
    it('should create a TopP sampler with specified temperature and top_p', () => {
      const sampler = Sampler.topP(0.8, 0.95);
      const config = sampler.getConfig();

      expect(config.type).toBe('TopP');
      expect(config.temperature).toBe(0.8);
      if (config.type === 'TopP') {
        expect(config.top_p).toBe(0.95);
      }
    });

    it('should accept different parameter values', () => {
      const sampler = Sampler.topP(1.0, 0.9);
      const config = sampler.getConfig();

      expect(config.type).toBe('TopP');
      expect(config.temperature).toBe(1.0);
      if (config.type === 'TopP') {
        expect(config.top_p).toBe(0.9);
      }
    });
  });

  describe('topK()', () => {
    it('should create a TopK sampler with specified temperature and top_k', () => {
      const sampler = Sampler.topK(0.7, 50);
      const config = sampler.getConfig();

      expect(config.type).toBe('TopK');
      expect(config.temperature).toBe(0.7);
      if (config.type === 'TopK') {
        expect(config.top_k).toBe(50);
      }
    });

    it('should accept different parameter values', () => {
      const sampler = Sampler.topK(0.5, 10);
      const config = sampler.getConfig();

      expect(config.type).toBe('TopK');
      expect(config.temperature).toBe(0.5);
      if (config.type === 'TopK') {
        expect(config.top_k).toBe(10);
      }
    });
  });

  describe('minP()', () => {
    it('should create a MinP sampler with specified temperature and min_p', () => {
      const sampler = Sampler.minP(0.8, 0.05);
      const config = sampler.getConfig();

      expect(config.type).toBe('MinP');
      expect(config.temperature).toBe(0.8);
      if (config.type === 'MinP') {
        expect(config.min_p).toBe(0.05);
      }
    });

    it('should accept different parameter values', () => {
      const sampler = Sampler.minP(1.0, 0.1);
      const config = sampler.getConfig();

      expect(config.type).toBe('MinP');
      expect(config.temperature).toBe(1.0);
      if (config.type === 'MinP') {
        expect(config.min_p).toBe(0.1);
      }
    });
  });

  describe('topKTopP()', () => {
    it('should create a TopKTopP sampler with specified parameters', () => {
      const sampler = Sampler.topKTopP(0.9, 40, 0.92);
      const config = sampler.getConfig();

      expect(config.type).toBe('TopKTopP');
      expect(config.temperature).toBe(0.9);
      if (config.type === 'TopKTopP') {
        expect(config.top_k).toBe(40);
        expect(config.top_p).toBe(0.92);
      }
    });

    it('should accept different parameter values', () => {
      const sampler = Sampler.topKTopP(0.6, 20, 0.95);
      const config = sampler.getConfig();

      expect(config.type).toBe('TopKTopP');
      expect(config.temperature).toBe(0.6);
      if (config.type === 'TopKTopP') {
        expect(config.top_k).toBe(20);
        expect(config.top_p).toBe(0.95);
      }
    });
  });

  describe('reasoning()', () => {
    it('should create a TopKTopP sampler with preset reasoning parameters', () => {
      const sampler = Sampler.reasoning();
      const config = sampler.getConfig();

      expect(config.type).toBe('TopKTopP');
      expect(config.temperature).toBe(0.6);
      if (config.type === 'TopKTopP') {
        expect(config.top_k).toBe(20);
        expect(config.top_p).toBe(0.95);
      }
    });

    it('should match topKTopP(0.6, 20, 0.95)', () => {
      const reasoning = Sampler.reasoning();
      const manual = Sampler.topKTopP(0.6, 20, 0.95);

      expect(reasoning.getConfig()).toEqual(manual.getConfig());
    });
  });

  describe('toJSON()', () => {
    it('should serialize to the same format as getConfig()', () => {
      const sampler = Sampler.topP(0.8, 0.95);

      expect(sampler.toJSON()).toEqual(sampler.getConfig());
    });

    it('should be serializable to JSON string', () => {
      const sampler = Sampler.topKTopP(0.6, 20, 0.95);
      const json = JSON.stringify(sampler);
      const parsed = JSON.parse(json);

      expect(parsed.type).toBe('TopKTopP');
      expect(parsed.temperature).toBe(0.6);
      expect(parsed.top_k).toBe(20);
      expect(parsed.top_p).toBe(0.95);
    });
  });

  describe('type safety', () => {
    it('should maintain correct types for all sampler variants', () => {
      const greedy = Sampler.greedy();
      const topP = Sampler.topP(0.8, 0.95);
      const topK = Sampler.topK(0.7, 50);
      const minP = Sampler.minP(0.8, 0.05);
      const topKTopP = Sampler.topKTopP(0.9, 40, 0.92);
      const reasoning = Sampler.reasoning();

      expect(greedy.getConfig().type).toBe('Multinomial');
      expect(topP.getConfig().type).toBe('TopP');
      expect(topK.getConfig().type).toBe('TopK');
      expect(minP.getConfig().type).toBe('MinP');
      expect(topKTopP.getConfig().type).toBe('TopKTopP');
      expect(reasoning.getConfig().type).toBe('TopKTopP');
    });
  });
});
