// inferlet-js/test/wasm/__tests__/kvs.wasm.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { __resetMockState } from '../../mocks/inferlet-core-kvs.js';
import { storeGet, storeSet, storeDelete, storeExists, storeListKeys } from '../../../src/kvs.js';

describe('KVS (WASM mock integration)', () => {
  beforeEach(() => {
    __resetMockState();
  });

  describe('storeSet() and storeGet()', () => {
    it('should store and retrieve a value', () => {
      storeSet('key1', 'value1');
      expect(storeGet('key1')).toBe('value1');
    });

    it('should return undefined for non-existent key', () => {
      expect(storeGet('nonexistent')).toBeUndefined();
    });

    it('should overwrite existing value', () => {
      storeSet('key', 'old');
      storeSet('key', 'new');
      expect(storeGet('key')).toBe('new');
    });

    it('should handle empty string values', () => {
      storeSet('empty', '');
      expect(storeGet('empty')).toBe('');
    });

    it('should handle special characters in keys and values', () => {
      storeSet('key:with:colons', 'value/with/slashes');
      expect(storeGet('key:with:colons')).toBe('value/with/slashes');
    });
  });

  describe('storeDelete()', () => {
    it('should delete a key', () => {
      storeSet('key', 'value');
      storeDelete('key');
      expect(storeGet('key')).toBeUndefined();
    });

    it('should not throw when deleting non-existent key', () => {
      expect(() => storeDelete('nonexistent')).not.toThrow();
    });

    it('should remove key from list after deletion', () => {
      storeSet('toDelete', 'value');
      expect(storeListKeys()).toContain('toDelete');
      storeDelete('toDelete');
      expect(storeListKeys()).not.toContain('toDelete');
    });
  });

  describe('storeExists()', () => {
    it('should return true for existing key', () => {
      storeSet('key', 'value');
      expect(storeExists('key')).toBe(true);
    });

    it('should return false for non-existent key', () => {
      expect(storeExists('nonexistent')).toBe(false);
    });

    it('should return true for key with empty value', () => {
      storeSet('empty', '');
      expect(storeExists('empty')).toBe(true);
    });

    it('should return false after deletion', () => {
      storeSet('key', 'value');
      storeDelete('key');
      expect(storeExists('key')).toBe(false);
    });
  });

  describe('storeListKeys()', () => {
    it('should return empty array when store is empty', () => {
      expect(storeListKeys()).toEqual([]);
    });

    it('should return all keys', () => {
      storeSet('a', '1');
      storeSet('b', '2');
      storeSet('c', '3');
      const keys = storeListKeys();
      expect(keys).toHaveLength(3);
      expect(keys).toContain('a');
      expect(keys).toContain('b');
      expect(keys).toContain('c');
    });

    it('should update when keys are added', () => {
      storeSet('first', '1');
      expect(storeListKeys()).toHaveLength(1);

      storeSet('second', '2');
      expect(storeListKeys()).toHaveLength(2);
    });

    it('should update when keys are deleted', () => {
      storeSet('a', '1');
      storeSet('b', '2');
      expect(storeListKeys()).toHaveLength(2);

      storeDelete('a');
      expect(storeListKeys()).toHaveLength(1);
      expect(storeListKeys()).toEqual(['b']);
    });
  });

  describe('isolation between tests', () => {
    it('first test - sets a value', () => {
      storeSet('isolated', 'test1');
      expect(storeGet('isolated')).toBe('test1');
    });

    it('second test - should not see previous value', () => {
      // Due to beforeEach reset, this should be undefined
      expect(storeGet('isolated')).toBeUndefined();
    });
  });
});
