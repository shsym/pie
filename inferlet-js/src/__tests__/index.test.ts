import { describe, it, expect } from 'vitest';
import { VERSION, getVersion } from '../index.js';

describe('inferlet-js', () => {
  it('should export VERSION constant', () => {
    expect(VERSION).toBe('0.1.0');
  });

  it('should return version from getVersion()', () => {
    expect(getVersion()).toBe('0.1.0');
  });

  it('should return same value from VERSION and getVersion()', () => {
    expect(getVersion()).toBe(VERSION);
  });
});
