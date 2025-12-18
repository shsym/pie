// inferlet-js/test/mocks/inferlet-core-runtime.js
// Mock implementation of inferlet:core/runtime

import { Model, DebugQueryResult } from './inferlet-core-common.js';

let mockModels = ['mock-llama-3.2-1b'];
let mockArguments = [];
let mockReturnValue = null;

export function getVersion() {
  return '1.0.0-mock';
}

export function getInstanceId() {
  return 'mock-instance-' + Date.now().toString(36);
}

export function getArguments() {
  return mockArguments;
}

export function setReturn(value) {
  mockReturnValue = value;
}

export function getModel(name) {
  if (mockModels.includes(name)) {
    return new Model(name);
  }
  return undefined;
}

export function getAllModels() {
  return mockModels;
}

export function getAllModelsWithTraits(traits) {
  // Mock: all models have all traits
  return mockModels;
}

export function debugQuery(query) {
  return new DebugQueryResult(`runtime-debug: ${query}`);
}

// Test helpers
export function __setMockModels(models) {
  mockModels = models;
}

export function __setMockArguments(args) {
  mockArguments = args;
}

export function __getMockReturnValue() {
  return mockReturnValue;
}

export function __resetMockState() {
  mockModels = ['mock-llama-3.2-1b'];
  mockArguments = [];
  mockReturnValue = null;
}
