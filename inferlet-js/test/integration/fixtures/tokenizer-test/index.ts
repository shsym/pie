// Tokenizer test inferlet
// Tests tokenization round-trip
import { send, getAutoModel, getArguments } from 'inferlet';

const args = getArguments();
const text = args[0] ?? 'Hello';

const model = getAutoModel();
const tokenizer = model.tokenizer;

// Tokenize
const tokens = tokenizer.tokenize(text);

// Detokenize
const decoded = tokenizer.detokenize(tokens);

// Send result as JSON
send(JSON.stringify({
  original: text,
  tokenCount: tokens.length,
  decoded: decoded,
  match: decoded === text,
}));
