// Basic echo test inferlet
// Simply echoes the command-line arguments back via send()
import { send, getArguments } from 'inferlet';

const args = getArguments();

if (args.length === 0) {
  send('no arguments');
} else {
  send(args.join(' '));
}
