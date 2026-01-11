import unittest
import numpy as np
from pie_worker.batching import Batch, _decode_brle
from pie_worker.runtime import Runtime, RuntimeConfig
from unittest.mock import MagicMock


class TestFlattenedMasks(unittest.TestCase):
    def test_flattened_mask_reconstruction(self):
        # 1. Setup Mock Runtime and Config
        config = RuntimeConfig.from_args(
            hf_repo="dummy", kv_page_size=16, activation_dtype="float16", device="cpu"
        )
        runtime = Runtime.__new__(Runtime)
        runtime.config = config
        runtime.adapters = {}

        # 2. Simulate Input Data
        # Scenario: 2 requests
        # Req 1: 2 tokens, context length 32 (2 KV pages)
        # Req 2: 1 token, context length 16 (1 KV page)

        # Construct BRLE masks
        # Mask 1 (Req 1, Token 1): [True]*33 (attend to 32 context + self)
        mask1_run = [33]
        # Mask 2 (Req 1, Token 2): [True]*34
        mask2_run = [34]
        # Mask 3 (Req 2, Token 1): [True]*17 (attend to 16 context + self)
        mask3_run = [17]

        flattened_masks = mask1_run + mask2_run + mask3_run

        # mask_indptr: points to start of each mask in flattened buffer
        # Tokens: Req1_T1 (len 1), Req1_T2 (len 1), Req2_T1 (len 1)
        # Offsets in flattened_masks: 0, 1, 2, 3
        mask_indptr = [0, 1, 2, 3]

        # Batch parameters
        # qo_indptr: [0, 2, 3] (Req 1 has 2 tokens, Req 2 has 1 token)
        qo_indptr = [0, 2, 3]

        # KV Page setup
        # Req 1: 2 pages (32 tokens) -> kv_page_indptr [0, 2], last_len 16
        # Req 2: 1 page (16 tokens) -> kv_page_indptr [2, 3], last_len 16
        kv_page_indptr = [0, 2, 3]
        kv_last_page_lens = [16, 16]

        # Construct args dictionary (simulating msgpack payload)
        args = {
            "token_ids": [1, 2, 3],  # dummy
            "position_ids": [32, 33, 16],  # dummy
            "kv_page_indices": [0, 1, 2],  # dummy
            "kv_page_indptr": kv_page_indptr,
            "kv_last_page_lens": kv_last_page_lens,
            "qo_indptr": qo_indptr,
            "flattened_masks": flattened_masks,
            "mask_indptr": mask_indptr,
            "single_token_mode": False,
            "adapter_indices": [None, None],
            "adapter_seeds": [None, None],
            "output_token_indices": [[], []],
            "output_token_samplers": [[], []],
            "output_embed_ptrs": [[], []],
            "output_embed_indices": [[], []],
        }

        # 3. Execut _build_batch_from_request
        batch, timing = runtime._build_batch_from_request(args)

        # 4. Verify attention masks
        # Expected:
        # Req 1: 2 rows. Row 0 has 33 ones. Row 1 has 34 ones.
        # Req 2: 1 row. Row 0 has 17 ones.

        self.assertEqual(len(batch.attention_masks), 2)

        # Verify Req 1
        req1_mask_flat = batch.attention_masks[0]
        # Shape should be (2 tokens) * (seq_len)
        # Seq len for Req 1 = 16 * (2-1) + 16 = 32 (context) + 2 (new) = 34? No wait.
        # Logic in runtime:
        # num_pages = 2.
        # seq_len = 16 * (2-1) + 16 = 32.
        # But this seq_len includes the NEW tokens if they were in the KV cache?
        # Actually runtime logic: seq_len = kv_page_size * (num_pages - 1) + kv_last_len
        # If the new tokens are NOT in KV cache yet (which they aren't during prefill/decode usually?),
        # let's trace:
        # Runtime: context_len = seq_len - req_token_count (2) = 30?
        # If we assume 2 pages represents the state *including* the new tokens (which is how pie works often),
        # then total seq len is 32.
        # Context len = 32 - 2 = 30.
        # Then valid lengths: 30 + 0 + 1 = 31, 30 + 1 + 1 = 32.

        # Adjust input test data to match this logic if needed or verifying the mask content matches "decoded"
        # The runtime just calls _decode_brle and places it.
        # _decode_brle([33]) -> [True]*33.
        # _decode_brle([34]) -> [True]*34.

        # Req 1 mask is flattened.
        # Row 1 (first token): Should have first 33 elts True (or whatever decoded gave).
        # Wait, runtime logic: checks if decoded len >= expected len.

        # Let's simple check that the flattened mask contains the right number of True values roughly
        # equivalent to what we put in.

        # Req 1 Mask:
        # Row 0: 33 Trues -> Truncated to expected_len 31 (context 30 + 1)
        # Row 1: 34 Trues -> Truncated to expected_len 32 (context 30 + 2)
        # Total Trues = 31 + 32 = 63.
        self.assertEqual(np.sum(req1_mask_flat), 63)

        # Req 2 Mask:
        # Row 0: 17 Trues -> Truncated to expected_len 16 (context 15 + 1)
        self.assertEqual(np.sum(batch.attention_masks[1]), 16)

        print(
            "Verification Successful: Batched masks reconstructed correctly from flattened inputs."
        )


if __name__ == "__main__":
    unittest.main()
