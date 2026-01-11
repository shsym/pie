import unittest
import random


class TestComplexMaskFlattening(unittest.TestCase):
    def test_lossless_flattening(self):
        random.seed(42)

        # 1. Generate Complex 3D Data
        # A batch can have many requests
        num_requests = 100
        original_3d: list[list[list[int]]] = []

        # Track expected flattened structures
        expected_flattened_masks: list[int] = []
        expected_mask_indptr: list[int] = [0]
        expected_qo_indptr: list[int] = [0]

        current_mask_ptr = 0
        current_token_count = 0

        print(f"Generating optimized data structure for {num_requests} requests...")

        for req_idx in range(num_requests):
            # Each request can have a variable number of tokens (new tokens during decode or prefill)
            num_tokens = random.randint(1, 50)
            req_masks = []

            for token_idx in range(num_tokens):
                # Each token has a BRLE mask. The length of this BRLE mask varies wildly
                # based on context length and complexity of the mask.
                # Simulate sparse to dense variations.
                brle_len = random.randint(1, 200)
                brle_data = [random.randint(0, 10000) for _ in range(brle_len)]

                req_masks.append(brle_data)

                # Flattening Logic (simulation of Rust side)
                # Concatenate the BRLE data
                expected_flattened_masks.extend(brle_data)

                # Update the pointer to the end of this token's data
                current_mask_ptr += len(brle_data)
                expected_mask_indptr.append(current_mask_ptr)

            original_3d.append(req_masks)

            # Update the pointer to the end of this request's tokens
            current_token_count += num_tokens
            expected_qo_indptr.append(current_token_count)

        print(f"Data Generation Complete.")
        print(f"  Total Tokens: {current_token_count}")
        print(f"  Total Flattened Elements: {len(expected_flattened_masks)}")
        print(f"  Mask Index Pointers: {len(expected_mask_indptr)}")

        # 2. Reconstruct 3D Data (Simulation of Python/Kernel side)
        # This proves we can recover the exact [Request][Token][BRLE] structure
        reconstructed_3d: list[list[list[int]]] = []

        # We need a global cursor for tokens to index into mask_indptr
        # This cursor increases monotonically as we process tokens across all requests
        global_token_cursor = 0

        # Iterate over requests using qo_indptr
        # qo_indptr tells us which tokens belong to which request
        for req_idx in range(len(expected_qo_indptr) - 1):
            req_start_token = expected_qo_indptr[req_idx]
            req_end_token = expected_qo_indptr[req_idx + 1]
            num_tokens_in_req = req_end_token - req_start_token

            req_reconstructed = []

            for _ in range(num_tokens_in_req):
                # For each token, use the global cursor to find its specific BRLE data range
                # in the massive flattened array via mask_indptr
                mask_start = expected_mask_indptr[global_token_cursor]
                mask_end = expected_mask_indptr[global_token_cursor + 1]

                brle = expected_flattened_masks[mask_start:mask_end]
                req_reconstructed.append(brle)

                global_token_cursor += 1

            reconstructed_3d.append(req_reconstructed)

        # 3. Validation
        print("Validating reconstruction...")
        self.assertEqual(
            len(original_3d), len(reconstructed_3d), "Request count mismatch"
        )

        for i in range(len(original_3d)):
            self.assertEqual(
                len(original_3d[i]),
                len(reconstructed_3d[i]),
                f"Request {i} token count mismatch",
            )
            for j in range(len(original_3d[i])):
                self.assertEqual(
                    original_3d[i][j],
                    reconstructed_3d[i][j],
                    f"Request {i} Token {j} mask content mismatch",
                )

        print(
            f"\nSUCCESS: Verified lossless reconstruction of fully variable 3D mask structure."
        )


if __name__ == "__main__":
    unittest.main()
