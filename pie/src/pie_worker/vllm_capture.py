"""Wireshark-mode capture for vLLM SchedulerOutput + model output.

Dumps the complete request→response data to JSONL for side-by-side
comparison between correct and broken configurations.

Usage: Set PIE_CAPTURE=/tmp/capture.jsonl to enable.
"""
import json
import sys


def capture_scheduler_output(scheduler_output, step_label, capture_file):
    """Dump SchedulerOutput to JSONL."""
    try:
        so = scheduler_output
        record = {
            "step": step_label,
            "new_reqs": [],
            "cached": {},
            "finished": list(so.finished_req_ids),
            "num_sched": dict(so.num_scheduled_tokens),
            "total_sched": so.total_num_scheduled_tokens,
        }

        for nr in so.scheduled_new_reqs:
            record["new_reqs"].append({
                "req_id": nr.req_id,
                "prompt_len": len(nr.prompt_token_ids),
                "prompt_head": nr.prompt_token_ids[:5],
                "prompt_tail": nr.prompt_token_ids[-3:],
                "num_computed": nr.num_computed_tokens,
                "num_blocks": [len(b) for b in nr.block_ids],
                "block_ids_head": [list(b[:5]) for b in nr.block_ids],
            })

        cr = so.scheduled_cached_reqs
        if hasattr(cr, "req_ids") and cr.req_ids:
            all_tid_lens = {}
            all_tid_tails = {}
            if isinstance(cr.all_token_ids, dict):
                all_tid_lens = {k: len(v) for k, v in cr.all_token_ids.items()}
                all_tid_tails = {k: list(v[-3:]) for k, v in cr.all_token_ids.items()}
            record["cached"] = {
                "req_ids": list(cr.req_ids),
                "num_computed": list(cr.num_computed_tokens),
                "num_output": list(cr.num_output_tokens),
                "all_token_ids_lens": all_tid_lens,
                "all_token_ids_tails": all_tid_tails,
                "new_block_ids": [
                    ([list(b[:3]) for b in bi] if bi else None)
                    for bi in (cr.new_block_ids or [])
                ],
            }

        with open(capture_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[CAPTURE-ERR] scheduler: {e}", file=sys.stderr, flush=True)


def capture_model_output(result, req_ids, step_label, capture_file):
    """Dump sampled tokens from model output."""
    try:
        record = {
            "step": step_label,
            "type": "output",
            "req_ids": list(req_ids) if req_ids else [],
        }

        # vLLM v1 ModelRunnerOutput has sampled_token_ids (dict or tensor)
        if hasattr(result, "sampled_token_ids"):
            st = result.sampled_token_ids
            if isinstance(st, dict):
                record["sampled"] = {str(k): list(v) for k, v in st.items()}
            elif hasattr(st, "tolist"):
                record["sampled_list"] = st.tolist()[:10]
            else:
                record["sampled_raw"] = str(st)[:200]
        elif hasattr(result, "sampled_token_ids_cpu"):
            cpu = result.sampled_token_ids_cpu
            if hasattr(cpu, "tolist"):
                record["sampled_cpu"] = cpu.tolist()[:10]
            else:
                record["sampled_cpu_raw"] = str(cpu)[:200]

        # Also capture logprobs if available
        if hasattr(result, "logprobs"):
            record["has_logprobs"] = result.logprobs is not None

        with open(capture_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[CAPTURE-ERR] output: {e}", file=sys.stderr, flush=True)
