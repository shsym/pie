from functools import partial
import time
import requests
import json
from pathlib import Path


def get_call_generate(backend, host, port, model_path, call_overhead_ms=0):
    if backend == "vllm":
        return partial(call_generate_vllm, url=f"{host}:{port}/v1/completions", model_path=model_path, call_overhead_ms=call_overhead_ms)
    elif backend == "sglang":
        # return partial(call_generate_srt_raw, url=f"{host}:{port}/generate")
        return partial(call_generate_vllm, url=f"{host}:{port}/v1/completions", model_path=model_path)

    else:
        raise ValueError(f"Invalid backend: {backend}")


def call_generate_vllm(prompt, temperature, max_tokens, stop=None, n=1, url=None, model_path=None, call_overhead_ms=0):
    assert url is not None

    data = {
        "model": model_path,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": n,
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200

    # Add call overhead
    time.sleep(call_overhead_ms / 1000)

    # print(res.json()['choices'][0]['text'])

    if n == 1:
        pred = res.json()["choices"][0]["text"]
    else:
        pred = [x["text"] for x in res.json()["choices"]]
    return pred


def call_generate_outlines(
        prompt, temperature, max_tokens, stop=None, regex=None, n=1, url=None
):
    assert url is not None

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "regex": regex,
        "n": n,
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    if n == 1:
        pred = res.json()["text"][0][len(prompt):]
    else:
        pred = [x[len(prompt):] for x in res.json()["text"]]
    return pred


def call_generate_srt_raw(prompt, temperature, max_tokens, stop=None, url=None):
    assert url is not None

    data = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": stop,
        },
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    obj = res.json()
    pred = obj["text"]
    return pred


def append_log(log_file_path: str, data: dict):
    if not log_file_path.endswith('.json'):
        raise ValueError("The log file path must end with '.json'")

    path = Path(log_file_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    logs = []

    if path.is_file() and path.stat().st_size > 0:
        try:
            with path.open('r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    logs = existing_data
        except json.JSONDecodeError:
            print(f"Warning: '{log_file_path}' contains invalid JSON. It will be overwritten.")

    logs.append(data)

    with path.open('w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
