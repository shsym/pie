import torch
from transformers import TorchAoConfig, AutoTokenizer

from l4ma import AttentionBuffer
from llama import LlamaForCausalLM


def llama3_format(prompt: str, hint: str | None, system: str = "You are a helpful, respectful and honest assistant."):
    temp = "<|begin_of_text|>"
    temp += f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    temp += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
    temp += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

    if hint:
        temp += hint

    return temp


def create_causal_mask(position_ids, ctx_len):
    # (batch, num_hd, q_len, head_dim) * (batch, num_hd, head_dim, ctx_len)
    #  (batch, num_hd, q_len, ctx_len) ->

    attn_mask = position_ids[:, None] < torch.arange(ctx_len, device=position_ids.device)[None, :]
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
    # hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    return attn_mask


# @torch.inference_mode()
def main(model):
    # default processor
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    text = llama3_format("Explain what Poodle is.", None)
    device = "cuda:0"

    buffer = AttentionBuffer(
        num_batch=1,
        capacity=1024,
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        dtype=torch.bfloat16,
        device=device
    )

    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    num_input_tokens = len(input_ids[0])

    position_ids = torch.arange(num_input_tokens, device=device).unsqueeze(0)

    max_new_tokens = 32

    token = 0
    output_ids = []

    buffer.clear()

    # input_ids = inputs.input_ids
    # attention_mask = inputs.attention_mask
    # pixel_values = inputs.pixel_values
    # image_grid_thw = inputs.image_grid_thw

    # print(position_ids)

    pos_offset = num_input_tokens - 1
    for i in range(max_new_tokens):

        # prefill
        if i == 0:

            aaa = torch.arange(num_input_tokens, device=position_ids.device)
            attention_mask = create_causal_mask(aaa, num_input_tokens)
            buffer_sink_ids = buffer.allocate(num_input_tokens)
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids
            )


        else:

            position_ids = torch.as_tensor([[pos_offset + i]], device=device)
            aaa = torch.tensor([pos_offset + i], device=device)
            attention_mask = create_causal_mask(aaa, num_input_tokens + i)

            buffer_sink_ids = buffer.allocate(1)
            logits = model(
                input_ids=torch.as_tensor([[token]], device=device),
                position_ids=position_ids.view(1, -1),
                attention_mask=attention_mask,
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids
            )

        last_token_logits = logits[0, -1, :]
        token = int(torch.argmax(last_token_logits))
        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            break

        output_text = tokenizer.decode(output_ids, skip_special_tokens=True,
                                       spaces_between_special_tokens=False, )
        print(output_text)


if __name__ == "__main__":
    # default: Load the model on the available device(s)
    quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", torch_dtype="bfloat16", device_map="cuda:0", quantization_config=quantization_config)

    main(model)
