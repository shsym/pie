import torch
from transformers import AutoProcessor, TorchAoConfig, AutoTokenizer
from qwen_utils import process_vision_info

from qwen import Qwen2_5_VLForConditionalGeneration
from l4ma import AttentionBuffer, get_rope_index


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
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "./zebra.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],

        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(text)
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

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    max_new_tokens = 32
    logits = None
    past_key_values = None

    token = 0
    output_ids = []

    buffer.clear()

    # input_ids = inputs.input_ids
    # attention_mask = inputs.attention_mask
    # pixel_values = inputs.pixel_values
    # image_grid_thw = inputs.image_grid_thw
    num_input_tokens = len(inputs.input_ids[0])
    position_ids, pos_offset = get_rope_index(model.config,
                                              input_ids=inputs.input_ids,
                                              image_grid_thw=inputs.image_grid_thw,
                                              video_grid_thw=None,
                                              second_per_grid_ts=None
                                              )

    # print(position_ids)
    pos_offset += num_input_tokens - 1
    for i in range(max_new_tokens):

        # prefill
        if i == 0:

            aaa = torch.arange(num_input_tokens, device=position_ids.device)
            attention_mask = create_causal_mask(aaa, num_input_tokens)
            buffer_sink_ids = buffer.allocate(num_input_tokens)
            logits = model(
                input_ids=inputs.input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids
            )


        else:

            position_ids = torch.as_tensor([[pos_offset + i]], device=device)
            aaa = torch.tensor([num_input_tokens + i], device=device)
            attention_mask = create_causal_mask(aaa, num_input_tokens + i)

            buffer_sink_ids = buffer.allocate(1)
            logits = model(
                input_ids=torch.as_tensor([[token]], device=device),
                position_ids=position_ids.view(1, -1).expand(3, 1, 1),
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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="bfloat16", device_map="cuda:0", quantization_config=quantization_config)

    main(model)
