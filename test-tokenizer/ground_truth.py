from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

output = tokenizer.encode("Hello, my dog is cute", add_special_tokens=False)  # [101, 7592, 1010, 2026, 3899, 2003, 10140, 102]

print(output)