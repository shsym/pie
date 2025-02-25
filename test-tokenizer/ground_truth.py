from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

output = tokenizer.encode("Hello, my dog is cute", add_special_tokens=False)  # [101, 7592, 1010, 2026, 3899, 2003, 10140, 102]

aa =  [139, 915, 176, 660, 280, 175, 495, 229, 475, 923, 84, 383, 577, 295, 258, 529, 385, 765, 817, 250, 36, 141, 289, 570, 591, 841, 706, 607, 316, 831, 392, 868]

decoded = tokenizer.decode(aa)

print(decoded)