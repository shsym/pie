"""
Beam search decoding example using inferlet.

This example demonstrates:
- Using beam search for text generation
- Configuring beam size and stop conditions
- Better quality output through multi-sequence exploration

Beam search maintains multiple candidate sequences at each step,
selecting the most likely overall sequences rather than greedily
choosing the best token at each position.
"""

from inferlet import Context, get_auto_model, get_arguments, set_return


def main() -> None:
    # Parse arguments
    args = get_arguments()
    prompt = args.get("prompt", "Explain the LLM decoding process ELI5.")
    max_tokens = int(args.get("max_tokens", 128))
    beam_size = int(args.get("beam_size", 4))
    system_prompt = args.get(
        "system", "You are a helpful, respectful and honest assistant."
    )

    # Load model
    model = get_auto_model()

    # Generate with beam search
    with Context(model) as ctx:
        ctx.system(system_prompt)
        ctx.user(prompt)

        result = ctx.generate_with_beam(
            beam_size=beam_size,
            max_tokens=max_tokens,
        )
        set_return(result)


if __name__ == "__main__":
    main()
