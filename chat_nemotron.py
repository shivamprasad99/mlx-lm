import mlx.core as mx
from mlx_lm import load

model_path = "/Users/shivam/Desktop/mlx-models/Nemotron-Labs-Diffusion-3B-4bit"
adapter_path = model_path + "/linear_spec_lora"

model, tokenizer = load(model_path)
model.load_linear_spec_lora(adapter_path)

messages = [{"role": "system", "content": "You are a helpful assistant."}]
print("Nemotron self-spec chat. Type quit to exit.")

while True:
    user = input("\nYou: ")
    if user.strip().lower() in ("quit", "exit"):
        break

    messages.append({"role": "user", "content": user})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = mx.array([tokenizer.encode(prompt, add_special_tokens=False)])

    generated_tokens = []
    printed = ""
    stats = None

    print("\nAssistant: ", end="", flush=True)
    for chunk, stats in model.self_spec_stream_generate(
        prompt_ids,
        max_new_tokens=1024,
        block_length=4,
        draft_steps=1,
        use_adapter=True,
        eos_token_id=model.args.eos_token_id,
        profile=True,
    ):
        mx.eval(chunk)
        chunk_tokens = chunk.tolist()[0]
        if model.args.eos_token_id is not None:
            chunk_tokens = [t for t in chunk_tokens if t != model.args.eos_token_id]
        generated_tokens.extend(chunk_tokens)

        reply_so_far = tokenizer.decode(generated_tokens)
        delta = reply_so_far[len(printed) :]
        if delta:
            print(delta, end="", flush=True)
            printed = reply_so_far

    reply = tokenizer.decode(generated_tokens).strip()
    print()
    if stats is not None:
        print(
            "[accepted={} drafted={} acceptance={:.2f}]".format(
                stats["accepted_tokens"],
                stats["drafted_tokens"],
                stats["acceptance_rate"],
            )
        )

    messages.append({"role": "assistant", "content": reply})
