import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"


class LocalLLM:

    def __init__(self, device="cuda:0"):
        self.device = device

        print(f"Loading {MODEL_NAME} on {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
        ).to(device)

        self.model.eval()

        print("Model loaded successfully.")

    def generate(
        self,
        system_prompt,
        user_prompt,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
    ):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            generation_args["temperature"] = (
                temperature if temperature is not None else 0.3
            )

        start = time.time()

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **generation_args,
            )

        elapsed = time.time() - start

        generated = output[0][inputs.input_ids.shape[1]:]

        response = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        return {
            "text": response,
            "time": elapsed,
            "tokens": len(generated),
        }
