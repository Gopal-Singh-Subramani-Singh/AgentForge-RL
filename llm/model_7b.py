import time
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"


class LocalLLM7B:

    def __init__(self, device="cuda:0"):
        self.device = device

        print(f"Loading {MODEL_NAME} in 4-bit on {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map={"": 0},
        )

        self.model.eval()

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free, total = torch.cuda.mem_get_info()

        print("Model loaded successfully.")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
        print(f"Free: {free / 1024**3:.2f} GB")
        print(f"Total: {total / 1024**3:.2f} GB")

    def generate(
        self,
        system_prompt,
        user_prompt,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
