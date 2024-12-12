from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

class LocalLM():
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(self.device)

        # Set sampling parameters
        self.params = {
            'temperature': args.temperature,
            'top_p': args.top_p,
            'repetition_penalty': args.repetition_penalty,
            'max_length': args.max_tokens,
            'pad_token_id': self.tokenizer.eos_token_id
        }

        if "qwen" in args.model.lower():
            args.stop_token_ids = [151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654]
        elif "llama" in args.model.lower():
            args.stop_token_ids = [128009, 128001, 128006, 128007, 128008]

    def apply_chat_template(self, text: str, system: str=None) -> str:
        if system is not None and len(system) > 0:
            chat = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        else:
            chat = [{"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def generate_response(self, prompt):
        # Generate text
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        # Generate text
        output = self.model.generate(input_ids, **self.params)
        # Decode response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def forward(self, text: str, system: str=None) -> str:
        prompt = self.apply_chat_template(text, system)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        response = self.generate_response(input_ids)
        return response

    def batch_forward(self, batch_texts: list[str]) -> list[str]:
        batch_prompts = [self.apply_chat_template(text) for text in batch_texts]
        return [self.generate_response(prompt) for prompt in batch_prompts]
    
    def batch_forward_with_sys(self, batch_texts: list[str]) -> list[str]:
        batch_prompts = [self.apply_chat_template(prompt, sys) for prompt, sys in batch_texts]
        return [self.generate_response(prompt) for prompt in batch_prompts]

    def batch_generate(self, batch_prompts):
        return [self.generate_response(prompt) for prompt in batch_prompts]
    
    def reset_seed(self, seed):
        set_seed(seed)
