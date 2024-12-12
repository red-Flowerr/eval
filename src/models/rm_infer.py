from vllm import LLM
from transformers import AutoTokenizer, set_seed
import torch
from tqdm import tqdm

class LocalLM():
    def __init__(self, args) -> None:
        self.args = args
        self.model = LLM(args.model_name,
                         dtype=args.dtype,
                         gpu_memory_utilization=0.95,
                         tensor_parallel_size=args.tensor_parallel,
                         trust_remote_code=True,
                        )  
        self.tokenizer = AutoTokenizer.from_pretrained(args.pt_model_name, trust_remote_code=True)


    def apply_chat_template(self, prompt: str, responses: list) -> str:
        result = []
        for response in tqdm(responses, desc='Add Template'):
            chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
            # 截断
            token_prompt = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.args.max_token)
            truncated_prompt = self.tokenizer.decode(token_prompt["input_ids"][0], skip_special_tokens=True)
            result.append(truncated_prompt)
        return result
    
    def batch_generate(self, batch_prompts):
        # rm_rewards = []
        # for prompt in batch_prompts:
        outputs = self.model.encode(batch_prompts)
        embeddings = [output.outputs.embedding[-1][0] for output in outputs]
        rm_rewards = torch.tensor(embeddings)
        rm_rewards = torch.sigmoid(rm_rewards)
        print(rm_rewards.tolist())
        return rm_rewards.tolist()
    
    def reset_seed(self, seed):
        
        set_seed(seed)

