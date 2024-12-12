from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, set_seed

# TODO: parallel
class LocalLM():
    def __init__(self, args) -> None:
        self.args = args
        self.model = LLM(model=args.model, 
                            dtype=args.dtype, 
                            tensor_parallel_size=args.tensor_parallel,
                            gpu_memory_utilization=0.95 if ("70b" in args.model.lower() or "72b" in args.model.lower()) else 0.9,
                            enable_prefix_caching=True,
                            trust_remote_code=True,
                            max_model_len=args.max_tokens, # if "gemma" in args.model.lower() else 8192,
                        )
        
        if "qwen" in args.model.lower():
            args.stop_token_ids = [151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654]
        elif "llama" in args.model.lower():
            args.stop_token_ids = [128009, 128001, 128006, 128007, 128008]
        
        # greedy
        if args.greedy_or_topk == 'greedy':
            self.params = SamplingParams(temperature=0.0, 
                                        top_p=1.0, 
                                        top_k=1, 
                                        repetition_penalty=1.0,max_tokens=args.max_tokens,
                                        stop_token_ids=args.stop_token_ids,
                                        seed=args.seed
                                        )
        elif args.greedy_or_topk == 'topk':
            self.params = SamplingParams(temperature=args.temperature,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty,
                                max_tokens=args.max_tokens,
                                stop_token_ids=args.stop_token_ids,
                                seed=args.seed)
            
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)

    def apply_chat_template(self, text: str, system: str=None) -> str:
        if system is not None and len(system) > 0:
            chat = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        else:
            chat = [{"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def forward(self, text: str, system: str=None) -> str:
        prompt = self.apply_chat_template(text, system)
        response = self.model.generate([prompt], self.params)
        response = response[0].outputs[0].text
        return response

    def batch_forward(self, batch_texts: list[str]) -> list[str]:
        batch_prompts = [self.apply_chat_template(text) for text in batch_texts]
        batch_responses = self.model.generate(batch_prompts, self.params)
        batch_responses = list(map(lambda x: x.outputs[0].text, batch_responses))
        return batch_responses
    
    def batch_forward_with_sys(self, batch_texts: list[str]) -> list[str]:
        batch_prompts = [self.apply_chat_template(prompt, sys) for prompt, sys in batch_texts]
        batch_responses = self.model.generate(batch_prompts, self.params)
        batch_responses = list(map(lambda x: x.outputs[0].text, batch_responses))
        return batch_responses

    def batch_generate(self, batch_prompts):
        batch_responses = self.model.generate(batch_prompts, self.params)
        batch_responses = list(map(lambda x: x.outputs[0].text, batch_responses))
        return batch_responses
    
    def reset_seed(self, seed):
        self.params = SamplingParams(temperature=self.args.temperature,
                                     top_p=self.args.top_p,
                                     repetition_penalty=self.args.repetition_penalty,
                                     max_tokens=self.args.max_tokens,
                                     stop_token_ids=self.args.stop_token_ids,
                                     seed=seed)
        set_seed(seed)

