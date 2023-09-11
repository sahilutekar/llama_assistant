from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class LlamaAssistant:
    def __init__(self, model_name_or_path, model_basename, n_threads=2, n_batch=512, n_gpu_layers=32):
        self.model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
        self.lcpp_llm = Llama(
            model_path=self.model_path,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers
        )

    def generate_response(self, prompt, max_tokens=256, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=150):
        prompt_template = f'''SYSTEM: You are a helpful, respectful, and honest assistant. Always answer as helpfully.

USER: {prompt}

ASSISTANT:
'''
        response = self.lcpp_llm(
            prompt=prompt_template,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            echo=True
        )

        return response["choices"][0]["text"]

if __name__ == "__main__":
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
    model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

    llama_assistant = LlamaAssistant(
        model_name_or_path=model_name_or_path,
        model_basename=model_basename
    )

    prompt = "Write a linear regression in python with code"
    response = llama_assistant.generate_response(prompt)

    print(response)
