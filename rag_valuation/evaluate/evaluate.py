import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Iterator
from rag_valuation.logger import eval_logger
from threading import Thread



MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))


def run(lines: list[dict]):
    eval_logger.info("Evaluating RAG model on generated responses...")
    eval_logger.info(f"Number of responses to generate: {len(lines)}")
    
    if not torch.cuda.is_available():
        #  error, only support gpu for now
        raise NotImplementedError("Only support gpu for now.")
        sys.exit()
    if torch.cuda.is_available():
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.use_default_system_prompt = False

    #  for each question in data/climate_fever_rag_contexts.jsonl, generate a response
    #  ... save the response in a new file

    file_path = "rag_valuation/data/climate_fever_rag_contexts.jsonl"
    with open(file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        response = generate(line, model, tokenizer, chat_history=[], system_prompt=None)
        print(response)

        # save response to a new file
        with open("rag_valuation/data/climate_fever_rag_responses.jsonl", "a") as f:
            f.write(response + "\n")

        
        if i > 10:
            break





def generate(
    message: str,
    model,
    tokenizer: AutoTokenizer,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message['question']})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        eval_logger.warning(
            f"Input length exceeds {MAX_INPUT_TOKEN_LENGTH}, truncating to {MAX_INPUT_TOKEN_LENGTH} tokens."
        )

    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Wait for the thread to finish
    t.join()

    # Capture the generated text
    full_response = "".join(streamer)

    return full_response