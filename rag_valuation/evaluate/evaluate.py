import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Iterator
from rag_valuation.logger import eval_logger
from threading import Thread
from tqdm import tqdm

from rag_valuation.searcher import searcher



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
    lines = []

    with open(file_path, "r") as f:
        for line in f:
            lines.append(json.loads(line))

    sys_prompt = """You are a test taking bot. You simply, and only respond with "(A) correct" or (B) misinformation". Do not provide any other text or explanation. 
Select the multiple-choice option that is appropriate:
Given the statement and choices, output the choice that best labels the statement. 
...If the statement is correct, respond with: "(A) correct"...
...If the statement is misinformation respond with: "(B) misinformation".
Do not provide explanation, do not conversate, simply respond with the correct choice."""
    
    for i in tqdm(range(len(lines)), desc="Generating Responses"):
        line = lines[i]
        # line is a string, but actually is a json
        response = generate(line, model, tokenizer, chat_history=[], system_prompt=sys_prompt)

        # save response to a new file
        with open("rag_valuation/data/climate_fever_rag_responses.txt", "a") as f:
            single_line_response = response.replace("\n", " ")  # Replace newlines with spaces
            f.write(single_line_response + "\n")

    eval_logger.info("Done generating responses.")

def run_with_searcher(lines: list[dict], csv_path: str, embeddings_path: str):
    s = searcher.RagSearcher(csv_path=csv_path, embeddings_path=embeddings_path)

    results = s.query("Your query here")
    print(results)


  

def generate(
    message: str,
    model,
    tokenizer: AutoTokenizer,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 115,
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
        do_sample=False,
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