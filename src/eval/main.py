import re
import time
import json
import os
from tqdm import tqdm
import transformers
import requests
from vllm import LLM, SamplingParams
import yaml
import argparse
from utils import acc_score, f1_scorer, compute_exact,seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='R-Search-7b-grpo', help="Specify the model to use for inference")
parser.add_argument('--retrieve_top_k', type=int, default=5, help="Number of top documents to retrieve for each query")
parser.add_argument('--dataset', type=str, choices=["2wikimultihopqa", "hotpotqa", "musique","nq","popqa","triviaqa","bamboogle"], default='2wikimultihopqa', help="Dataset to evaluate on")
parser.add_argument('--method', type=str, default="R-Search", choices=["R-Search","base_retri","base_wo_retri"], help="Method for question answering")
parser.add_argument('--resume_path', type=str, default="", help="Path to checkpoint file to resume generation from")
parser.add_argument('--temperature', type=float, default=0.1, help="Sampling temperature for generation")
parser.add_argument('--max_tokens', type=int, default=1024, help="Max_tokens parameter for generation")
parser.add_argument('--top_p', type=float, default=0.9, help="Top-p sampling parameter for generation")
parser.add_argument('--gpu_memory_utilization', type=float, default=0.38, help="Fraction of GPU memory to use")

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def base_retri(question):
    template = config["prompt"]["base_retri"]
    prompt = template.format(refs=search(question), question=question)

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
    answer=call_vllm(prompt)[0].outputs[0].text
    return answer



def base_wo_retri(question):
    template = config["prompt"]["base_wo_retri"]
    prompt = template.format(question=question)

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
    answer=call_vllm(prompt)[0].outputs[0].text
    return answer


def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    return None

def search(query: str):
    payload = {"queries": [query], "topk": args.retrieve_top_k, "return_scores": True}
    results = requests.post(retri_url, json=payload).json()['result']

    format_reference = ''
    for idx, doc_item in enumerate(results[0]):
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"(Title: {title}) {text}\n"
    return format_reference



def R_Search(question):
    thought=""
    answer=""
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_template},
             {"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt = system_template + '\n' + question
    print('\n################# [Start Reasoning + Searching] ##################\n')

    cnt = 0
    while True:
        outputs = call_vllm(prompt, stop=["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"])
        generation_result = outputs[0].outputs[0]
        if not generation_result.stop_reason:
            response=generation_result.text.strip()
            thought += response
            matches = list(re.finditer(r'<answer>(.*?)</answer>', response, re.DOTALL))
            if matches:
                answer=matches[-1].group(1).strip()
            else:
                answer=response
            print("Final Output:\n", generation_result.text.strip())
            break
        output_text = generation_result.text.strip()
        tmp_query = get_query(output_text)
        search_results = search(tmp_query) if tmp_query else ''
        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        thought += search_text
        cnt += 1

        print(f"Iteration {cnt}:\n", prompt)
        # if cnt>4:
        #     break
    return answer,thought


def call_vllm(prompt,stop=None):
    sampling_params = SamplingParams(max_tokens=args.max_tokens,temperature=args.temperature,top_p=args.top_p,stop=stop,include_stop_str_in_output=True)
    response = llm.generate(prompt, sampling_params)
    return response

if __name__ == "__main__":
    seed_everything(44)
    args = parser.parse_args()
    if args.dataset not in ["2wikimultihopqa", "hotpotqa", "musique"]:
        retri_url=f'{config["search"]["wiki-18_url"]}_wiki-18'
    else:
        retri_url=f'{config["search"][f"{args.dataset}_url"]}_{args.dataset}'
    formatted_time = time.strftime("%Y%m%d-%H%M%S")
    with open(f"../../data/eval/{args.dataset}/test.jsonl", encoding="utf-8") as fin:
        qa_data = [json.loads(f) for f in fin]

    save_path = f"output/{args.dataset}/{args.method}/{args.model}"
    os.makedirs(save_path, exist_ok=True)

    system_template=config["prompt"][args.method]
    curr_search_template = '\n\n{output_text}<observation>{search_results}</observation>\n\n'


    all_result = []

    if args.resume_path:
        with open(args.resume_path, "r", encoding="utf-8") as fin:
            resume_data = [json.loads(i) for i in fin.readlines()]
            all_result = resume_data
            filepath = args.resume_path
    else:
        resume_data = []
        filepath = (
            f"{save_path}/topk-{args.retrieve_top_k}_{formatted_time}.jsonl"
        )
    last_id = len(resume_data)
    model_path = config["model"][args.model]
    llm = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True, dtype='bfloat16', gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    for idx in tqdm(range(last_id, len(qa_data))):
        cb = qa_data[idx]
        question = cb["question"].strip()
        if question[-1] != "?":
            question += "?"
        if "R-Search" in args.method:
            output = R_Search(question)
        elif args.method == "base_retri":
            output = base_retri(question)
        elif args.method == "base_wo_retri":
            output = base_wo_retri(question)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        if output:
            if "R-Search" in args.method:
                result = {
                    "id": idx,
                    "question": cb["question"],
                    "answer": cb["golden_answers"],
                    "output": output[0],
                    "thought": output[1]
                }
            else:
                result = {
                    "id": idx,
                    "question": cb["question"],
                    "answer": cb["golden_answers"],
                    "output": output
                }
            all_result.append(result)
            with open(filepath, "a", buffering=1) as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    predictions = [data["output"] for data in all_result]
    answers = [data["answer"] for data in all_result]
    
    eval_result = {"Acc": acc_score(predictions, answers), "F1": f1_scorer(predictions, answers), "EM": compute_exact(predictions, answers)}
    
    if eval_result:
        with open(filepath, "a", buffering=1) as fout:
            fout.write(json.dumps(eval_result, ensure_ascii=False) + "\n")