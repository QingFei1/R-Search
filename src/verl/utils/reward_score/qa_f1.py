import re
import string
import random
import yaml
from openai import OpenAI
from retry import retry
import concurrent
from collections import Counter
with open('scripts/evidence.yaml', 'r') as f:
    config = yaml.safe_load(f)

client_local = OpenAI(
    base_url=config["LOCAL_BASE_URL"],
    api_key=config["LOCAL_API_KEY"], 
)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def F1_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    score = 0.0
    for ground_truth in golden_answers:
        score = max(score, qa_f1_score(prediction, ground_truth))
    return round(score, 2)

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str, pattern=r'<answer>(.*?)</answer>'):
    format_flag = False
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    obs_matches = list(re.finditer(r'<observation>(.*?)</observation>', solution_str, re.DOTALL)) if "answer" not in pattern else []

    if "answer" in pattern:
        answer = matches[-1].group(1).strip()
        if len(matches) <= 1:
            return format_flag, None
        if len(matches) == 2 and answer and answer !="and":
            format_flag = True
        return format_flag, answer

    if len(matches)==1:
        return (True if len(obs_matches) == 1 else False), None
    evidence = matches[-1].group(1).strip()
    if len(obs_matches) >= 2 and len(matches) == 2 and evidence:
        format_flag = True
    elif len(obs_matches) < 2:
        return False, None

    return format_flag, evidence


@retry(((Exception)), delay=1, backoff=2, max_delay=20,jitter=(1, 15))
def call_llm(client, model, prompt,stop=None):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.1,
            top_p=0.9
        )
        response=completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        raise  
    return response

def evidence_eval(evidence, query, golden_answers):
    print("====evidence_eval start===")
    template = config["PROMPT"]
    prompt = template.format(evidence=evidence, query=query).strip()
    
    def call_model(client, model, prompt):
        return call_llm(client, model, prompt)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(call_model, client_local, config["LOCAL_MODEL"], prompt),
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        response = results[0]
 

    print("====evidence_eval===",prompt,"\nllama3:",response,"\nanswer:",golden_answers)
    if response:
        score=F1_check(response, golden_answers)

    return score


def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    ans_format,answer = extract_solution(solution_str=solution_str)
    evid_format,evidence = extract_solution(solution_str=solution_str, pattern=r'<original_evidence>(.*?)</original_evidence>')
    do_print = random.randint(1, 64) == 1
    acc_score=0
    format_score=0
    evidence_score=0
    
    if ans_format:
        format_score+=0.2
    if evid_format:
        format_score+=0.2

    if answer is None:
        acc_score=0
    else:
        acc_score+=F1_check(answer, ground_truth['target'])
    if evidence:
        pattern = r"<\|im_start\|>user\s*(.*?)<\|im_end\|>"
        match = re.search(pattern, solution_str, re.DOTALL)
        query = match.group(1).strip()
        evidence_score=evidence_eval(evidence, query, ground_truth['target'])
    all_score=format_score+acc_score+evidence_score
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer,ans_format}")
        print(f"Extracted evidence: {evidence,evid_format}")
        print(f"reward_score: acc_score-{acc_score},evidence_score-{evidence_score},format_score-{format_score}")
    return all_score

