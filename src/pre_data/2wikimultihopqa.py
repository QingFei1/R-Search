import os
import json
import random
import datasets
import argparse
import random
random.seed(66)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/corpus/2wikimultihopqa')
    parser.add_argument('--system_template', default=None)

    args = parser.parse_args()

    data_source = args.local_dir.split("/")[-1]

    # Load train data
    train_data_path = os.path.join(args.local_dir, 'train.json')
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_json_data = json.load(f)

    train_data = []
    for idx, item in enumerate(train_json_data):

        train_data.append({
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": args.system_template},
                {"role": "user","content": item['question']}],
            "ability": "qa_serach",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": item['answer']}},
            "extra_info": {
                "index": idx,
            }
        })

    # Load dev data
    dev_data_path = os.path.join(args.local_dir, 'dev.json')
    with open(dev_data_path, 'r', encoding='utf-8') as f:
        dev_json_data = random.sample(json.load(f),100)

    dev_data = []
    for idx, item in enumerate(dev_json_data):
        dev_data.append({
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": args.system_template},
                {"role": "user","content": item['question']}],
            "ability": "qa_serach",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": item['answer']}},
            "extra_info": {
                "index": idx,
            }
        })

    # Convert to datasets.Dataset format
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(dev_data)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))