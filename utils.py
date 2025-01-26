from statistics import mean
from torch.utils.data import Dataset
import multiprocessing
import json
import numpy as np
import torch
import re
import random
import time
import datetime
from openai import OpenAI, OpenAIError
from collections import Counter


def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    #keys = d(keys)
    return dict(keys)


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


def call_model(prompt, retries=3):
    client = OpenAI(
        api_key="API key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            return response.choices[0].message.content.strip()

        except OpenAIError as e:
            if attempt < retries - 1:
                print(f"Request failed, retrying... Error: {e}")
                time.sleep(30)
            else:
                return f"API request failed: {e}"


def call_model2(prompt, retries=3):
    client = OpenAI(
        api_key="API key",
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="glm-4-plus",
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
            )

            return completion.choices[0].message.content.strip()

        except OpenAIError as e:
            if attempt < retries - 1:
                print(f"Request failed, retrying... Error: {e}")
                time.sleep(30)
            else:
                return f"API request failed: {e}"


def call_model3(prompt, retries=3):
    client = OpenAI(
        api_key="API key",
        base_url="https://api.llama-api.com"
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama3.1-70b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            time.sleep(2)
            return response.choices[0].message.content.strip()

        except OpenAIError as e:
            if attempt < retries - 1:
                print(f"Request failed, retrying... Error: {e}")
                time.sleep(30)
            else:
                return f"API request failed: {e}"


def get_judgment(prompt):
    client = OpenAI(
        api_key="API key",
        base_url="url"
    )
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message.content.strip()


def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset == "math":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["problem"].strip())
                answers.append(json_res["answer"].strip())

    elif args.dataset == "commonsensqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])

    elif args.dataset in ("mmlu_chemistry", "mmlu_elec", "mmlu_ml", "mmlu_physics"):
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                question_text = json_res["question"].strip()
                choices_text = json_res["choices"]
                full_question = f'"{question_text}" "choices": {choices_text}'
                questions.append(full_question)
                answers.append(json_res["answer"])

    elif args.dataset == "strategyqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "1"
                else:
                    a = "0"
                questions.append(q)
                answers.append(a)

    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers


# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output


def setup_data_loader(args):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             batch_size=args.minibatch_size,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True)

    return dataloader


def check_consistency(ans1, ans2):
    if ans1 == ans2:
        ans = ans1
    else:
        prompt = f"ans1: {ans1}; ans2: {ans2}. \n ans1 and ans2 are answers to the same question. Please determine whether these two answers are consistent. Just provide the judgment: TRUE or FALSE."
        consistency_check_result = get_judgment(prompt)
        if "TRUE" in consistency_check_result:
            ans = ans1
        else:
            ans = ans2
    return ans


def rebuild_answers(answers):
    result = []

    for i in range(len(answers)):
        consistent_found = False

        for j in range(len(result)):
            if check_consistency(result[j], answers[i]) == result[j]:
                result.append(result[j])
                consistent_found = True
                break

        if not consistent_found:
            result.append(answers[i])

    return result


def extract_answer(args, answer):
    if args.dataset in ("commonsensqa"):
        match = re.search(r"(?:final answer is|the answer is|closest option is)[^A-E]*([A-E])(?:[^a-zA-Z]|$)", answer)
        if match:
            return match.group(1)
    elif args.dataset in ("gsm8k", "strategyqa", "mmlu_chemistry", "mmlu_elec" , "mmlu_ml", "mmlu_physics"):
        match = re.search(r"(?:final answer is|the answer is|closest option is)[^\d]*(\d+)(?:[^a-zA-Z]|$)", answer)
        if match:
            return match.group(1)
    elif args.dataset == "math":
        match = re.search(r"(?:final answer is)\s*(.*)", answer, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_most_frequent_answer(args, history):
    answers = [extract_answer(args, entry) for entry in history]
    answer_counter = Counter(answers)

    if answer_counter:
        most_frequent_answer = answer_counter.most_common(1)[0][0]
        return most_frequent_answer
    else:
        return answers[-1] if answers else None

