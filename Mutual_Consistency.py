import argparse
from utils import *
import time
from transformers import GPT2Tokenizer
from prompt_mutual import prompt_template


def get_prompts(args):
    if args.dataset == 'gsm8k':
        prompt = prompt_template['prompt_number']
    elif args.dataset == 'math':
        prompt = prompt_template['prompt_math']
    elif args.dataset == 'commonsensqa':
        prompt = prompt_template['prompt_cs']
    elif args.dataset == 'strategyqa':
        prompt = prompt_template['prompt_stQA']
    elif args.dataset in ("mmlu_chemistry", "mmlu_elec", "mmlu_ml", "mmlu_physics"):
        prompt = prompt_template['prompt_mmlu']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return prompt


def expert_a(question, num_responses=3):
    args = parse_arguments()
    prompt = get_prompts(args)
    prompt_a = f"""question:{question}\n{prompt}"""   
    responses_a = []
    all_answers_a = []
    for _ in range(num_responses):
        response_a = call_model(prompt_a)
        responses_a.append(response_a)

        answer_a = extract_answer(args, response_a)
        all_answers_a.append(answer_a)

    return prompt_a, responses_a, all_answers_a


def expert_b(question, num_responses=3):
    args = parse_arguments()
    prompt = get_prompts(args)
    prompt_b = f"""question:{question}\n{prompt}"""   
    responses_b = []
    all_answers_b = []
    for _ in range(num_responses):
        response_b = call_model2(prompt_b)

        responses_b.append(response_b)

        answer_b = extract_answer(args, response_b)
        all_answers_b.append(answer_b)

    return prompt_b, responses_b, all_answers_b


# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def interactive(question, max_iterations=7):
    """
    Experts interact iteratively until a consistent answer is obtained.
    """
    args = parse_arguments()
    history = []
    all_answer = []
    iteration = 0
    pred = None
    start_time = time.time()

    # Initialize an empty string to accumulate all prompt strings
    total_prompts = ""
    total_solution = ""

    while iteration < max_iterations:
        iteration += 1
        print(f"\n******************* Iteration {iteration} ********************")

        # Expert A answer the question.
        prompt_a, responses_a, all_answers_a = expert_a(question)
        print(f"********* Expert A's Prompt *******\n{prompt_a}")
        print(f"********* Expert A's Responses *******\n{responses_a}")
        print(f"********* Expert A's Answers *******\n{all_answers_a}")

        # Append the current prompt_a and solution to concatenated_prompts
        total_prompts += prompt_a + "\n"

        total_solution += ", ".join(map(str, responses_a)) + "\n"

        # Expert B answer the question
        prompt_b, responses_b, all_answers_b = expert_b(question)
        print(f"********* Expert B's Prompt *******\n{prompt_b}")
        print(f"********* Expert B's Responses *******\n{responses_b}")
        print(f"********* Expert B's Answers *******\n{all_answers_b}")

        # Append the current prompt_b and solution to concatenated_prompts
        total_prompts += prompt_b + "\n"
        # 将 responses_b 列表转换为字符串，并加上换行符
        total_solution += ", ".join(map(str, responses_b)) + "\n"

        # Append both expert's strategies to history
        responses_a_str = ", ".join(map(str, responses_a))
        responses_b_str = ", ".join(map(str, responses_b))
        history.append(f"Expert A's strategy:\n{responses_a_str}")
        history.append(f"Expert B's strategy:\n{responses_b_str}")

        # Combine the answers from both experts
        combined_ans = all_answers_a + all_answers_b

        if args.dataset == 'math':
            combined_answers = rebuild_answers(combined_ans)
            all_ans_a = combined_answers[:len(all_answers_a)]
            all_ans_b = combined_answers[len(all_answers_a):]
        else:
            combined_answers = combined_ans
            all_ans_a = all_answers_a
            all_ans_b = all_answers_b

        all_answer.append(combined_answers)
        common_answers = set(all_ans_a).intersection(all_ans_b)

        # Case (1): Only one common answer, use it as pred
        if len(common_answers) == 1:
            pred = common_answers.pop()  # Get the single common answer
            break

        # Case (2): Multiple common answers, select the one with the highest count
        elif len(common_answers) > 1:
            answer_counts = {ans: combined_answers.count(ans) for ans in common_answers}
            max_count = max(answer_counts.values())
            most_frequent_answers = [ans for ans, count in answer_counts.items() if count == max_count]

            # Case (3): If there's a tie, continue to next iteration
            if len(most_frequent_answers) == 1:
                pred = most_frequent_answers[0]
                break

            # If there's a tie, move to the next iteration
            print(f"Tie between answers: {most_frequent_answers}, moving to next iteration.")

        # Case (4): No common answers, move to next iteration
        else:
            print(f"No common answers, moving to next iteration.")

    # If maximum iterations are reached and no consistent answer, use the most frequent answer overall
    if pred is None:
        pred = extract_most_frequent_answer(args, history)
        print(f"\nMaximum iterations {max_iterations} reached, ending loop.")


    # End timing
    end_time = time.time()
    time_spent = end_time - start_time

    # Token statistics
    total_input_tokens = len(tokenizer.encode(total_prompts)) if total_prompts else 0
    total_output_tokens = len(tokenizer.encode(total_solution)) if total_solution else 0
    total_tokens = total_input_tokens + total_output_tokens

    return pred, all_answer, time_spent, total_input_tokens, total_output_tokens, total_tokens


def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    total = 0
    correct_list = []
    # Initialize accumulators for time and token statistics
    total_time_spent = 0
    total_input_tokens_accum = 0
    total_output_tokens_accum = 0
    total_tokens_accum = 0

    with open(args.output_dir, "a") as wp:

        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1:
                continue

            output_line = {}
            
            print('*************************')
            print("{}st data".format(i+1))

            # Prepare question template ...
            x, y = data
            x = x[0].strip()
            if args.dataset in ("mmlu_chemistry", "mmlu_elec", "mmlu_ml", "mmlu_physics"):
                y = str(y.item())
            else:
                y = y[0].strip()

            output_line["question"] = x
            output_line["gold_ans"] = y

            ans, all_answer, time_spent, total_input_tokens, total_output_tokens, total_tokens = interactive(x)

            # Accumulate time and token statistics
            total_time_spent += time_spent
            total_input_tokens_accum += total_input_tokens
            total_output_tokens_accum += total_output_tokens
            total_tokens_accum += total_tokens
            if args.dataset == 'math':
                pred = check_consistency(y, ans)
            else:
                pred = ans
            output_line["pred_ans"] = pred
            output_line["all_ans"] = all_answer
            output_json = json.dumps(output_line)
            wp.write(output_json + '\n')

            # Choose the most frequent answer from the list ...
                    
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')
                    
            # Checking answer ...
            correct = (np.array([pred]) == np.array([y])).sum().item()
            correct_list.append(correct)
            total += 1 #np.array([y]).size(0)
                    
            if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                break
            #raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))

    # Calculate averages for time and token statistics
    avg_time_spent = total_time_spent / total if total > 0 else 0
    avg_input_tokens = total_input_tokens_accum / total if total > 0 else 0
    avg_output_tokens = total_output_tokens_accum / total if total > 0 else 0
    avg_total_tokens = total_tokens_accum / total if total > 0 else 0

    # Print the averages
    print("Average time spent per question: {:.2f} seconds".format(avg_time_spent))
    print("Average input tokens per question: {}".format(avg_input_tokens))
    print("Average output tokens per question: {}".format(avg_output_tokens))
    print("Average total tokens per question: {}".format(avg_total_tokens))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["gsm8k", "math", "mmlu_chemistry", "mmlu_elec", "mmlu_ml", "mmlu_physics", "commonsensqa", "strategyqa"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot" , "v_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/Mutual_con", help="output directory"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for GPT-3"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
    elif args.dataset == "math":
        args.dataset_path = "./dataset/math/test.jsonl"
    elif args.dataset == "mmlu":
        args.dataset_path = "./dataset/mmlu_stem/test.jsonl"
        # mmlu subdataset
    elif args.dataset == "mmlu_chemistry":
        args.dataset_path = "./dataset/mmlu_stem/mmlu_chemistry.jsonl"
    elif args.dataset == "mmlu_elec":
        args.dataset_path = "./dataset/mmlu_stem/mmlu_elec.jsonl"
    elif args.dataset == "mmlu_ml":
        args.dataset_path = "./dataset/mmlu_stem/mmlu_machine_learning.jsonl"
    elif args.dataset == "mmlu_physics":
        args.dataset_path = "./dataset/mmlu_stem/mmlu_physics.jsonl"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    return args


if __name__ == "__main__":
    main()
