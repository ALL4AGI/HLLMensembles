import argparse
from utils import *
import time
from transformers import GPT2Tokenizer
from prompt_serial import prompt_template


def get_prompts(args):
    if args.dataset == 'gsm8k':
        first_prompt_a = prompt_template['first_prompt_number_a']
        other_prompt = prompt_template['other_prompt_number']
    elif args.dataset == 'math':
        first_prompt_a = prompt_template['first_prompt_math_a']
        other_prompt = prompt_template['other_prompt_math']
    elif args.dataset in ("mmlu_chemistry", "mmlu_elec", "mmlu_ml", "mmlu_physics"):
        first_prompt_a = prompt_template['first_prompt_mmlu_a']
        other_prompt = prompt_template['other_prompt_mmlu']
    elif args.dataset == 'commonsensqa':
        first_prompt_a = prompt_template['first_prompt_csqa_a']
        other_prompt = prompt_template['other_prompt_csqa']
    elif args.dataset == 'strategyqa':
        first_prompt_a = prompt_template['first_prompt_strategy_a']
        other_prompt = prompt_template['other_prompt_strategy']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return first_prompt_a, other_prompt


def expert_a(question, history, iteration):
    args = parse_arguments()
    first_prompt_a, other_prompt = get_prompts(args)
    if iteration == 1:
        prompt = f"""question:{question}\n\nYou are Expert A. {first_prompt_a}"""
    else:
        prompt = f"""You are Expert A. {other_prompt}"""
    if history:
        if len(history) > 2:
            history_to_use = history[-2:]
        else:
            history_to_use = history
        history_str = "\n".join(history_to_use)
        prompt = f"question:{question}\n\nPrevious strategy:\n{history_str}\n\n" + prompt

    print(f"***Expert A's prompt***\n{prompt}")
    response = call_model(prompt)
    return response, prompt


def expert_b(question, history, iteration):
    args = parse_arguments()
    first_prompt_a, other_prompt = get_prompts(args)
    if iteration == 1:
        prompt = f"""\nYou are Expert B. {other_prompt}"""
    else:
        prompt = f"""\nYou are Expert B. {other_prompt}"""
    if history:
        if len(history) > 2:
            history_to_use = history[-2:]
        else:
            history_to_use = history
        history_str = "\n".join(history_to_use)
        prompt = f"question:{question}\n\nPrevious strategy:\n{history_str}\n\n" + prompt
    print(f"***Expert B's prompt***\n{prompt}")
    response = call_model2(prompt)
    return response, prompt


def expert_c(question, history, iteration):
    args = parse_arguments()
    first_prompt_a, other_prompt = get_prompts(args)
    if iteration == 1:
        prompt = f"""You are Expert C. {other_prompt}"""
    else:
        prompt = f"""You are Expert C. {other_prompt}"""
    if history:
        if len(history) > 3:
            history_to_use = history[-3:]
        else:
            history_to_use = history
        history_str = "\n".join(history_to_use)
        prompt = f"question:{question}\n\nPrevious strategy:\n{history_str}\n\n" + prompt
    print(f"***Expert C's prompt***\n{prompt}")
    response = call_model3(prompt)
    return response, prompt


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def interactive(question, selected_experts=None, max_iterations=7):
    """
    Experts interact iteratively until a consistent answer is obtained.
    """
    if selected_experts is None:
        selected_experts = ['A', 'B']
    args = parse_arguments()
    history = []
    iteration = 0
    pred = None
    # Start timing
    start_time = time.time()
    # Initialize an empty string to accumulate all prompt strings
    total_prompts = ""
    total_solution = ""
    while iteration < max_iterations:
        iteration += 1
        print(f"\n******************* Iteration {iteration} ********************")

        # Expert A provides an initial answer or responds to feedback from Expert B
        expert_a_solution, prompt_a = expert_a(question, history, iteration)
        print(f"********* Expert A's Solution *******\n{expert_a_solution}")
        # Append expert's strategies to history
        history.append(f"Expert A's strategy:\n{expert_a_solution}")
        # Append the current prompt_a and solution to concatenated_prompts
        total_prompts += prompt_a + "\n"
        total_solution += expert_a_solution + "\n"

        # Confirm if it is the final answer.
        if "final answer is" in expert_a_solution:
            pred = extract_answer(args, expert_a_solution)
            break

        # Expert B checks and provides feedback
        expert_b_solution, prompt_b = expert_b(question, history, iteration)
        print(f"******* Expert B's Solution *********\n{expert_b_solution}")
        # Append both expert's strategies to history
        history.append(f"Expert B's strategy:\n{expert_b_solution}")
        # Append the current prompt_b and solution to concatenated_prompts
        total_prompts += prompt_b + "\n"
        total_solution += expert_b_solution + "\n"

        # Confirm if it is the final answer.
        if "final answer is" in expert_b_solution:
            pred = extract_answer(args, expert_b_solution)
            break

        if 'C' in selected_experts:
            # Expert C checks and provides feedback$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            expert_c_solution, prompt_c = expert_c(question, history, iteration)
            print(f"******* Expert C's Solution *********\n{expert_c_solution}")
            # Append both expert's strategies to history
            history.append(f"Expert C's strategy:\n{expert_c_solution}")
            # Append the current prompt_c and solution to concatenated_prompts
            total_prompts += prompt_c + "\n"
            total_solution += expert_c_solution + "\n"

            # 检查是否确认最终答案
            if "final answer is" in expert_c_solution:
                pred = extract_answer(args, expert_c_solution)
                break



    # If maximum iterations are reached and no consistent answer, use the most frequent answer
    if pred is None:
        print(f"\nMaximum iterations {max_iterations} reached, ending loop.")

    # Extract all answers from history
    all_answer = [extract_answer(args, entry) for entry in history]
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
            print("{}st data".format(i + 1))
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
            total += 1  #np.array([y]).size(0)

            if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
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
        choices=["gsm8k", "math", "mmlu_chemistry", "mmlu_elec", "mmlu_ml", "mmlu_physics"
                 , "commonsensqa", "strategyqa"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0,
        help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    parser.add_argument(
        "--output_dir", type=str, default="experiment\Series", help="output directory"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0,
        help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=1, help="temperature for GPT-3"
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
        #mmlu subdataset
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


