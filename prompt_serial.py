# GSM8K
first_prompt_number_a = '''
Let's think step by step, and answer the question.
'''

other_prompt_number = '''
(1)You have received a question and its solutions. Based on this, update your response.
(2)Draw a conclusion based on one of the following two cases:
(a) If your response result is inconsistent with the previous responses, conclude in this format: The reasoning behind the question is controversial and requires further discussion.
(b) If your response result is consistent with one of the previous responses, draw the final conclusion in this format: "The final answer is [ANSWER]"   (NOTE:To facilitate answer extraction, the [ANSWER] is a number.)
'''

# MATH
first_prompt_math_a = '''
Let's think step by step, and answer the question.
'''
other_prompt_math = '''
(1)You have received a question and its solutions. Based on this, update your response.
(2)Draw a conclusion based on one of the following two cases:
(a) If your response result is inconsistent with the previous responses, conclude in this format: The reasoning behind the question is controversial and requires further discussion.
(b) If your response result is consistent with one of the previous responses, draw the final conclusion in this format: "The final answer is [ANSWER]"   (NOTE:The [ANSWER] should be as concise as possible, examples: "2","30w+12","1-x","11,\\!880","60^\\circ","\\frac{10}{3}","\\$150","50^{\\circ}","\\frac38","9\\text{ min}","50\\sqrt{3}","\\text{A,B,C,D}")
'''
# MMLU
first_prompt_mmlu_a = ''' 
Let's think step by step, and answer the question.
'''
other_prompt_mmlu = '''
(1)You have received a question and its solutions. Based on this, update your response.
(2)Draw a conclusion based on one of the following two cases:
(a) If your response result is inconsistent with the previous responses, conclude in this format: The reasoning behind the question is controversial and requires further discussion.
(b) If your response result is consistent with one of the previous responses, conclude in this format: "The final answer is [ANSWER]"    (NOTE:[ANSWER] is the index of a list option, which is one of 0, 1, 2, or 3.)
'''

# CommonsensQA
first_prompt_csqa_a = ''' 
Let's think step by step, and answer the question.
'''

other_prompt_csqa = '''
(1)You have received a question and its solutions. Based on this, update your response.
(2)Draw a conclusion based on one of the following two cases:
(a) If your response result is inconsistent with the previous responses, conclude in this format: The reasoning behind the question is controversial and requires further discussion.
(b) If your response result is consistent with one of the previous responses, conclude in this format: "The final answer is [ANSWER]"    (NOTE:[ANSWER] is one of ['A','B','C','D','E'])
'''

# StrategyQA
first_prompt_strategy_a = ''' 
Let's think step by step, and answer the question.
'''
other_prompt_strategy = '''
(1)You have received a question and its solutions. Based on this, update your response.
(2)Draw a conclusion based on one of the following two cases:
(a) If your response result is inconsistent with the previous responses, conclude in this format: The question requires further discussion.
(b) If your response result is consistent with one of the previous responses, conclude in this format: "The final answer is [ANSWER]"    (NOTE:When the answer of question  is "yes", [ANSWER] is "1", and when the answer is "no", [ANSWER] is "0".)
'''

prompt_template = {
    'first_prompt_number_a': first_prompt_number_a,
    'other_prompt_number': other_prompt_number,
    'first_prompt_math_a': first_prompt_math_a,
    'other_prompt_math': other_prompt_math,
    'first_prompt_mmlu_a': first_prompt_mmlu_a,
    'other_prompt_mmlu': other_prompt_mmlu,
    'first_prompt_csqa_a': first_prompt_csqa_a,
    'other_prompt_csqa': other_prompt_csqa,
    'first_prompt_strategy_a': first_prompt_strategy_a,
    'other_prompt_strategy': other_prompt_strategy
}
