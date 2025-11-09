# GSM8K
first_prompt_number = '''
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "The final answer is [ANSWER]"   (NOTE:To facilitate answer extraction, the [ANSWER] is a number.)
'''
other_prompt_number = '''
(1) You have received a question and two different answers to it. Based on this, update your response.
(2) Conclude in this format: "The final answer is [ANSWER]" (NOTE:To facilitate answer extraction, the [ANSWER] is a number.)
'''

# MATH
first_prompt_math = '''
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "The final answer is [ANSWER]"   (NOTE:The [ANSWER] should be as concise as possible, examples: "2","30w+12","1-x","11,\\!880","60^\\circ","\\frac{10}{3}","\\$150","50^{\\circ}","\\frac38","9\\text{ min}","50\\sqrt{3}","\\text{A,B,C,D}")
'''
other_prompt_math = '''
(1) You have received a question and two different answers to it. Based on this, update your response.
(2) Conclude in this format: "The final answer is [ANSWER]" (NOTE:The [ANSWER] should be as concise as possible, examples: "2","30w+12","1-x","11,\\!880","60^\\circ","\\frac{10}{3}","\\$150","50^{\\circ}","\\frac38","9\\text{ min}","50\\sqrt{3}","\\text{A,B,C,D}")
'''
# MMLU
first_prompt_mmlu = '''
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "The final answer is [ANSWER]"   (NOTE:[ANSWER] is the index of a list option, which is one of 0, 1, 2, or 3.)
'''
other_prompt_mmlu = '''
(1) You have received a question and two different answers to it. Based on this, update your response.
(2) Conclude in this format: "The final answer is [ANSWER]"  (NOTE:[ANSWER] is the index of a list option, which is one of 0, 1, 2, or 3.)
'''

# commonsensqa
first_prompt_cs = '''
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "Among A through E, the final answer is [ANSWER]" (NOTE:[ANSWER] is one of the options among ['A', 'B', 'C', 'D', 'E'])
'''
other_prompt_cs = '''
(1) You have received a question and two different answers to it. Based on this, update your response.
(2) Conclude in this format: "Among A through E, the final answer is [ANSWER]" (NOTE:[ANSWER] is one of the options among ['A', 'B', 'C', 'D', 'E'])
'''
# StrategyQA
first_prompt_stQA = '''
(1)  Let's think step by step, and answer the question.
(2) Conclude in this format: "Among A through E, the final answer is [ANSWER]" (NOTE:When the answer of question  is "yes", [ANSWER] is "1", and when the answer is "no", [ANSWER] is "0")
'''
other_prompt_stQA = '''
(1) You have received a question and two different answers to it. Based on this, update your response.
(2) Conclude in this format: "Among A through E, the final answer is [ANSWER]" (NOTE:When the answer of question  is "yes", [ANSWER] is "1", and when the answer is "no", [ANSWER] is "0")
'''

prompt_template = {
    'first_prompt_number': first_prompt_number,
    'other_prompt_number': other_prompt_number,
    'first_prompt_math': first_prompt_math,
    'other_prompt_math': other_prompt_math,
    'first_prompt_mmlu': first_prompt_mmlu,
    'other_prompt_mmlu': other_prompt_mmlu,
    'first_prompt_cs': first_prompt_cs,
    'other_prompt_cs': other_prompt_cs,
    'first_prompt_stQA': first_prompt_stQA,
    'other_prompt_stQA': other_prompt_stQA
}
