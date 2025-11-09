# GSM8K
prompt_number = '''
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "The final answer is [ANSWER]"   (NOTE:To facilitate answer extraction, the [ANSWER] is a number.)
'''

# math
prompt_math = '''
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "The final answer is [ANSWER]"   (NOTE:The [ANSWER] should be as concise as possible, examples: "2","30w+12","1-x","11,\\!880","60^\\circ","\\frac{10}{3}","\\$150","50^{\\circ}","\\frac38","9\\text{ min}","50\\sqrt{3}","\\text{A,B,C,D}")
'''
# MMLU
prompt_mmlu = '''
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "The final answer is [ANSWER]"   (NOTE:[ANSWER] is the index of a list option, which is one of 0, 1, 2, or 3.)
'''
# commonsensqa
prompt_cs = ''' 
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "Among A through E, the final answer is [ANSWER]"  (NOTE:[ANSWER] is one of the options among ['A', 'B', 'C', 'D', 'E'])
'''

# StrategyQA
prompt_stQA = '''
(1) Let's think step by step, and answer the question.
(2) Conclude in this format: "The final answer is [ANSWER]"  (NOTE:When the answer of question  is "yes", [ANSWER] is "1", and when the answer is "no", [ANSWER] is "0".)
'''

prompt_template = {
    'prompt_number': prompt_number,
    'prompt_math': prompt_math,
    'prompt_mmlu': prompt_mmlu,
    'prompt_cs': prompt_cs,
    'prompt_stQA': prompt_stQA
}

