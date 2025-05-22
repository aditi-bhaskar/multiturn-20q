from .llm_judge import LLMJudge
from .token_amount import TokenAmount

from .bert import BertScore
from .bleurt_score import BLEURTScore
from .bleu_score import SentenceBLEU
from .wmd import WMDDistance
from .pass_rate import PassRate
from .jaccard_similarity import JaccardSimilarity

metric_info = {
    'token_amount': (TokenAmount, []),
    'llm_judge': (LLMJudge, ['task_name', 'model', 'temperature', 'max_new_tokens', 'rescale_func']),
    'bert_score': (BertScore, ['bert_score_model', 'model', 'temperature', 'max_new_tokens']),
    'bleu_score': (SentenceBLEU, ['model', 'temperature', 'max_new_tokens']),
    'bleurt_score': (BLEURTScore, ['model', 'temperature', 'max_new_tokens']),
    'wmd': (WMDDistance, ['wmd_model', 'model', 'temperature', 'max_new_tokens']),
    'pass_rate': (PassRate, ['model', 'temperature', 'max_new_tokens']),
    'jaccard_similarity': (JaccardSimilarity, ['model', 'temperature', 'max_new_tokens'])
}

registered_general_metrics = ['token_amount']

registered_task_metrics = {
    'document-editing': {
        'llm_metrics': ['llm_judge->interactivity'],
        'llm_judge_rescale_func': lambda x: (x - 2.5) * 2,
        'task_specific': 'bleu_score',
        'others': ['jaccard_similarity', 'wmd']  # , 'bleurt_score'
        },
    'code-generation': {
        'llm_metrics': ['llm_judge->interactivity'],
        'llm_judge_rescale_func': lambda x: (x - 2.5) * 2,
        'task_specific': 'pass_rate',
        'others': [],
    },
    'question-answering': {
        'llm_metrics': ['llm_judge->interactivity'],
        'llm_judge_rescale_func': lambda x: (x - 2.5) * 2,
        'task_specific': 'llm_judge->accuracy',
        'others': [],
    },
    'maybe-ambiguous-qa': {
        'llm_metrics': [],
        'llm_judge_rescale_func': lambda x: x,
        'task_specific': 'llm_judge->clr_or_answer_acc',
        'others': [],
    },
    #  aditi edit
    #  here, I have 2 different reward functions outlined. 
    # the first is generic, from the math task, and the second is more specific, and makes use of the information gain of each question

    #  original/math reward function -- 
    '20q': {  
        'llm_metrics': ['llm_judge->interactivity'],
        'llm_judge_rescale_func': lambda x: (x - 2.5) * 2,
        'task_specific': 'llm_judge->accuracy',
        'others': [],
    },


    #  TODO rerun the generate conv dpo on this reward function to understand the output!?

    # # specific/20q reward function -- 
    # '20q': {
    #     'llm_metrics': ['llm_judge->information_gain'],
    #     'llm_judge_rescale_func': lambda x: x * 2,  # scale to [-2, 2] assuming original in [0, 1]
    #     'task_specific': 'llm_judge->information_gain',
    #     'others': [],
    # },

}
