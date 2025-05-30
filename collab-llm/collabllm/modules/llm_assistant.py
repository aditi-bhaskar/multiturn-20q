from typing import List
from collabllm.utils.api import get_llm_output
from collabllm.utils.template import chat_template
from collabllm.prompts import LLM_ASSISTANT_PROMPT_COT, LLM_ASSISTANT_PROMPT_ZS, \
    LLM_ASSISTANT_PROMPT_PROACT, LLM_ASSISTANT_PROMPT_PROACT_COT, LLM_ASSISTANT_PROMPT_PROACT_COT_GT, LLM_ASSISTANT_PROMPT_PROACT_COT_20Q

import datetime

class LLMAssistant(object):
    registered_prompts = {
        'none': None,
        'zero-shot': LLM_ASSISTANT_PROMPT_ZS,
        'cot': LLM_ASSISTANT_PROMPT_COT,
        'proact': LLM_ASSISTANT_PROMPT_PROACT,
        'proact_cot': LLM_ASSISTANT_PROMPT_PROACT_COT,
        'proact_gt_cot': LLM_ASSISTANT_PROMPT_PROACT_COT_GT,
        'proact_cot_20q': LLM_ASSISTANT_PROMPT_PROACT_COT_20Q # added by aditi
    }

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def __init__(self, method='zero-shot', **llm_kwargs):
        """
        Initialize the LLMAssistant model.
        """
        super().__init__()
        self.method = method
        self.prompt_handler = self.registered_prompts[method]
        self.max_new_tokens = llm_kwargs.get('max_new_tokens', 512)
        self.llm_kwargs = llm_kwargs

    def __call__(self, messages: List[dict], **kwargs):
        assert messages[-1]['role'] == 'user'

        if not self.method in ['proact_gt', 'proact_gt_cot']:
            kwargs = {}

        if self.method == 'none':
            prompt = messages
            if len(prompt) and prompt[0]['role'] == 'system':
                print('[LLMAssistant] System message detected.')
            # Convert messages list to string prompt
            prompt_str = chat_template(messages)
        else:
            prompt_str = self.prompt_handler(chat_history=chat_template(messages),
                                            max_new_tokens=self.max_new_tokens,
                                            **kwargs)

        cnt = 0
        while True:
            cnt += 1
            # Rebuild prompt if needed (can optimize later)
            if self.method == 'none':
                prompt_str = chat_template(messages)
            else:
                prompt_str = self.prompt_handler(chat_history=chat_template(messages),
                                                max_new_tokens=self.max_new_tokens,
                                                **kwargs)

            response = get_llm_output(prompt_str, **self.llm_kwargs)

            print(prompt_str)
            breakpoint

            if isinstance(response, dict):
                try:
                    keys = response.keys()
                    current_problem = response.pop('current_problem')
                    thought = response.pop('thought')
                    response = response['response']

                    with open(f'logs/llm_assistant_{self.timestamp}.txt', 'a+') as f:
                        f.write(f'\n\n[LLMAssistant] `current_problem`={current_problem} | `thought`={thought} | `response`={response}\n\n')
                    break
                except Exception as e:
                    print(f'[LLMAssistant] {e}')
                    import pdb; pdb.set_trace()
            else:
                break
            if cnt > 5:
                import pdb; pdb.set_trace()
        return response.strip()
