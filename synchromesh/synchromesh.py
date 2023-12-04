#!/usr/bin/env python3

import os
import regex
import time
import json
import re
import sys

from typing import List, Dict
from completion_engine import CompletionEngine, LarkCompletionEngine
from language_model import LanguageModel, RandomLanguageModel, OpenAIModel, HuggingFaceModel
import trie

# Implements the Constrained Semantic Decoding algorithm.
def predict_constrained(completion_engine: CompletionEngine, lm: LanguageModel,
                        top_k: int = 1, verbose: bool = False,
                        batch_size: int = 50, stop_tokens: List[str]=None,
                        max_violations: int = 20,
                        fast_forward: bool = False) -> str:
    completion_points: Dict[str, regex.Pattern] = {}

    completion_points[''] = completion_engine.complete('')

    token_trie = trie.Trie.from_vocabulary(lm.vocabulary())

    prediction = ''
    n_violations = 0
    
    while not completion_engine.is_complete(prediction):
        # Ask for unconstrained prediction.
        if verbose:
            print('Prefix:', prediction)

        if False and fast_forward:
            # While there's only one valid token at this point, simply add
            # that token instead of querying the model. This can be slow but can also
            # save many calls to the model in use cases where the completion engine can
            # output very long constraints (e.g. only let the model choose between generating
            # two long sequences, so after it starts to output one the rest is determined).
            while True:
                valid_tokens = token_trie.antimonotonic_filter(
                    lambda t: is_prefix_valid(completion_engine,
                                              completion_points,
                                              prediction + t)
                )

                if len(valid_tokens) == 1:
                    prediction += lm.get_token(valid_tokens[0])
                    if completion_engine.is_complete(prediction):
                        return prediction
                else:
                    # print([lm.get_token(i) for i in valid_tokens])
                    break

            if verbose:
                print('After fast forwarding:', prediction)

        # continuation = text

        if sys.argv[1] == 'llama':
            continuation = lm.predict_unconstrained(prediction, batch_size, stop=stop_tokens)

        continuation = continuation.replace('</s>', '')
        for agg in ['SUM', 'AVG', 'MIN', 'MAX', 'COUNT', 'DISTINCT']:
            pat = r'' + agg + r'\s+\('
            continuation = re.sub(pat, f'{agg}(', continuation)
            pat = r'' + agg.lower() + r'\s+\('
            continuation = re.sub(pat, f'{agg}(', continuation)
        continuation = continuation.strip()

        # if not continuation.endswith(';'):
        #     continuation += ';'

        found_violation = False

        if verbose:
            print('Continuation:', continuation)

        if not continuation:
            # HACK: LM really thinks is done. This will not make progress.
            # Trusting it for now.
            if verbose:
                print('Empty continuation. Stopping early because model refuses to keep going.')
            break

        # 逐个check llm生成的每个token是否合理，这里是否可以改造成回溯？
        # 如果某个token的路走不通了
        for token in lm.tokenize(continuation):
            if is_prefix_valid(completion_engine, completion_points,
                               prediction + lm.get_token(token)):
                prediction += lm.get_token(token)
            else:
                if completion_engine.is_complete(prediction):
                    break
                found_violation = True
                if verbose:
                    print(f"***Found violation at token: {repr(lm.get_token(token))}")
                    print(f"Valid prefix: {prediction}")
                    # for k, v in completion_engine.parser.parser.lexer.lexers.items():
                    #     print('Key:', k)
                    #     print('Value:', v)
                    #     print()
                    # breakpoint()

                    is_prefix_valid(completion_engine, completion_points, prediction + lm.get_token(token))
                    completion_engine.complete(text)
                break

        if found_violation:
            n_violations += 1

            if n_violations > max_violations:
                break

            # Do constrained prediction for next token.
            if verbose:
                print(f"Constrained prediction for: {prediction}")
                print('Determining valid tokens...')
            # breakpoint()
            valid_tokens = token_trie.antimonotonic_filter(
                lambda t: is_prefix_valid(completion_engine,
                                          completion_points,
                                          prediction + t)
            )

            if verbose:
                print('Done:', len(valid_tokens), 'tokens.')

            assert len(valid_tokens) > 0, f"No valid tokens after {repr(prediction)}"

            # 无回溯，重新接着prediction之后，在合理的token之中去选择生成下一个prediction token
            predictions, probabilities = lm.predict_token(prediction,
                                                        valid_tokens,
                                                        top_k)

            predicted_token = predictions[0]
            prediction += lm.get_token(predicted_token)
        else:
            # 表示目前的continuation里的token都合法，此时无需再继续生成了，直接作为最后的结果。
            break

        breakpoint()

    return prediction

def get_longest_completion_point(completion_points, s):
    longest_completion_point = 0
    for i in range(len(s)+1):
        if s[:i] in completion_points:
            longest_completion_point = i
    return longest_completion_point

def is_prefix_valid(completion_engine: CompletionEngine,
                    completion_points: Dict[str, regex.Pattern],
                    s: str) -> bool:
    # print('completion_points:', completion_points)
    # print('s:', s)
    # breakpoint()
    # completion_points用来记录已check合理的前缀，首先跳过这些合理的前缀，接着对后面的字符串remainder继续check

    # 1- Find longest completion point that is a prefix of s.
    longest_completion_point = get_longest_completion_point(completion_points, s)

    # 2- Take the 'remainder'.
    completion_point_regex = completion_points[s[:longest_completion_point]]
    remainder = s[longest_completion_point:]
    # remainder = s

    # 3- Feed it character by character to the regex given by the completion point, and handle 3 cases:
    # for i in range(longest_completion_point, len(s)):
    for i in range(len(remainder)):
        # If we have a violation of the regex.
        if not completion_point_regex.fullmatch(remainder[:i+1], partial=True):
            # Check if we have a full match up to the previous character.
            if completion_point_regex.fullmatch(remainder[:i]):
                # We found another completion point, reduce the problem and call recursively.
                new_completion_point = s[:longest_completion_point] + remainder[:i]                # print(new_completion_point)
                # print('new_completion_point:', new_completion_point)
                # print('s:', s)
                new_completion_point_regex = completion_engine.complete(new_completion_point)
                completion_points[new_completion_point] = new_completion_point_regex
                # print(new_completion_point)
                # print(s)
                # breakpoint()
                return is_prefix_valid(completion_engine, completion_points, s)
            else:
                # print(remainder, i, remainder[:i])
                # breakpoint()
                # completion_point_regex.fullmatch(remainder[:i])
                return False

    #    Case c- Got to the end with no violations, return True
    return True



def test_fast_forward():
    fixed_response = r"""
        ?response: "the answer is abcdefghijklmnopqrstuvwxyz"
    """

    prompt = """You are a helpful assistant.

Human: What day is today?
Assistant: Thursday

Human: What is the answer?
Assistant:"""

    num_samples = 1
    api_key = os.environ.get('OPENAI_API_KEY')
    for i in range(num_samples):
        comp_engine = LarkCompletionEngine(fixed_response, 'response', False)

    ada = OpenAIModel(model="text-ada-001", prompt_template=prompt,
                      api_key=api_key, temperature=1.)
    print(predict_constrained(comp_engine, ada, 1, True,
                              stop_tokens=["\n"], fast_forward=True))


def add_sql_syntax(_GRAMMAR_TEXT):
    mysql_datetime_functions = [
        'NOW',
        'CURDATE',
        'CURTIME',
        'CURRENT_DATE',
        'CURRENT_TIME',
        'CURRENT_TIMESTAMP',
        'DATE',
        'TIME',
        'TIMESTAMP',
        'EXTRACT',
        'DATE_ADD',
        'DATE_SUB',
        'DATEDIFF',
        'TIMEDIFF',
        'DATE_FORMAT',
        'TIME_FORMAT',
        'FROM_UNIXTIME',
        'UNIX_TIMESTAMP',
        'STR_TO_DATE',
        'SEC_TO_TIME',
        'CONVERT_TZ',
        'DAY',
        'MONTH',
        'YEAR',
        'HOUR',
        'MINUTE',
        'SECOND',
        'LAST_DAY',
        'ADDDATE',
        'SUBDATE',
        'MAKEDATE',
        'MAKETIME',
        'PERIOD_ADD',
        'PERIOD_DIFF',
        'TO_DAYS',
        'FROM_DAYS',
        'WEEK',
        'DAYNAME',
        'MONTHNAME',
        'DAYOFWEEK',
        'DAYOFMONTH',
        'DAYOFYEAR',
        'WEEKDAY',
        'WEEKOFYEAR',
        'EXTRACTVALUE',
        'PERIOD_ADD',
        'PERIOD_DIFF',
        'TO_SECONDS',
        'FROM_UNIXTIME',
        'UNIX_TIMESTAMP',
        'SEC_TO_TIME',
        'STR_TO_DATE',
        'CONVERT_TZ',
        'SYSDATE',
        'UTC_DATE',
        'UTC_TIME',
        'UTC_TIMESTAMP'
    ]

    mysql_datetime_functions = ' | '.join([f'"{i}"i "(" ")"' for i in mysql_datetime_functions])
    
    mysql_datetime_functions = """datetime_functions: "NOW" "(" ")"
                | "CURDATE" "(" ")"
                | "CURTIME" "(" ")"
                | "CURRENT_DATE" "(" ")"
                | "CURRENT_TIME" "(" ")"
                | "CURRENT_TIMESTAMP" "(" ")"
                | "DATE_ADD" "(" expression "," interval_expression ")"
                | "DATE_SUB" "(" expression "," interval_expression ")"
                | "DATEDIFF" "(" expression "," expression ")"
                | "DATE_FORMAT" "(" expression "," string ")"
                | "FROM_UNIXTIME" "(" expression ")"
                | "UNIX_TIMESTAMP" "(" expression ")"
                | "STR_TO_DATE" "(" expression "," string ")"
                | "SEC_TO_TIME" "(" expression ")"
                | "CONVERT_TZ" "(" expression "," string "," string ")"
    """

if __name__ == '__main__':
    gpt4_api_key = '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'
    azure_api_key = "70c6b6ae32ef4807b38c2b2b9bc1ef32"
    os.environ["OPENAI_API_KEY"] = azure_api_key
    # test_fast_forward()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    invalid_text = []
    start_idx = 0
    invalid_index = []
    if sys.argv[2] == 'spider':
        fn = '/cfs/cfs-k04hi56h/users/yanghuanye/clientstore/SecurityLLM/preprocessed/spider_dev_alpaca_prompt.json'
        data = json.load(open(fn))
        # fn = '/workspace/user_code/stanford_alpaca/spider/dev.json'
        table_file = '/cfs/cfs-k04hi56h/users/yanghuanye/clientstore/SecurityLLM/data/spider_tables.json'
        start_idx = 750
        # invalid_index = []
        # start_idx = 30
        # invalid_index = [30, 33, 34]
        # start_idx = 983
        # invalid_index = [720, 744, 804, 983]
    elif sys.argv[2] == 'security':
        fn = '/workspace/user_code/chatbot_text2sql/preprocessed/security_1018_valid_val_reviewed_merged_v2_alpaca_with_type_with_comment_prompt.json'
        data = json.load(open(fn))
        table_file = '/cfs/cfs-k04hi56h/users/yanghuanye/clientstore/SecurityLLM/data/security_tables.json'
        # start_idx = 201
        # invalid_index = [170]
    elif sys.argv[2] == 'test':
        init_text = 'select intersect select;'
        init_text = 'SELECT country FROM singer WHERE age  <  30 INTERSECT SELECT country FROM singer WHERE age  >  40'
        data = [{'text': init_text, 'query': init_text}]
        # start_idx = 0
        # invalid_index = []
    # start_idx = 0
    GRAMMAR_PATH = './sql.lark'
    GRAMMAR_PATH = sys.argv[3]
        
    # GRAMMAR_PATH = '/workspace/user_code/sql_to_ibis/sql_to_ibis/grammar/sql.lark'
    # GRAMMAR_PATH = '/workspace/user_code/synchromesh/synchromesh/spider_grammer.lark'
    with open(file=GRAMMAR_PATH) as sql_grammar_file:
        _GRAMMAR_TEXT= sql_grammar_file.read()


    text = "select column1 from my_table"
    prompt = """write a sql query to get column1 from my table. SQL:"""
    ada = OpenAIModel(model="gpt-3.5-turbo", prompt_template=prompt,
                      api_key=api_key, temperature=1.)


    # from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    # model_path = '/cfs/cfs-k04hi56h/users/yanghuanye/output/LLM/ChatGLM-6B/THUDM-chatglm-6b/spider_train'
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    if sys.argv[1] == 'llama':
        from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
        model_path = '/cfs/cfs-k04hi56h/users/yanghuanye/outputs/LLM/alpaca-ft-nl2sql/alpaca_recovered_weights/spider_train_alpaca_prompt/'
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path)
        model = model.half()
        model.eval()
        model = model.to('cuda')
        tokenizer.pad_token = tokenizer.eos_token

    
    engine_mapping = {}
    for idx, i in enumerate(data[:]):
        text = i['query'].strip()
        # if text.endswith(';'):
        #     text = text.replace(';', '')
        # if not text.endswith(';'):
        #     text = text + ';'
        text = text.replace('"', '\'')
        print(idx, text.replace('\n', ' '))

        # if 'EXCEPT' in text.upper() or 'INTERSECT' in text.upper() or 'UNION' in text.upper():
        #     invalid_index.append(idx)
        #     # print(comp_engine.parser.parse(text))
        #     continue
        if idx < start_idx:
            continue
        
        # if idx in invalid_index:
        #     # print(comp_engine.parser.parse(text))
        #     continue

        NEW_GRAMMAR_TEXT = _GRAMMAR_TEXT

        if 'table_list' in i:
            table_list = i['table_list']
            tables = [i[0] for i in table_list]
            try:
                column_names = {j['column_name'] for i in table_list for j in i[1]}
            except:
                column_names = {j for i in table_list for j in i[1]}
            table_name = ' | '.join([f'"{i}"i' for i in tables])
            column_name = ' | '.join([f'"{i}"i' for i in column_names])
            NEW_GRAMMAR_TEXT += f'\nTABLE_NAME: {table_name}'
            NEW_GRAMMAR_TEXT += f'\nCOLUMN_NAME: {column_name}'
            
        if NEW_GRAMMAR_TEXT not in engine_mapping:
            try:
                comp_engine = LarkCompletionEngine(NEW_GRAMMAR_TEXT, 'start')
            except:
                breakpoint()
                comp_engine = LarkCompletionEngine(NEW_GRAMMAR_TEXT, 'start')
            engine_mapping[NEW_GRAMMAR_TEXT] = comp_engine
        
        comp_engine = engine_mapping[NEW_GRAMMAR_TEXT]
        comp_engine.parse(text)
        continue
        # if idx < 1034:
        #     continue
        # breakpoint()
        res = {}
        res['gt'] = text
        try:
            # print(comp_engine.parser.parse(text))
            # breakpoint()
            if sys.argv[1] == 'llama':
                batch_size = 256
                prompt = i['text']
                ada = HuggingFaceModel(model=model, prompt_template=prompt,
                        tokenizer=tokenizer, temperature=1.)
            #     text = ada.predict_unconstrained(prompt, batch_size, stop=[';'])
            #     text = text.replace('</s>', '')
            if idx not in invalid_index or text in invalid_text:
                prediction = predict_constrained(comp_engine, ada, 1, True, batch_size=batch_size, stop_tokens=[], fast_forward=True)
                print(f'\n{prediction}\n')
                res['new'] = prediction
            else:
                invalid_text.append(text)
        except Exception as e:
            # print(text)
            # raise e
            print(e)
            # print(invalid_index)
            breakpoint()
            pass

        res['old'] = text
        if len(sys.argv) > 4:
            output_file = sys.argv[4]
            with open(output_file, 'a') as f:
                f.write(json.dumps(res)+'\n')
    
    print(len(invalid_index), invalid_index)
        
