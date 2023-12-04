import re
import json
from collections import defaultdict
from pprint import pprint
import pandas as pd 

fn = '/cfs/cfs-k04hi56h/users/yanghuanye/clientstore/SecurityLLM/preprocessed/spider_dev_alpaca_prompt.json'
data = json.load(open(fn))

new_v2_log = 'spider_llama_new_preds_v2.log'
old_log = 'spider_llama_old_preds.log'


def read_log(log_file):
    pattern = r'^([0-9]+) (\w+) (\w+) score:(\w+) .* pred:(.*)'
    with open(log_file) as f:
        logs = [i for i in f.readlines() if re.match(pattern, i)]
    
    pred_logs = defaultdict(dict)
    for i in logs:
        target = re.findall(pattern, i)[0]
        pred_logs[int(target[0])][target[2]] = (target[3], target[1], target[4])
    return pred_logs


new_v2_preds = read_log(new_v2_log)
old_preds = read_log(old_log)

badcase_v1 = defaultdict(list)
badcase_v2 = defaultdict(list)
badcase_v3 = defaultdict(list)
results = defaultdict(list)
for score_type in ['exec', 'match']:
    for idx, item in enumerate(data):
        new_v2_pred = new_v2_preds[idx]
        new_v2_pred = new_v2_pred[score_type]
        new_v2_score = (new_v2_pred[0]=='1' or new_v2_pred[0]=='True')

        old_pred = old_preds[idx]
        old_pred = old_pred[score_type]
        old_pred_score = (old_pred[0]=='1' or old_pred[0]=='True')
        res = dict(
            idx=idx, 
            sql_type=new_v2_pred[1],
            new_pred=new_v2_pred[-1],
            old_pred=old_pred[-1],
            gt=item['query'],
        )
        results[score_type].append(res)
        if old_pred_score and not new_v2_score:
            badcase_v1[score_type].append(idx)
        elif not old_pred_score and new_v2_score:
            badcase_v2[score_type].append(idx)
        elif not old_pred_score and not new_v2_score:
            badcase_v3[score_type].append(idx)
    print(f'{score_type} v1:', len(badcase_v1[score_type]))
    print(f'{score_type} v2:', len(badcase_v2[score_type]))
    print(f'{score_type} v3:', len(badcase_v3[score_type]))

score_type = 'exec'
for idx in badcase_v2[score_type]:
    pprint(results[score_type][idx])
    breakpoint()

    

        