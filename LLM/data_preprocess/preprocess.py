import json

import pandas as pd

test_data = json.load(open('../../datasets/test.jsonl'))
# prompt_template_df = pd.read_csv('../prompt_design/zero_shot.csv')
# prompt_template_df = pd.read_csv('../prompt_design/one_shot.csv')
prompt_template_df = pd.read_csv('../prompt_design/multi_shot.csv')

prompts = {index: [] for index in range(len(prompt_template_df))}
refs = []

for data in test_data:
    diff = data['diff']
    refs.append(data['commit_message'])
    for index, prompt_template in prompt_template_df['prompts'].items():
        updated_template = prompt_template.replace('[diff]', diff)
        prompts[index].append(updated_template)

prefix = 'multi'

df = pd.DataFrame({
    f"{prefix}_prompt_1": prompts[0],
    f"{prefix}_prompt_2": prompts[1],
    f"{prefix}_prompt_3": prompts[2],
    'references': refs,
})

df.to_csv('chat_api_multi_data.csv', index=False)