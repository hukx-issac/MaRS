import transformers
import torch
from modelscope import snapshot_download
import time
import re

def load_data(path):
    with open(path,'r',encoding='utf-8') as f_read:
        content = f_read.read()
    data = []
    for text in content.strip().split('\n'):
        items = text.split('####')
        review, label = items[0], items[1]

        label = eval(label)
        data.append({
            'text': review,
            'label': label
        })
    return data

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

prompt = '''Please perform Aspect Sentiment Triplet Extraction task. Please tag the triplet (aspect, opinion, sentiment),
where aspect and opinion are derived from text, and sentiment is one of positive, negative and neutral. 
Please note that there may be more than one triplet in a text, please tag them completely. 
You must return them in this format without any other comments or texts: [("aspect","opinion","sentiment")], meaning all you have to do is to give me the label of the last text.'''

model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct",cache_dir='./LLMs')

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

results = []
pattern = r'\[\((.*?)\)\]'


dataset_name = 'laptop14'

print(f'****************{dataset_name}*****************')

path_r = './data/V2/{0}/test.txt'.format(dataset_name)
path_w = './data/V2/{0}/test_predict.txt'.format(dataset_name)
data = load_data(path_r)

for item in data:
    messages = [
        {"role": "system", "content": "You are an expert in information extraction and sentiment analysis"},
        {"role": "user", "content": prompt+"\n"+"Text: "+item["text"]+'\n'+'Label:'}
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        pad_token_id = pipeline.tokenizer.eos_token_id,
    )
    response = outputs[0]["generated_text"][-1]['content']
    if '\'' not in response:
        response.replace('\"','\'')
    match = re.search(pattern,response)
    if match:
        try:
            results.append(item['text']+'####'+str(eval('[('+match.group(1)+')]')))
        except Exception as e:
            print(f"!!!!!Exception: {e}")
    else:
        results.append(item['text']+'####'+str(eval('[]')))
with open(path_w,'w',encoding='utf-8') as f_write:
    f_write.write('\n'.join(results))

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))