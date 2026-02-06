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

cot = '''You are an expert in information extraction and sentiment analysis. Please analyze the given text for
aspect sentiment triplet extraction using the following steps:
Concepts:
- Aspect: an attribute or entity. It is usually a noun or a noun phrase.
- Opinion: a descriptive term or phrase that expresses sentiment polarity towards a certain aspect. It is usually an adjective or a descriptive phrase.
- Sentiment: a polarity of opinion expressed towards a certain aspect, as positive, neutral and negative.
Instructions:
1. Read the text and identify all aspects mentioned.
2. For each identified aspect, analyse the opinion and the sentiment.
3. Return in the format [("Aspect", "Opinion", "Sentiment")]. Each triplet must contain an aspect, an opinion, and a sentiment.
4. There may be more than one triplet in a text, please extract them completely, returning them in the format: [("Aspect1", "Opinion1", "Sentiment1"),("Aspect2", "Opinion2", "Sentiment2")]
5. Aspect and opinion are derived from text.

Next, let's think step by step. Please analyze the following text:
'''

model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct",cache_dir='./LLMs')

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",

)
pattern = r'\[\((.*?)\)\]'


dataset_name = 'laptop14'

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

count = 0

results = []
print(f'****************{dataset_name}*****************')

path_r = './data/V2/{0}/test.txt'.format(dataset_name)
path_w = './data/V2/{0}/test_predict.txt'.format(dataset_name)
data = load_data(path_r)

for item in data:
    count+=1
    messages = [
        {"role": "system", "content": "You are an expert in information extraction and sentiment analysis."},
        {"role": "user", "content": cot+"\n"+item["text"]}
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=512,
        pad_token_id = pipeline.tokenizer.eos_token_id,
    )
    response = outputs[0]["generated_text"][-1]['content']
    print()
    print('***************************************************')
    print(response)
    matches = re.findall(pattern,response)
    if matches:
        try:
            results.append(item['text']+'####'+str(eval('[('+matches[-1]+')]')))
            print(results[-1])
        except Exception:
            pass
    else:
        results.append(item['text']+'####'+str(eval('[]')))
with open(path_w,'w',encoding='utf-8') as f_write:
    f_write.write('\n'.join(results))

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print(count)