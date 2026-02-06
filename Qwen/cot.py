from modelscope import AutoModelForCausalLM, AutoTokenizer
import time
import re

def load_data(path):
    # ipdb.set_trace()
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

device = "cuda" 

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen2-7B-Instruct",
    cache_dir="/ai/tsp/ASTE/Qwen/LLMs",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2-7B-Instruct", cache_dir = "/ai/tsp/ASTE/Qwen/LLMs")

results = []
pattern = r'\[\((.*?)\)\]'

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

dataset_name = 'rest16'

count = 0
print(f'****************{dataset_name}*****************')
path_r = './data/V2/{0}/test.txt'.format(dataset_name)
path_w = './data/V2/{0}/test_predict.txt'.format(dataset_name)
data = load_data(path_r)

for item in data:
    count += 1
    messages = [
        {"role": "system", "content": "You are an expert in information extraction and sentiment analysis"},
        {"role": "user", "content": cot+'\n'+item['text']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    if '\'' not in response:
        response.replace('\"','\'')
    matches = re.findall(pattern, response)
    if matches:
        try:
            results.append(item['text']+'####'+str(eval('[('+matches[-1]+')]')))
            print(results[-1])
        except Exception as e:
            print(f"!!!!!Exception: {e}")
    else:
        results.append(item['text']+'####'+str(eval("[]")))

with open(path_w,'w',encoding='utf-8') as f_write:
    f_write.write('\n'.join(results))

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print(count)