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

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

prompt = '''Please perform Aspect Sentiment Triplet Extraction task. Please tag the triplet (aspect, opinion, sentiment),
where aspect and opinion are derived from text, and sentiment is one of positive, negative and neutral. 
Please note that there may be more than one triplet in a text, please tag them completely. 
You must return them in this format without any other comments or texts: [("aspect","opinion","sentiment")], meaning all you have to do is to give me the label of the last text.'''

device = "cuda" 

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen2-7B-Instruct",
    cache_dir="./LLMs",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2-7B-Instruct", cache_dir = "./LLMs")

results = []
pattern = r'\[\((.*?)\)\]'

dataset_name = 'rest16'

print(f'****************{dataset_name}*****************')
path_r = './data/V2/{0}/test.txt'.format(dataset_name)
path_w = './data/V2/{0}/test_predict.txt'.format(dataset_name)
data = load_data(path_r)
for item in data:
    messages = [
        {"role": "system", "content": "You are an expert in information extraction and sentiment analysis"},
        {"role": "user", "content": prompt+'\n'+'Text: '+item['text']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=256
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if '\'' not in response:
        response.replace('\"','\'')
    match = re.match(pattern, response)
    if match:
        try:
            results.append(item['text']+'####'+str(eval('[('+match.group(1)+')]')))
        except Exception as e:
            print(f"!!!!!Exception: {e}")
    else:
        results.append(item['text']+'####'+str(eval("[]")))

with open(path_w,'w',encoding='utf-8') as f_write:
    f_write.write('\n'.join(results))

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))