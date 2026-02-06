import transformers
import torch
from modelscope import snapshot_download
import time
import random
import re
import itertools
from string import Template

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

def bernoulli(threshold):
    return random.random() <= threshold

def initial():
    aos = ['a','o','s']
    permutations = list(itertools.permutations(aos))
    paths = []
    for per in permutations:
        paths.append({'text':'','pre_path':[per[0],per[1],per[2]], 'cur_node':per[0], 'cum_rethink': -1, 'res':[{'aspect': '', 'opinion':'','sentiment':''}]})        
    return paths

def forward(messages):
    outputs = pipeline(
            messages,
            max_new_tokens=512,
            pad_token_id = pipeline.tokenizer.eos_token_id,
        )
    response = outputs[0]["generated_text"][-1]['content']
    return response

def parse_result1(response):
    pattern = r'<(.*?)>'
    matches = re.findall(pattern, response)
    if matches:
        try:
            return str(eval(matches[-1]))
        except Exception:
            return '[]'
    else:
        return '[]'

def parse_result2(response):
    pattern = r'<(.*?)>'
    matches = re.findall(pattern, response)
    if matches:
        try:
            return matches[-1]
        except Exception:
            return '[]'
    else:
        return '[]'

def parse_result3(meessages):
    pattern = r'\[\((.*?)\)\]'
    matches = re.findall(pattern, meessages[-1]['content'])
    if matches:
        try:
            return str(eval('[('+matches[-1]+')]'))
        except Exception:
            return '[]'
    else:
        return '[]'

def unify_paths(selfconsist_line, path_results):
    temp_dict = {}
    for res in path_results:
        try:
            res_list = eval(res)
        except Exception:
            print(e)
            res_list = []
        for triplet in res_list:
            tri_str = str(triplet).lower()
            if tri_str not in temp_dict:
                temp_dict[tri_str] = 1
            else:
                temp_dict[tri_str] += 1

    unify_result = []
    for tri_str in temp_dict.keys():
        if temp_dict[tri_str]>=selfconsist_line:
            try:
                unify_result.append(eval(tri_str))
            except Exception as e:
                print(e)
    
    print(path_results)
    print()
    if len(unify_result)==0 or str(unify_result)=='[]':
        try:
            res = str(eval(path_results[0]))
        except Exception:
            res = '[]'
        return res
    else:
        return str(unify_result)

aos_vrs = {
    'laptop14': {'aspect':0.7186, 'opinion':0.5994, 'sentiment':0.6066},
    'rest14': {'aspect':0.6854, 'opinion':0.5504, 'sentiment':0.5738},
    'rest15': {'aspect':0.6650, 'opinion':0.5079, 'sentiment':0.5252},
    'rest16': {'aspect':0.6491, 'opinion':0.5332, 'sentiment':0.5439},
}

max_rethink = 2
min_rethink = 1
selfconsist_line = 2

trc_frags = {
    'dawn':Template('''You are an expert in information extraction and sentiment analysis. Please analyze the given text for aspect sentiment triplet extraction.
                    Aspect is an attribute or entity. It is usually a noun or a noun phrase derived from text.
                    Opinion is a descriptive term or phrase that expresses sentiment polarity towards a certain aspect. It is usually an adjective or a descriptive phrase derived from text.
                    Sentiment is a polarity of opinion expressed towards a certain aspect, only as positive, neutral or negative.
                    Let's think step by step.'''),

    'aspect':{0:Template("The text is: $text. Please first identify the aspect in the given text, which may contain more than one. Note that aspect is derived from text. Please output in this format: <'['aspect1', 'aspect2']'>."),
              1:Template("You have previously identified the $element_type of the text: $aspects, go ahead and identify the aspect corresponding to the $element_type. Note that aspect is derived from text. Please output in this format: <'['aspect1', 'aspect2']'>."),
              2:Template("In earlier you have identified the opinion and sentiment pair(s) in text: $aspects, so next please identify the aspect(s) corresponding to the pair(s). Note that aspect is derived from text. Please output in this format: <'['aspect1', 'aspect2']'>.")},

    'opinion':{0:Template("The text is: $text. Please first identify the opinion in the given text, which may contain more than one. Note that opinion is derived from text. Please output in this format: <'['opinion1', 'opinion2']'>."),
              1:Template("You have previously identified the $element_type of the text: $opinions, go ahead and identify the opinion corresponding to the $element_type. Note that opinion is derived from text. Please output in this format: <'['opinion1', 'opinion2']'>."),
              2:Template("In earlier you have identified the aspect and sentiment pair(s) in text: $opinions, so next please identify the opinion(s) corresponding to the pair(s). Note that opinion is derived from text. Please output in this format: <'['opinion1', 'opinion2']'>.")},

    'sentiment':{0:Template("The text is: $text. Please first identify the sentiment that may be present in the given text, which may contain more than one. Note that sentiment is one of  positive, neutral and negative. Please output in this format: <'['sentiment1', 'sentiment2']'>."),
              1:Template("You have previously identified the $element_type of the text: $sentiments, go ahead and identify the sentiment correspondings to the $element_type. Note that sentiment is one of  positive, neutral and negative. Please output in this format: <'['sentiment1', 'sentiment2']'>."),
              2:Template("In earlier you have identified the aspect and opinion pair(s) in text: $sentiments, so next please identify the sentiment(s) corresponding to the pair(s). Note that sentiment is one of  positive, neutral and negative. Please output in this format: <'['sentiment1', 'sentiment2']'>.")},

    'rethink':Template("Please think again about $element_type(s) in text, especially focusing on span margins for aspect and opinion or the consistency between sentiment and the pair(s). Note that aspect and opinion are derived from text, and sentiment is one of  positive, neutral and negative. Please output in this format: <'$element_type'>."),

    'conclude':Template('''According to the thought process, please conclude the triplet in text and return it in the format [("aspect", "opinion", "sentiment")]. Each triplet must contain an aspect, an opinion, and a sentiment.
                There may be more than one triplet in a text, please return them completely, returning them in the format: [("aspect1", "opinion1", "sentiment1"),("aspect2", "opinion2", "sentiment2")].
                Note that aspect and opinion are derived from text, and sentiment is one of positive, neutral and negative.''')
}

mapping = {
    'a': 'aspect',
    'o': 'opinion',
    's': 'sentiment'
}

model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct",cache_dir='./LLMs')

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)


dataset_name = 'laptop14'
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

count1 = 0
count2 = 0

print(f'****************{dataset_name}*****************')

path_r = './data/V2/{0}/test.txt'.format(dataset_name)
path_w = './data/V2/{0}/test_predict.txt'.format(dataset_name)
path_w_aos = './data/V2/{0}/test_aos_predict.txt'.format(dataset_name)
path_w_process = './data/V2/{0}/test_process_predict.txt'.format(dataset_name)
data = load_data(path_r)

with open(path_w, 'r', encoding='utf-8') as f_r:
    pre_results_len = len(f_r.read().split('\n'))

if pre_results_len==0:
    prefix = ''
else:
    prefix = '\n'

add_results = []
add_aos_results = []
add_process_results = []
for item in data[pre_results_len:]:
    aspects = []
    opinions = []
    sentiments = []

    count1+=1
    paths = initial()

    path_results = []
    multi_messages = []
    for i in range(len(paths)):
        messages = [
            {"role": "system", "content": "You are an expert in information extraction and sentiment analysis."},
            {"role": "user", "content": trc_frags['dawn'].template}
        ]
        response = forward(messages)
        messages.append({"role": "system", "content": response})

        cur_index = paths[i]['pre_path'].index(paths[i]['cur_node'])
        
        while cur_index<=2:
            if cur_index>2:
                break
            paths[i]['cur_node']=paths[i]['pre_path'][cur_index]
            if paths[i]['cum_rethink'] == -1:
                if paths[i]['cur_node'] == 'a':
                    if cur_index==0:
                        messages.append({"role": "user", "content": trc_frags['aspect'][0].substitute(text=item)})
                    elif cur_index==1:
                        messages.append({"role": "user", "content": trc_frags['aspect'][1].substitute(element_type=mapping[paths[i]['pre_path'][cur_index-1]], aspects=str(aspects))})
                    else:
                        messages.append({"role": "user", "content": trc_frags['aspect'][cur_index].substitute(aspects=str(aspects))})
                    response = forward(messages)
                    aspects = parse_result1(response)
                    messages.append({"role": "system", "content": response})
                elif paths[i]['cur_node'] == 'o':
                    if cur_index==0:
                        messages.append({"role": "user", "content": trc_frags['opinion'][0].substitute(text=item)})
                    elif cur_index==1:
                        messages.append({"role": "user", "content": trc_frags['opinion'][1].substitute(element_type=mapping[paths[i]['pre_path'][cur_index-1]], opinions=str(opinions))})
                    else:
                        messages.append({"role": "user", "content": trc_frags['opinion'][cur_index].substitute(opinions=str(opinions))})
                    response = forward(messages)
                    opinions = parse_result1(response)
                    messages.append({"role": "system", "content": response})
                else:
                    if cur_index==0:
                        messages.append({"role": "user", "content": trc_frags['sentiment'][0].substitute(text=item)})
                    elif cur_index==1:
                        messages.append({"role": "user", "content": trc_frags['sentiment'][1].substitute(element_type=mapping[paths[i]['pre_path'][cur_index-1]], sentiments=str(sentiments))})
                    else:
                        messages.append({"role": "user", "content": trc_frags['sentiment'][cur_index].substitute(sentiments=str(sentiments))})
                    response = forward(messages)
                    sentiments = parse_result1(response)
                    messages.append({"role": "system", "content": response})
                paths[i]['cum_rethink']+=1
            else: 
                while True:
                    if paths[i]['cum_rethink']<max_rethink and bernoulli(aos_vrs[dataset_name][mapping[paths[i]['pre_path'][cur_index]]]): 
                        messages.append({"role": "user", "content": trc_frags['rethink'].substitute(element_type=mapping[paths[i]['pre_path'][cur_index]])})
                        response = forward(messages)
                        if mapping[paths[i]['pre_path'][cur_index]] == 'aspect':
                            aspects = parse_result2(response)
                        elif mapping[paths[i]['pre_path'][cur_index]] == 'opinion':
                            opinions = parse_result2(response)
                        else:
                            sentiments = parse_result2(response)
                        messages.append({"role": "system", "content": response})
                        paths[i]['cum_rethink']+=1
                    else: 
                        cur_index+=1
                        paths[i]['cum_rethink']=-1
                        break
        messages.append({"role": "user", "content": trc_frags['conclude'].template})
        response = forward(messages)
        messages.append({"role": "system", "content": response})
        count2 += len(messages)/2
        multi_messages.append(messages)
        path_results.append(parse_result3(messages))
    add_results.append(item['text']+'####'+unify_paths(selfconsist_line, path_results))
    add_aos_results.append(item['text']+'####'+str(path_results))
    add_process_results.append(item['text']+'####'+str(multi_messages))

    if count1%3==0:
        with open(path_w,'a',encoding='utf-8') as f_write:
            f_write.write(prefix+'\n'.join(add_results))

        with open(path_w_aos,'a',encoding='utf-8') as aos_f_write:
            aos_f_write.write(prefix+'\n'.join(add_aos_results))

        with open(path_w_process,'a',encoding='utf-8') as process_f_write:
            process_f_write.write(prefix+'\n'.join(add_process_results))
        
        add_results = []
        add_aos_results = []
        add_process_results = []

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('count1:', count1)
        print('count2:', count2)
    
with open(path_w,'a',encoding='utf-8') as f_write:
    f_write.write(prefix+'\n'.join(add_results))
with open(path_w_aos,'a',encoding='utf-8') as aos_f_write:
    aos_f_write.write(prefix+'\n'.join(add_aos_results))
with open(path_w_process,'a',encoding='utf-8') as process_f_write:
    process_f_write.write(prefix+'\n'.join(add_process_results))

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print('count1:', count1)
print('count2:', count2)
