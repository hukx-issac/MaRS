import torch
import torch.nn.functional as F
from utils.common_utils import Logging
from tools.metric import Metric

from utils.eval_utils import get_triplets_set



def evaluate(model, tokenizer, is_test, dataset, stop_words, logging, args):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        # all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        all_tokenized = []


        for i in range(dataset.batch_count):
            sentence_ids, tokens, masks, token_ranges, tags, tokenized, _, _ = dataset.get_batch(i)

            preds, _, _, _ = model(tokens, masks) 
            preds = torch.argmax(preds, dim=3) 
            all_preds.append(preds) 
            all_labels.append(tags) 

            sens_lens = [len(token_range) for token_range in token_ranges]
            all_sens_lengths.extend(sens_lens) 
            all_token_ranges.extend(token_ranges) 
            all_ids.extend(sentence_ids) 
            all_tokenized.extend(tokenized)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()

        metric = Metric(args, stop_words, all_tokenized, all_ids, all_preds, all_labels, all_sens_lengths, all_token_ranges, ignore_index=-1, logging=logging)
        predicted_set, golden_set = metric.get_sets()
        
        if is_test:
            if args.mt==0:
                path = args.prefix+args.data_version+'/'+args.dataset+'/test.txt'
            else:
                path = args.prefix+args.data_version+'/'+args.dataset+f'/test_mt{args.mt}.txt'
        else:
            path = args.prefix+args.data_version+'/'+args.dataset+'/dev.txt'
        with open(path,'r') as file:
            data = file.read().strip()
        data = [text.split('####')[0] for text in data.split('\n')]
        temp_dict = dict()
        for id in range(len(data)):
            temp_dict[id] = []
        for label in predicted_set:
            id = int(label.split('-')[0])
            temp_dict[id].append('-'.join(label.split('-')[1:]))
        
        pred_results = []
        mapping = {2:'NEG', 3:'NEU', 4:'POS'}
        for id in range(len(data)):
            labels = temp_dict[id]
            words = data[id].split()
            res = []
            for label in sorted(labels):
                a_l, a_r, o_l, o_r, polarity = map(int,label.split('-'))
                aspect = ' '.join(words[a_l:a_r+1])
                opinion = ' '.join(words[o_l:o_r+1])
                sentiment = mapping[polarity]
                res.append((aspect,opinion,sentiment))
            pred_results.append(data[id]+'####'+str(res))
        
        aspect_results = metric.score_aspect(predicted_set, golden_set)
        opinion_results = metric.score_opinion(predicted_set, golden_set)
        pair_results = metric.score_pairs(predicted_set, golden_set)
        
        precision, recall, f1 = metric.score_triplets(predicted_set, golden_set)

        aspect_results = [100 * i for i in aspect_results]
        opinion_results = [100 * i for i in opinion_results]
        pair_results = [100 * i for i in pair_results]
        
        precision = 100 * precision
        recall = 100 * recall
        f1 = 100 * f1
        
        logging('Aspect\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(aspect_results[0], aspect_results[1], aspect_results[2]))
        logging('Opinion\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(opinion_results[0], opinion_results[1], opinion_results[2]))
        logging('Pair\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(pair_results[0], pair_results[1], pair_results[2]))
        logging('Triplet\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1, pred_results
