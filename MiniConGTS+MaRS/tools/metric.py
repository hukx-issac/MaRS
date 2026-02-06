import numpy as np

class Metric():
    '''评价指标 precision recall f1'''
    def __init__(self, args, stop_words, tokenized, ids, predictions, goldens, sen_lengths, tokens_ranges, ignore_index=-1, logging=print):
        
        _g = np.array(goldens)
        _g[_g==-1] = 0
        
        print('sum_pred: ', np.array(predictions).sum(), ' sum_gt: ', _g.sum())
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = ignore_index
        self.data_num = len(self.predictions)
        self.ids = ids
        self.stop_words = stop_words
        self.tokenized = tokenized
        self.logging = logging

    def get_spans(self, tags, length, token_range, type):
        spans = []
        start = -1
        for i in range(length):
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_triplet_golden(self, tag):
        triplets = []
        for row in range(1, tag.shape[0]-1):
            for col in range(1, tag.shape[1]-1):
                if row==col:
                    pass
                elif tag[row][col] in self.args.sentiment2id.values():
                    sentiment = int(tag[row][col])
                    al, pl = row, col
                    ar = al
                    pr = pl
                    while tag[ar+1][pr] == 1:
                        ar += 1
                    while tag[ar][pr+1] == 1:
                        pr += 1

                    triplets.append([al, ar, pl, pr, sentiment])
                
        return triplets
    
    def find_triplet(self, tag, ws, tokenized):
        triplets = []
        for row in range(1, tag.shape[0]-1):
            for col in range(1, tag.shape[1]-1):
                if row==col:
                    pass
                elif tag[row][col] in self.args.sentiment2id.values():
                    sentiment = int(tag[row][col])
                    al, pl = row, col
                    ar = al
                    pr = pl
                    while tag[ar+1][pr] == 1:
                        ar += 1
                    while tag[ar][pr+1] == 1:
                        pr += 1
                    
                    '''filting the illegal preds'''
                    
                    condition1 = al in np.array(ws)[:, 0] and ar in np.array(ws)[:, 1] and pl in np.array(ws)[:, 0] and pr in np.array(ws)[:, 1]
                    condition2 = True
                    for ii in range(al, ar+1):
                        for jj in range(pl, pr+1):
                            if ii == jj:
                                condition2 = False
                    
                    condition3 = True
                    for tk in tokenized[al: ar+1]:

                        if tk in self.stop_words:
                            condition3 = False
                            break
                        
                    condition4 = True
                    for tk in tokenized[pl: pr+1]:

                        if tk in self.stop_words:
                            condition4 = False
                            break

                    conditions = condition1 and condition2 and condition3 and condition4                        

                    if conditions:
                        triplets.append([al, ar, pl, pr, sentiment])
                
        return triplets
    
    def get_sets(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):

            id = self.ids[i]
            golden_tuples = self.find_triplet_golden(np.array(self.goldens[i]))

            for golden_tuple in golden_tuples:
                golden_set.add(id + '-' + '-'.join(map(str, golden_tuple)))
                
            tag = np.array(self.predictions[i])

            tag[0][:] = -1
            tag[-1][:] = -1
            tag[:, 0] = -1
            tag[:, -1] = -1

            predicted_triplets = self.find_triplet(tag, self.tokens_ranges[i], self.tokenized[i])  
            for pair in predicted_triplets:
                predicted_set.add(str(id) + '-' + '-'.join(map(str, pair)))
        return predicted_set, golden_set

    def score_triplets(self, predicted_set, golden_set):
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_pairs(self, predicted_set, golden_set):
        predicted_set = set(['-'.join(i.split('-')[0: 5]) for i in predicted_set])
        golden_set = set(['-'.join(i.split('-')[0: 5]) for i in golden_set])
        
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    def score_aspect(self, predicted_set, golden_set):
        predicted_set = set(['-'.join(i.split('-')[0: 3]) for i in predicted_set])
        golden_set = set(['-'.join(i.split('-')[0: 3]) for i in golden_set])

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_opinion(self, predicted_set, golden_set):
        predicted_set = set([i.split('-')[0] + '-' + ('-'.join(i.split('-')[3: 5])) for i in predicted_set])
        golden_set = set([i.split('-')[0] + '-' + ('-'.join(i.split('-')[3: 5])) for i in golden_set])

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
