import json

def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []

    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                words, tuples = words.split(), eval(tuples)

                tuples.sort(key=lambda x: (x[0][-1], x[1][-1]))
                sents.append(words)
                labels.append(tuples)

    return sents, labels


def read_shot_ratio_from_file(data_path):

    sents, labels = [], []
    transfer = {'positive': 'POS', 'negative': 'NEG', 'neutral': "NEU"}
    with open(data_path, encoding='UTF-8') as fp:
        for line in fp:
            line = json.loads(line)
            tuples = []
            for t in line['relation']:
                sentiment = transfer[t['type']]
                aspect, opinion = None, None
                for ao in t['args']:
                    if ao['type'] == 'aspect':
                        aspect = ao['offset']
                    if ao['type'] == 'opinion':
                        opinion = ao['offset']
                tuples.append((aspect, opinion, sentiment))
            tuples.sort(key=lambda x: (x[0][-1], x[1][-1]))
            sents.append(line['tokens'])
            labels.append(tuples)
    return sents, labels


if __name__ == "__main__":
    data_path = "/home/zhoushen/ABSA/ABSAData/ASTE/rest1456/train.txt"
    read_line_examples_from_file(data_path)
