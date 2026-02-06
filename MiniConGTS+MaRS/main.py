import json
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm import trange
import datetime
import os, random

from utils.common_utils import Logging

from utils.data_utils import load_data_instances
from data.data_preparing import DataIterator

from modules.models.roberta import Model
from modules.f_loss import FocalLoss

from tools.trainer import Trainer

if __name__ == '__main__':
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42,help='seed')
    parser.add_argument('--mt', type=int, default=0, help='metamorphic testing')
    parser.add_argument('--mutant', type=str, default="None", help='mutant type')
    parser.add_argument('--max_sequence_len', type=int, default=100, help='max length of the tagging matrix')
    parser.add_argument('--sentiment2id', type=dict, default={'negative': 2, 'neutral': 3, 'positive': 4}, help='mapping sentiments to ids')
    parser.add_argument('--model_cache_dir', type=str, default='./modules/models/', help='model cache path')
    parser.add_argument('--model_name_or_path', type=str, default='./PLMs/roberta-base', help='reberta model path')
    parser.add_argument('--batch_size', type=int, default=16, help='json data path')
    parser.add_argument('--device', type=str, default="cuda", help='gpu or cpu')
    parser.add_argument('--prefix', type=str, default="./data/", help='dataset and embedding path prefix')

    parser.add_argument('--data_version', type=str, default="D2", choices=["D1", "D2"], help='dataset and embedding path prefix')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"], help='dataset')

    parser.add_argument('--bert_feature_dim', type=int, default=768, help='dimension of pretrained bert feature')
    parser.add_argument('--epochs', type=int, default=200, help='training epoch number')
    parser.add_argument('--class_num', type=int, default=5, help='label number')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"], help='option: pair, triplet')
    parser.add_argument('--model_save_dir', type=str, default="./modules/models/saved_models/", help='model path prefix')
    parser.add_argument('--log_path', type=str, default=None, help='log path')

    args = parser.parse_known_args()[0]
    if args.log_path is None:
        args.log_path = './logs/log_{}_{}_{}.log'.format(args.data_version, args.dataset, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    logging = Logging(file_name=args.log_path).logging


    def seed_torch(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
    seed = args.seed
    seed_torch(seed)
    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logging(f"""
            \n\n
            ========= - * - =========
            date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            seed: {seed}
            ========= - * - =========
            """
        )


    train_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/train.json')))
    dev_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/dev.json')))
    if args.mt==0:
        test_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/test.json')))
    else:
        test_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + f'/test_mt{args.mt}.json')))
    train_instances = load_data_instances(tokenizer, train_sentence_packs, args)
    dev_instances = load_data_instances(tokenizer, dev_sentence_packs, args)
    test_instances = load_data_instances(tokenizer, test_sentence_packs, args)

    trainset = DataIterator(train_instances, args)
    devset = DataIterator(dev_instances, args)
    testset = DataIterator(test_instances, args)


    mapping = {
        'lap14': './modules/models/old_saved_models/D2-lap14-64.7059-seed6-epoch297.pt',
        'res14': './modules/models/old_saved_models/D2-res14-75.7135-seed2-epoch188.pt',
        'res15': './modules/models/old_saved_models/D2-res15-65.8477-seed5-epoch215.pt',
        'res16': './modules/models/saved_models/D2-res16-74.6606-seed5-epoch262.pt',

    }

    model = torch.load(mapping[args.dataset])
    optimizer = torch.optim.Adam([
                    {'params': model.bert.parameters(), 'lr': 1e-5},
                    {'params': model.linear1.parameters(), 'lr': 1e-2},
                    {'params': model.cls_linear.parameters(), 'lr': 1e-3},
                    {'params': model.cls_linear1.parameters(), 'lr': 1e-3}
                ], lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 600, 1000], gamma=0.5, verbose=True)

    weight = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0]).float().cuda()
    f_loss = FocalLoss(weight, ignore_index=-1)

    weight1 = torch.tensor([1.0, 4.0]).float().cuda()
    f_loss1 = FocalLoss(weight1, ignore_index=-1)

    beta_1 = 1e-0
    beta_2 = 1e-1
    bear_max = 10
    last = 10


    trainer = Trainer(model, tokenizer, trainset, devset, testset, optimizer, (f_loss, f_loss1), lr_scheduler, args, logging, beta_1, beta_2, bear_max, last, plot=True)
    trainer.train()
    trainer.test()

    