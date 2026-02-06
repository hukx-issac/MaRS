import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils.common_utils import stop_words
from tools.evaluate import evaluate
from torch import nn

from tqdm import trange
import copy, json

class Trainer():
    def __init__(self, model, tokenizer, trainset, devset, testset, optimizer, criterion, lr_scheduler, args, logging, beta_1, beta_2, bear_max, last, plot=False):
        self.model = model
        self.tokenizer = tokenizer
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.optimizer = optimizer

        self.f_loss = criterion[0]
        self.f_loss1 = criterion[1]
        self.lr_scheduler = lr_scheduler
        self.best_joint_f1 = 0
        self.best_joint_f1_test = 0
        self.best_joint_epoch = 0
        self.best_joint_epoch_test = 0

        self.writer = SummaryWriter()
        self.args = args
        self.logging = logging

        self.evaluate = evaluate
        self.stop_words = stop_words

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.plot = plot

        self.bear_max = bear_max
        self.last = last
        self.contrastive = True

    def train(self):
        bear = 0
        last = self.last

        for i in range(self.args.epochs):

            if self.plot:
                if i % 10 == 0:
                    model = copy.deepcopy(self.model)
                    gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4 = gather_features(model, self.testset)

            self.logging('\n\nEpoch:{}'.format(i+1))
            self.logging(f"contrastive: {self.contrastive} | bear/max: {bear}/{self.bear_max} | last: {last}")

            epoch_sum_loss = []

            for j in trange(self.trainset.batch_count):
                self.model.train()

                sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes = self.trainset.get_batch(j)

                logits, logits1, sim_matrices, add_loss = self.model(bert_tokens, masks, is_train=True, token_classes=token_classes, tagging_matrices=tagging_matrices)

                logits_flatten = logits.reshape([-1, logits.shape[3]])

                tagging_matrices_flatten = tagging_matrices.reshape([-1])
                
                loss0 = self.f_loss(logits_flatten, tagging_matrices_flatten)
                loss0 = self.model.awl(loss0,add_loss)

                tags1 = tagging_matrices.clone()
                tags1[tags1>0] = 1
                logits1_flatten = logits1.reshape([-1, logits1.shape[3]])
                tags1_flatten = tags1.reshape([-1]).to(self.args.device)
                loss1 = self.f_loss1(logits1_flatten.float(), tags1_flatten)

                loss_cl = (sim_matrices * cl_masks).mean()
                
                if self.contrastive:
                    loss = loss0 + self.beta_1 * loss1 + self.beta_2 * loss_cl
                else:
                    loss = loss0 + self.beta_1 * loss1
        
                epoch_sum_loss.append(loss)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('train loss', loss, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss0', loss0, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss1', loss1, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss_cl', loss_cl, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr1', self.optimizer.param_groups[1]['lr'], i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr2', self.optimizer.param_groups[2]['lr'], i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr3', self.optimizer.param_groups[3]['lr'], i*self.trainset.batch_count+j+1)
                
            epoch_avg_loss = sum(epoch_sum_loss) / len(epoch_sum_loss)
            
            self.logging('{}\tAvg loss: {:.10f}'.format(str(datetime.datetime.now()), epoch_avg_loss))
                
            joint_precision, joint_recall, joint_f1, _ = self.evaluate(self.model, self.tokenizer, False, self.devset, self.stop_words, self.logging, self.args)
            joint_precision_test, joint_recall_test, joint_f1_test, test_pred_results = self.evaluate(self.model, self.tokenizer, True, self.testset, self.stop_words, self.logging, self.args)

            self.writer.add_scalar('dev f1', joint_f1, i+1)
            self.writer.add_scalar('test f1', joint_f1_test, i+1)
            self.writer.add_scalar('dev precision', joint_precision, i+1)
            self.writer.add_scalar('test precision', joint_precision_test, i+1)
            self.writer.add_scalar('dev recall', joint_recall, i+1)
            self.writer.add_scalar('test recall', joint_recall_test, i+1)
            self.writer.add_scalar('best dev f1', self.best_joint_f1, i+1)
            self.writer.add_scalar('best test f1', self.best_joint_f1_test, i+1)

            self.lr_scheduler.step()

            self.logging('best epoch: {}\tbest dev {} f1: {:.5f}'.format(self.best_joint_epoch+1, self.args.task, self.best_joint_f1))
            self.logging('best epoch: {}\tbest test {} f1: {:.5f}'.format(self.best_joint_epoch_test+1, self.args.task, self.best_joint_f1_test))

        self.writer.close()

    def test(self):
        bear = 0
        last = self.last
        self.model.eval()
        if self.args.mutant == 'GF':
            mu = 0
            sigma = 0.01
            params = self.model.parameters()
            for param in params:
                noise = torch.randn_like(param) *sigma+mu
                param.detach().add_(noise)
        elif self.args.mutant == 'WS':
            shuffle_percentage = 0.01
            for name,param in self.model.named_parameters():
                if 'weight' in name and isinstance(param,nn.Parameter) and param.requires_grad:
                    weights = param.data
                    shuffle_neurons = int(weights.size(0)*shuffle_percentage)
                    if shuffle_neurons:
                        shuffle_indices = torch.randperm(weights.size(0))[:shuffle_neurons]
                        weights[shuffle_indices] = torch.rand_like(weights[shuffle_indices])
        elif self.args.mutant == 'NEB':
            blocking_percentage = 0.01
            for name,param in self.model.named_parameters():
                if 'weight' in name and isinstance(param,nn.Parameter) and param.requires_grad:
                    weights = param.data
                    blocking_neurons = int(weights.size(0)*blocking_percentage)
                    if blocking_neurons:
                        blocking_indices = torch.randperm(weights.size(0))[:blocking_neurons]
                        weights[blocking_indices] = 0
        elif self.args.mutant == 'NS':
            switch_percentage = 0.01
            for name,param in self.model.named_parameters():
                if 'weight' in name and isinstance(param,nn.Parameter) and param.requires_grad:
                    weights = param.data
                    switch_neurons = int(weights.size(0)*switch_percentage)
                    if switch_neurons>2:
                        switch_indices = torch.randperm(weights.size(0))[:switch_neurons]
                        for i in range(0,switch_neurons-1,2):
                            neuron1_weights = weights[switch_indices[i]].clone()
                            neuron2_weights = weights[switch_indices[i+1]].clone()
                            weights[switch_indices[i]] = neuron2_weights
                            weights[switch_indices[i+1]] = neuron1_weights

        if self.args.mt==0:
            
            joint_precision_test, joint_recall_test, joint_f1_test, test_pred_results = self.evaluate(self.model, self.tokenizer, True, self.testset, self.stop_words, self.logging, self.args)
            if self.args.mutant=='None':
                path = self.args.prefix+self.args.data_version+'/'+self.args.dataset+'/test_predict.txt'
            else:
                path = self.args.prefix+self.args.data_version+'/'+self.args.dataset+f'/test_Mutant={self.args.mutant}_predict.txt'
            with open(path,'w') as file:
                file.write('\n'.join(test_pred_results))
            print(f'P, R, F1: {joint_precision_test}, {joint_recall_test}, {joint_f1_test}')
                
        else:
            joint_precision_test, joint_recall_test, joint_f1_test, test_pred_results = self.evaluate(self.model, self.tokenizer, True, self.testset, self.stop_words, self.logging, self.args)
            if self.args.mutant=='None':
                path = self.args.prefix+self.args.data_version+'/'+self.args.dataset+f'/test_mt{self.args.mt}_predict.txt'
            else:
                path = self.args.prefix+self.args.data_version+'/'+self.args.dataset+f'/test_mt{self.args.mt}_Mutant={self.args.mutant}_predict.txt'
            
            with open(path,'w') as file:
                file.write('\n'.join(test_pred_results))