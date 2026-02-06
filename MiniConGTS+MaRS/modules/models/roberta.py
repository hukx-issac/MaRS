import torch
from transformers import RobertaModel
from torch import nn
import torch.nn.functional as F
from .AutomaticWeightedLoss import AutomaticWeightedLoss

class Model(torch.nn.Module):
    def __init__(self, tokenizer, args):
        super(Model, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.bert = RobertaModel.from_pretrained(args.model_name_or_path)
        self.norm0 = torch.nn.LayerNorm(args.bert_feature_dim)
        self.drop_feature = torch.nn.Dropout(0.1)

        self.linear1 = torch.nn.Linear(args.bert_feature_dim*2, self.args.max_sequence_len)
        self.norm1 = torch.nn.LayerNorm(self.args.max_sequence_len)
        self.cls_linear = torch.nn.Linear(self.args.max_sequence_len, args.class_num)
        self.cls_linear1 = torch.nn.Linear(self.args.max_sequence_len, 2)
        self.gelu = torch.nn.GELU()

        self.is_aspect = nn.Linear(self.args.class_num, 2, bias=True)
        self.is_opinion = nn.Linear(self.args.class_num, 2, bias=True)
        self.is_sentiment = nn.Linear(self.args.class_num, 2, bias=True)
        self.awl = AutomaticWeightedLoss()

    def forward(self, tokens, masks, is_train=False, token_classes=None, tagging_matrices=None):
        bert_feature, _ = self.bert(tokens, masks, return_dict=False)
        
        bert_feature = self.norm0(bert_feature)
        bert_feature = bert_feature.unsqueeze(2).expand([-1, -1, self.args.max_sequence_len, -1])
        bert_feature_T = bert_feature.transpose(1, 2)
        
        features = torch.cat([bert_feature, bert_feature_T], dim=3)
        
        
        sim_matrix = torch.nn.functional.cosine_similarity(bert_feature, bert_feature_T, dim=3)
        sim_matrix = sim_matrix * masks
        
        hidden = self.linear1(features)
        hidden = self.norm1(hidden)
        hidden = self.gelu(hidden)

        logits = self.cls_linear(hidden)
        logits1 = self.cls_linear1(hidden)
        
        masks0 = masks.unsqueeze(3).expand([-1, -1, -1, self.args.class_num])
        masks1 = masks.unsqueeze(3).expand([-1, -1, -1, 2])
        
        if is_train:
            aspect_indices = torch.tensor([(i,j,j) for i in range(len(token_classes)) for j in range(len(token_classes[i])) if token_classes[i][j]==1])
            opinion_indices = torch.tensor([(i,j,j) for i in range(len(token_classes)) for j in range(len(token_classes[i])) if token_classes[i][j]==2 or token_classes[i][j]==3 or token_classes[i][j]==4])
            sentiment_indices = torch.where((tagging_matrices==2)|(tagging_matrices==3)|(tagging_matrices==4)|(tagging_matrices==1))
            
            aspect_logits = logits[aspect_indices[:,0],aspect_indices[:,1],aspect_indices[:,2]]
            opinion_logits = logits[opinion_indices[:,0],opinion_indices[:,1],opinion_indices[:,2]]
            sentiment_logits = logits[sentiment_indices]

            aspect_pred = torch.sigmoid(self.is_aspect(aspect_logits))
            aspect_label = torch.ones(aspect_pred.shape[0]).to(logits).long()
            a_loss = F.cross_entropy(aspect_pred,aspect_label,reduction='mean')

            opinion_pred = torch.sigmoid(self.is_opinion(opinion_logits))
            opinion_label = torch.ones(opinion_pred.shape[0]).to(logits).long()
            o_loss = F.cross_entropy(opinion_pred,opinion_label,reduction='mean')

            sentiment_pred = torch.sigmoid(self.is_sentiment(sentiment_logits))
            sentiment_label = torch.ones(sentiment_pred.shape[0]).to(logits).long()
            s_loss = F.cross_entropy(sentiment_pred,sentiment_label,reduction='mean')

            if self.args.dataset == 'lap14':
                alpha, beta, gamma = 0.5222, 0.4740, 0.5203
            elif self.args.dataset == 'res14':
                alpha, beta, gamma = 0.5596, 0.4582, 0.5010
            elif self.args.dataset == 'res15':
                alpha, beta, gamma = 0.5948, 0.4552, 0.5108
            elif self.args.dataset == 'res16':
                alpha, beta, gamma = 0.5502, 0.4639, 0.5169

            additional_loss = alpha*a_loss + beta*o_loss + gamma*s_loss
        else:
            additional_loss = 0

        logits = masks0 * logits
        logits1 = masks1 * logits1

        return logits, logits1, sim_matrix, additional_loss
    
    def weight(self, loss1, loss2):
        return self.awl(loss1,loss2)
    
