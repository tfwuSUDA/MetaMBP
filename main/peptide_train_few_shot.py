import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
sys.path.append("../util")
from myutil import *
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(0.2)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, mask= None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, 1e-9)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)

        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class TextCNN(nn.Module):
    """docstring for ClassName"""
    def __init__(self, filter_num=32):
        super(TextCNN, self).__init__()
        self.vocab_size     = 24
        self.channel        = 1
        self.filter_num     = filter_num
        self.emb_dim        = 128
        self.embedding      = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx = 0)
        self.filter_sizes   = [1,2,4,8,16,24,32,64]
        self.convs          = nn.ModuleList([nn.Conv2d(self.channel, self.filter_num, (fsz, self.emb_dim)) for fsz in self.filter_sizes])
        self.bilstm         = nn.LSTM(input_size=self.emb_dim, hidden_size = 64, num_layers=1, bidirectional=True)
        self.attn           = SelfAttention(8, self.emb_dim, self.emb_dim, 0.1)

    def forward(self,x):
        x=x.view(x.size(0),-1)
        mask=(x!=0).int()
        # lstm
        lengths = torch.tensor(np.count_nonzero(x.cpu(), axis=1))
        x_lengths, idx = lengths.sort(0, descending=True)
        _, un_idx = torch.sort(idx, dim=0)
        x = x[idx]
        x = self.embedding(x)
        x = self.attn(x, mask.unsqueeze(1).unsqueeze(2))

        x_packed_input = pack_padded_sequence(input=x, lengths=x_lengths, batch_first=True)
        # print("x_packed_input",x_packed_input)
        packed_out,_=self.bilstm(x_packed_input)
        # print("packedout",packed_out)
        x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=100)
        # print(x.shape)
        # 根据un_idx将输出转回原输入顺序
        # x = torch.index_select(x, 0, un_idx)
        x = torch.index_select(x, 0, un_idx.cuda(0))
        x = x.view(x.size(0), 1, x.size(1), self.emb_dim)
        # print(x.shape)
        x = [F.relu(conv(x)) for conv in self.convs]
        # print('conv x', len(x), [x_item.size() for x_item in x])
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # print('max_pool2d x', len(x), [x_item.size() for x_item in x])
        x = [x_item.view(x_item.size(0), -1).unsqueeze(2) for x_item in x]
        # print('flatten x', len(x), [x_item.size() for x_item in x])
        x = torch.cat(x, 2)
        # print('concat x', x.size())
        return x

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.filter_num = 32
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,self.filter_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(self.filter_num, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(self.filter_num,self.filter_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(self.filter_num, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main(args):
    # Hyper Parameters
    FEATURE_DIM = args.feature_dim
    RELATION_DIM = args.relation_dim
    CLASS_NUM = args.class_num
    SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
    BATCH_NUM_PER_CLASS = args.batch_num_per_class
    EPISODE = args.episode
    TEST_EPISODE = args.test_episode
    LEARNING_RATE = args.learning_rate
    GPU = args.gpu
    FILTER_NUM = args.filter_num
    EMB_DIM = args.emb_dim
    MAX_LEN = args.max_len
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    data_folder = './data/task_data/Meta Dataset/BPD-ALL-RT'  # 推荐使用绝对路径
    metatrain_folders,metatest_folders = tg.peptide_class_folders(data_folder, SEED)

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = TextCNN(FILTER_NUM)
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = CosineAnnealingWarmRestarts(feature_encoder_optim, T_0=250)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = CosineAnnealingWarmRestarts(relation_network_optim,T_0=250)

    feature_encoder_model_path = str(f"./models/LSTM_attn_peptide_feature_encoder_" + str(FILTER_NUM) + "filters_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +f"shot.pkl")
    relation_network_path = str(f"./models/LSTM_attn_peptide_relation_network_" + str(FILTER_NUM) + "filters_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +f"shot.pkl")
    # feature_encoder_model_path = '../result/pretrain_TextCNN/2024-09-24T09:09:45/Epoch_10.pkg'
    if os.path.exists(feature_encoder_model_path):
        feature_encoder = load_params(feature_encoder, feature_encoder_model_path)
        # feature_encoder.load_state_dict(torch.load(str("./models/peptide_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(relation_network_path):
        relation_network = load_params(relation_network, relation_network_path)
        # relation_network.load_state_dict(torch.load(str("./models/peptide_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")
    
    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    from tqdm import tqdm
    for episode in tqdm(range(EPISODE)):
        # init dataset
        # support_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.PeptideTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        support_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False, max_len=MAX_LEN)

        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True, max_len=MAX_LEN)


        # sample datas
        support_inputs,support_labels = support_dataloader.__iter__().__next__()
        batches,batch_labels = batch_dataloader.__iter__().__next__()
        # calculate features
        support_features = feature_encoder(Variable(support_inputs).cuda(GPU))
        support_features = support_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,feature_encoder.filter_num,-1)
        support_features = torch.sum(support_features,1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda(GPU))
        
            
            
        # calculate relations
        # each batch sample link to every samples to calculate relations
        support_features_ext = support_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((support_features_ext,batch_features_ext),2).view(-1,feature_encoder.filter_num*2,len(feature_encoder.filter_sizes)).unsqueeze(1)
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

        mse = nn.MSELoss().cuda(GPU)
        # ce = nn.CrossEntropyLoss()
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()
        feature_encoder_scheduler.step()
        relation_network_scheduler.step()
        


        if (episode+1)%500 == 0:
            print()
            print("episode:",episode+1,"loss",loss.data.item())

        BATCH_SIZE = CLASS_NUM*BATCH_NUM_PER_CLASS
        if episode%5000 == 0:

            # test
            print("Testing...")
            total_rewards = 0
            for i in range(TEST_EPISODE):
                task = tg.PeptideTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                support_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                query_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=False)

                support_inputs,support_labels = support_dataloader.__iter__().__next__()
                query_inputs,query_labels = query_dataloader.__iter__().__next__()
                    
                # calculate features for support input
                with torch.no_grad():
                    support_features = feature_encoder(Variable(support_inputs).cuda(GPU))
                    support_features = support_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,feature_encoder.filter_num,-1)
                    support_features = torch.sum(support_features,1).squeeze(1)
                    query_features = feature_encoder(Variable(query_inputs).cuda(GPU))

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    support_features_ext = support_features.unsqueeze(0).repeat(BATCH_SIZE,1,1,1)

                    query_features_ext = query_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1)
                    query_features_ext = torch.transpose(query_features_ext,0,1)
                    relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,feature_encoder.filter_num*2,len(feature_encoder.filter_sizes)).unsqueeze(1)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                _,predict_labels = torch.max(relations.data,1)

                rewards = [1 if predict_labels[j]==query_labels[j] else 0 for j in range(BATCH_SIZE)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards/1.0/BATCH_SIZE/TEST_EPISODE

            print("test accuracy:",test_accuracy)

            if test_accuracy > last_accuracy:

                # save networks
                # torch.save(feature_encoder.state_dict(),str(f"../models/LSTM_attn_peptide_feature_encoder_"  + str(FILTER_NUM) + "filters_" +  str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +f"shot.pkl"))
                # torch.save(relation_network.state_dict(),str(f"../models/LSTM_attn_peptide_relation_network_"  + str(FILTER_NUM) + "filters_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +f"shot.pkl"))

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy
                # print(f'best_acc: {last_accuracy}')
            print(f'best_acc: {last_accuracy}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Few Shot Peptide Recognition")
    parser.add_argument("-f","--feature_dim",type = int, default = 1024)
    parser.add_argument("-r","--relation_dim",type = int, default = 8)
    parser.add_argument("-w","--class_num",type = int, default = 5)
    parser.add_argument("-s","--sample_num_per_class",type = int, default = 20)
    parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
    parser.add_argument("-e","--episode",type = int, default= 500001)
    parser.add_argument("-t","--test_episode", type = int, default = 300)
    parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
    parser.add_argument("-g","--gpu",type=int, default=0)
    parser.add_argument("-d","--emb_dim",type=int,default=128)
    parser.add_argument("-i","--filter_num",type=int,default=32)
    parser.add_argument("--max_len",type=int,default=100)
    parser.add_argument("--seed",type=int,default=1200)
    args = parser.parse_args()

    SEED = args.seed
    seed_everything(SEED)  # 设置几乎所有的随机种子 随机种子，可使得结果可复现
    main(args)
