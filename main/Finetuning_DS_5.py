import argparse
import datetime
import traceback
import numpy as np
import pandas as pd
import os,random,json,math
import torch.nn as nn
import torch,torch.utils
from dateutil import tz
import torch.nn.functional as F
from rich.progress import track
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import keras
import sys
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

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
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

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        # print(attention_scores.shape)
        # print(mask.shape)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, 1e-9)
        # print(attention_scores)
        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)

        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class TextCNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self, filter_num, emb_dim, class_num=16):
        super(TextCNNEncoder, self).__init__()
        self.vocab_size     = 24
        self.channel        = 1
        self.filter_num     = filter_num
        self.emb_dim        = emb_dim
        self.embedding      = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx = 0)
        self.bilstm         = nn.LSTM(input_size=self.emb_dim, hidden_size = 64, num_layers=1, bidirectional=True)
        self.attn           = SelfAttention(8, self.emb_dim, self.emb_dim, 0.1)
        self.filter_sizes   = [1,2,4,8,16,24,32,64]
        self.convs          = nn.ModuleList([nn.Conv2d(self.channel, self.filter_num, (fsz, self.emb_dim)) for fsz in self.filter_sizes])
        self.linear         = nn.Linear(256, 128)
        self.cls            = nn.Linear(128, class_num)

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
        x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=517)
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

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1).squeeze(-1)  # [256, 288]
        # print('concat x', x.size())
        hidden = self.linear(x)
        out = torch.sigmoid(self.cls(hidden))
        return out
        
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)  # 对模组进行随机数设定
    np.random.seed(seed)  # 对numpy模组进行随机数设定
    torch.manual_seed(seed)  # 对torch中的CPU部分进行随机数设定
    torch.cuda.manual_seed(seed)  # 对torch中的GPU部分进行随机数设定
    torch.cuda.manual_seed_all(seed)  # 当使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True  # 设置每次返回的卷积算法是一致的
    torch.backends.cudnn.benchmark = False  # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False则每次的算法一致
    torch.backends.cudnn.enabled = True  # pytorch 使用CUDANN 加速，即使用GPU加速


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, idx, label):
        self.idx = idx
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return self.idx[i],self.label[i]

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss
class OhemLoss(nn.Module):
    def __init__(self, keep_num):
        super(OhemLoss, self).__init__()
        self.keep_num = keep_num
    def forward(self, pred, target):
        loss = torch.nn.BCELoss(reduce=False)(pred, target)
        loss_sorted, idx = torch.sort(loss, descending=True)
        loss_keep = loss_sorted[:self.keep_num]
        return loss_keep.sum() / self.keep_num
def GetSourceData(root, dir, lb):
    seqs = []
    print('\n')
    print('now is ', dir)
    file = '{}CD_.txt'.format(dir)
    file_path = os.path.join(root, dir, file)

    with open(file_path) as f:
        for each in f:
            if each == '\n' or each[0] == '>':
                continue
            else:
                seqs.append(each.rstrip())

    # data and label
    label = len(seqs) * [lb]
    seqs_train, seqs_test, label_train, label_test = train_test_split(seqs, label, test_size=0.2, random_state=0)
    print('train data:', len(seqs_train))
    print('test data:', len(seqs_test))
    print('train label:', len(label_train))
    print('test_label:', len(label_test))
    print('total numbel:', len(seqs_train)+len(seqs_test))

    return seqs_train, seqs_test, label_train, label_test



def DataClean(data):
    max_len = 0
    for i in range(len(data)):
        st = data[i]
        # get the maximum length of all the sequences
        if(len(st) > max_len): max_len = len(st)

    return data, max_len



def PadEncode(data, max_len):

    # encoding
    # amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    aa_dict = {'A':1, 'B': 20, 'C':2, 'D':3, 'E':4, 'U':3, 'F':5, 'G':6,
               'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'O':11,
               'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'X':20, 'Y':21, 'Z':20, '0':0}
    data_e = []
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i]
        for j in st:
            # index = amino_acids.index(j)
            index = aa_dict[j]
            elemt.append(index)
        if length < max_len:
            elemt += [0]*(max_len-length)
        else:
            elemt = elemt[:max_len]
        data_e.append(elemt)

    return data_e



def GetSequenceData(dirs, root):
    # getting training data and test data
    count, max_length = 0, 0
    tr_data, te_data, tr_label, te_label = [], [], [], []
    for dir in dirs:
        # 1.getting data from file
        tr_x, te_x, tr_y, te_y = GetSourceData(root, dir, count)
        count += 1

        # 2.getting the maximum length of all sequences
        tr_x, len_tr = DataClean(tr_x)
        te_x, len_te = DataClean(te_x)
        if len_tr > max_length: max_length = len_tr
        if len_te > max_length: max_length = len_te

        # 3.dataset
        tr_data += tr_x
        te_data += te_x
        tr_label += tr_y
        te_label += te_y


    # data coding and padding vector to the filling length
    traindata = PadEncode(tr_data, 517)
    testdata = PadEncode(te_data, 517)

    # data type conversion
    train_data = np.array(traindata)
    test_data = np.array(testdata)
    train_label = np.array(tr_label)
    test_label = np.array(te_label)

    return [train_data, test_data, train_label, test_label]



def GetData(path):
    dirs = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP'] # functional peptides

    # get sequence data
    sequence_data = GetSequenceData(dirs, path)

    return sequence_data
def catch(data, label):
    unique_sequences = {}  # 使用字典来存储唯一的序列及其标签
    chongfu = 0  # 统计重复数据数量

    for i in range(len(data)):
        sequence = data[i]
        seq_tuple = tuple(sequence)  # 将 numpy array 转换为 tuple 以便用作字典的键
        # print(f"unique_sequences: {unique_sequences}")
        if seq_tuple in unique_sequences:
            # 如果序列已存在，合并标签
            unique_sequences[seq_tuple] += label[i]
            chongfu += 1
        else:
            # 如果序列不存在，添加新的序列及其标签
            unique_sequences[seq_tuple] = label[i]

    print('Total number of the same data:', chongfu)

    # 将字典转换为去重后的数据和标签
    dedup_data = np.array(list(unique_sequences.keys()))
    dedup_labels = np.array(list(unique_sequences.values()))

    return dedup_data, dedup_labels



def train(args):
    seed_everything(args.seed)  # 设置几乎所有的随机种子 随机种子，可使得结果可复现

    # 生成数据
    data_path = args.dataset

    # idx, label = genIdxforMultiLabel(data_path, args.max_len)
    sequence_data = GetData(data_path)
    train_idx, test_idx, train_label, test_label = sequence_data[0], sequence_data[1], sequence_data[2], sequence_data[3]

    train_label = keras.utils.to_categorical(train_label)
    train_idx, train_label = catch(train_idx, train_label)

    test_label = keras.utils.to_categorical(test_label)
    test_idx, test_label = catch(test_idx, test_label)
    train_label = torch.tensor(train_label, dtype=torch.float)
    test_label = torch.tensor(test_label, dtype=torch.float)
    # train_idx, test_idx, train_label, test_label = train_test_split(idx, label, test_size=0.2, random_state=args.seed)

    meta_model_path = args.mata_model
    
    # 判断是否有GPU
    if args.use_cuda:
        device = torch.device('cuda:'+str(int(args.use_cuda)-1) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print('device: ',device)

    # 构建模型
    # 定义损失函数（使用自定义的Focal Loss）
    if args.loss_fun == "BCE":
        criterion = torch.nn.BCELoss()
    elif args.loss_fun=="Focal":
        criterion = FocalLoss(gamma=2)
    elif args.loss_fun == "Ohem":
        criterion = OhemLoss(keep_num=int(args.batch_size/2))
    elif args.loss_fun == "OhemUW":
        criterion = OhemUncertaintyLoss(keep_num=int(args.batch_size/2))
    elif args.loss_fun == "UW":
        criterion = UncertaintyLoss(args.batch_size)
    model                    = TextCNNEncoder(32, 128, args.class_num).to(device)
    if args.use_meta:
        model                = load_params(model, meta_model_path)
    
    print(model)
    optimizer                = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == 1:
        scheduler            = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)
    elif args.scheduler == 2:
        scheduler            = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.gamma)
    elif args.scheduler == 3:
        scheduler            = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(10))

    if not args.expmode:
        # tensorboard, record the change of auc, acc and loss
        current_time = datetime.datetime.now(tz.gettz('Asia/Shanghai')).strftime("%Y-%m-%dT%H:%M:%S")
        print(current_time)
        # writer = SummaryWriter(log_dir="./runs/"+current_time)
        saveprefix               = f'../result/{args.No_}/{current_time}/'
        saveroot                 = '/'.join(saveprefix.split('/')[:-1])
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)
        # 保存传递参数
        with open(saveroot + '/Hyperparameter.json', "w") as f:
            json.dump(args.__dict__, f, indent=4)
        if args.print_model_tofile:
            with open(saveroot + "/model.txt", "w") as f:
                # 打印模型到model.txt
                print(model, file = f)
                # 打印模型参数
                for params in model.state_dict():   
                    f.write("{}\t{}\n".format(params, model.state_dict()[params]))
    else:
        saveprefix               = f'../result/{args.No_}/{args.timeID}/'
        saveroot                 = '/'.join(saveprefix.split('/')[:-1])

    # 加载数据
    traindataset     = MyDataSet(train_idx, train_label)
    testdataset      = MyDataSet(test_idx, test_label)
    trainloader      = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size)
    testloader       = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size)
    best_auc         = 0
    avg_loss         = 0
    progress_bar     = tqdm(range(1, args.epoch_num+1))
    for epoch in progress_bar:
        model.train()
        arr_loss = []
        for idxes, labels in trainloader:
            optimizer.zero_grad()
            output          = model(idxes.to(device))
            loss            = criterion(output.float(), labels.to(device))
            # print(loss)
            loss.backward()
            optimizer.step()
            arr_loss.append(loss.item())
        avg_loss = np.mean(arr_loss)
        
        model.eval()
        with torch.no_grad():
            preds               = []
            preds_pro           = []
            y_true              = []
            arr_loss            = []
            for idxes, labels in testloader:
                output          = model(idxes.to(device))
                loss            = criterion(output, labels.to(device))
                arr_loss.append(loss.item())
                pred            = (output>0.5).float()
                score           = output
                preds.extend(pred.cpu().detach().data.numpy())
                preds_pro.extend(score.cpu().detach().data.numpy())
                y_true.extend(labels.cpu().detach().data.numpy())
            val_loss                                               = np.mean(arr_loss)
            # 转换为 NumPy 格式，便于使用 sklearn 评估
            preds_pro = torch.tensor(preds_pro)
            Y_pred = np.array(preds)
            Y_test_numpy = np.array(y_true)
        if not args.expmode:
            save_multi_label_metric(epoch, preds_pro, Y_test_numpy, saveroot)
            # save_multi_label_metric(epoch, Y_pred, preds_pro, Y_test_numpy, saveroot)

 
        progress_bar.set_description(
            f'Epoch {epoch} trainLoss:{avg_loss:.4} valLoss:{val_loss:.4}')
        if args.scheduler:
            scheduler.step()
        if args.save_model and int(args.save_model)==epoch:
            torch.save(model, saveroot+f"/epoch{epoch}model.pkg")
            exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input file

    parser.add_argument('-No_', type=str, default='',required = True, help='Name of experienments')

    # parser.add_argument('-train_csv', type=str, default='./dataset/ACP/ACP20mainTrain.csv',help='Path of the sequence training dataset')
    # parser.add_argument('-test_csv', type=str, default='./dataset/ACP/ACP20mainTest.csv',help='Path of the sequence testing dataset')
    parser.add_argument('-class_name', type=str, default="ACP",help="select dataset to fintune")
    parser.add_argument('-dataset', type=str, default="",help="select dataset to fintune")
    parser.add_argument('-loss_fun', type=str, default="",help="select loss function")
    parser.add_argument('-class_num', type=int, default=16)
    
    parser.add_argument('-use_meta', type=int, default=1, help='use meta model')

    parser.add_argument('-gamma', type=float, default=0.96, help='optimizer_gamma')
    parser.add_argument('-dropout', type=float, default=0.5, help='TextCNN dropout')
    parser.add_argument('-step_size', type=int, help='optimizer_step')
    parser.add_argument('-frozen', type=int, default=0, help='if to frozen parameters of model')
    parser.add_argument('-learning_rate', type=float, default=1e-5, help='Learning rate') 
    parser.add_argument('-weight_decay', type=float, default=5e-4, help='L2 rate')
    parser.add_argument('-emb_dim', type=int, default=128, help='Embedding layer dimension')
    parser.add_argument('-epoch_num', type=int, default=2000, help='Maximum number of epochs')
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-max_len', type=int, default=100, help='max_len of seq')
    parser.add_argument('-alpha', type=float, default=0.2, help='value of leakyReLU') 

    parser.add_argument('-mata_model', type=str, default="",
                        help='The path of meta model, if "", the model will be trained from scratch')

    parser.add_argument('-use_cuda', type=int, default=1, help='whether enable cuda & which gpu to utilize')
    parser.add_argument('-seed', type=int, default=1200, help='seed')
    parser.add_argument('-print_model_tofile', type=bool, default=True, help='If to show architecture of model')
    parser.add_argument('-save_model', type=int, default=0, help='If to save model')
    parser.add_argument('-save_best', type=int, default=1, help='If to save best model automatically')
    parser.add_argument('-timeID', type=str, default='', help='ID to save model')
    parser.add_argument('-early_stop', type=int, default=0, help='If to early stop')
    parser.add_argument('-scheduler', type=int, default=0, help='If to enable scheduler')
    parser.add_argument('-expmode', type=int, default=0, help='Experiment Mode')
    args = parser.parse_args()

    from pprint import pprint
    pprint(args.__dict__)
    print("实验模式开" if args.expmode else "实验模式关")
    check = int(input("请确认参数是否正确: (1继续 | 0终止运行)"))
    if(check):
        try:
            start_time = datetime.datetime.now()
            train(args)
        except Exception as e:
            #这个是输出错误类别的，如果捕捉的是通用错误，其实这个看不出来什么
            print('str(Exception):\t', str(Exception))      #输出  str(Exception):	<type 'exceptions.Exception'>
            #这个是输出错误的具体原因，这步可以不用加str，输出 
            print('str(e):\t\t', str(e))   #输出 str(e):		integer division or modulo by zero
            print('repr(e):\t', repr(e)) #输出 repr(e):	ZeroDivisionError('integer division or modulo by zero',)
            print('traceback.print_exc():')
            #以下两步都是输出错误的具体位置的
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            exit()
        finally:
            end_time = datetime.datetime.now()
            print('End time(min):', (end_time - start_time).seconds / 60)
    else:
        print("参数否定, 请重新设置参数")
        exit()
