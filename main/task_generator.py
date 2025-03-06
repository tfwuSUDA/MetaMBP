# code is based on https://github.com/katerakelly/pytorch-maml
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from torch.utils.data.sampler import Sampler
SEED = 1200
def read_tsv_data(filename, skip_first=True):
    classname = '/'.join(filename.split('/')[:-1])
    sequences = []
    with open(filename, 'r') as file:
        if skip_first:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(classname+'/'+str(list[2]))
    return sequences
def peptide_class_folders(data_folder, SEED):
    # data_folder = './data/task_data/BPD-ALL-RT/'
    # data_folder = './data/BPD-ALL-RT_HT/'

    # 创建两个列表来存放特定的文件夹
    metatrain_class_folders = []
    metaval_class_folders = []
    class_folders = [os.path.join(data_folder, label) \
                for label in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, label)) \
                ]
    random.seed(SEED)


    #-------------------------- 随机序列随机放入
    # random.shuffle(class_folders)
    # num_train = 30
    # metatrain_class_folders = class_folders[:num_train]
    # metaval_class_folders = class_folders[num_train:]


    #-------------------------- 分别放入随机序列


    # 处理特殊的文件夹
    for folder in class_folders[:]:
        folder_name = os.path.basename(folder)
        if folder_name == "Random Sequence Train":
            metatrain_class_folders.append(folder)
            class_folders.remove(folder)
        elif folder_name == "Random Sequence Test":
            metaval_class_folders.append(folder)
            class_folders.remove(folder)

    # 打乱剩余的文件夹
    random.shuffle(class_folders)

    # 然后按照 num_train 进行分割
    num_train = 30
    metatrain_class_folders.extend(class_folders[:num_train])
    metaval_class_folders.extend(class_folders[num_train:])

    return metatrain_class_folders, metaval_class_folders

class PeptideTask(object):
    '''用来生成每个情景下的支持集和查询集'''
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_class_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, class_folders, num_classes, train_num,test_num):

        self.class_folders = class_folders

        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        # print((self.class_folders,self.num_classes))
        # print("cls folders:", class_folders)
        class_folders = random.sample(self.class_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            tsvfile = [os.path.join(c, x) for x in os.listdir(c)]
            temp = read_tsv_data(tsvfile[0])
            samples[c] = random.sample(temp, len(temp))

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join('/',*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', padding_max_len = 174):
        self.task = task
        self.split = split
        self.padding_max_len = padding_max_len
        self.pep_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.aa_dict = {'A':1, 'B': 20, 'C':2, 'D':3, 'E':4, 'U':3, 'F':5, 'G':6,
                        'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'O':11,
                        'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'X':20, 'Y':21, 'Z':20, '0':0}

    def __len__(self):
        return len(self.pep_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Peptide(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Peptide, self).__init__(*args, **kwargs)
    def __getitem__(self, idx):
        pep = self.pep_roots[idx]
        pep_codes = []
        max_seq_len = self.padding_max_len  # 设置你希望的最大序列长度
        input_seq = pep.split('/')[-1]
        # input_seq = re.sub(r"[ZB]", "X", input_seq)

        # 填充或截断蛋白质序列到指定的 max_seq_len
        if len(input_seq) > max_seq_len:
            pep = input_seq[:max_seq_len]
        else:
            pep = input_seq + '0' * (max_seq_len - len(input_seq))

        current_pep = []
        for aa in pep:
            current_pep.append(self.aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep))
        label = self.labels[idx]
        return rnn_utils.pad_sequence(pep_codes, batch_first=True), torch.tensor(label)

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train',shuffle=True, max_len=100):
    # NOTE: batch size here is # instances PER CLASS

    dataset = Peptide(task,split=split,padding_max_len=max_len)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

