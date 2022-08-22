#!usr/bin/env python
# coding:utf-8
"""
our model
"""

import torch
from math import floor
from dataset.single_classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.layers import SumAttention
from util import Type
from model.rnn import RNN
from transformers import BertTokenizer, BertModel

class ResidualBlock(torch.nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = torch.nn.Sequential(
            torch.nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            torch.nn.BatchNorm1d(outchannel),
            torch.nn.Tanh(),
            torch.nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            torch.nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = torch.nn.Sequential(
                        torch.nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        torch.nn.BatchNorm1d(outchannel)
                    )

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        out = self.dropout(out)
        return out

class DocEmbeddingType(Type):
    """Standard names for doc embedding type.
    """
    AVG = 'AVG'
    ATTENTION = 'Attention'
    LAST_HIDDEN = 'LastHidden'

    @classmethod
    def str(cls):
        return ",".join(
            [cls.AVG, cls.ATTENTION, cls.LAST_HIDDEN])

class ArticleEncoder(Classifier):
    def __init__(self, dataset, config):
        super(ArticleEncoder, self).__init__(dataset, config)
        self.doc_embedding_type = config.TextRNN.doc_embedding_type
        self.desc_label = dataset.desc_label
        self.desc_len = dataset.desc_len
        self.rnn = RNN(
            config.embedding.dimension, config.TextRNN.hidden_dimension,
            num_layers=config.TextRNN.num_layers, batch_first=True,
            bidirectional=config.TextRNN.bidirectional,
            rnn_type=config.TextRNN.rnn_type)
        hidden_dimension = config.TextRNN.hidden_dimension
        if config.TextRNN.bidirectional:
            hidden_dimension *= 2
        self.sum_attention = SumAttention(hidden_dimension,
                                          config.TextRNN.attention_dimension,
                                          config.device)
        self.bn = torch.nn.BatchNorm1d(hidden_dimension)

    def forward(self):
        if self.config.feature.feature_names[0] == "token":
            embedding = self.token_embedding(
                self.desc_label.to(self.config.device))
            #length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
            length = self.desc_len.to(self.config.device)
        else:
            embedding = self.char_embedding(
                self.desc_label.to(self.config.device))
            length = batch[cDataset.DOC_CHAR_LEN].to(self.config.device)
        output, last_hidden = self.rnn(embedding, length) #(202,50,200)(batch,desc_len,2*hidden_dim)

        if self.doc_embedding_type == DocEmbeddingType.AVG:
            doc_embedding = torch.sum(output, 1) / length.unsqueeze(1)
        elif self.doc_embedding_type == DocEmbeddingType.ATTENTION:
            doc_embedding = self.sum_attention(output)
        elif self.doc_embedding_type == DocEmbeddingType.LAST_HIDDEN:
            doc_embedding = last_hidden
        else:
            raise TypeError(
                "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                    self.doc_embedding_type, DocEmbeddingType.str()))
        desc = torch.nn.functional.relu(self.bn(doc_embedding))
        return desc

class MultiResCNN_Art(Classifier):
    def __init__(self, dataset, config):
        super(MultiResCNN_Art, self).__init__(dataset, config)
        torch.manual_seed(1337)
        filter_sizes = config.MultiResCNN_Art.kernel_sizes
        self.filter_num = len(filter_sizes)
        hidden = self.filter_num * config.MultiResCNN_Art.num_kernels
        self.convs = torch.nn.ModuleList()
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = torch.nn.ModuleList()
            tmp = torch.nn.Conv1d(config.embedding.dimension, config.embedding.dimension, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            #xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            for idx in range(config.MultiResCNN_Art.conv_layer):
                tmp = ResidualBlock(config.embedding.dimension, config.MultiResCNN_Art.num_kernels, filter_size, 1, True,
                                    config.train.hidden_layer_dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.convs.add_module('channel-{}'.format(filter_size), one_channel)
        #encode the charge label description
        self.Article_encoder = ArticleEncoder(dataset,config)
        #hidden_dimension = 2 * config.TextRNN.hidden_dimension
        #self.bn1 = torch.nn.BatchNorm1d(hidden)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = torch.nn.Linear(hidden, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        #params = super(MultiResCNN_Art, self).get_parameter_optimizer_dict()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.Article_encoder.parameters()})
        params.append({'params': self.final.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        embedding = self.token_embedding(
            batch[cDataset.DOC_TOKEN].to(self.config.device))
        embedding = embedding.transpose(1, 2)  #[batch,embedding_size,seq_length]

        conv_result = []
        for conv in self.convs:
            tmp = embedding
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.nn.functional.relu(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)  #[batch, seq_len,sum(num_filter_maps)]

        # apply relevant article description
        desc = self.Article_encoder()  # [lc, article_num_filters*len(filter_size)][202,200]
        #desc = torch.nn.functional.relu(self.bn1(des)) #(lc, hidden_dimension)
        # apply attention
        alpha = torch.nn.functional.softmax(desc.matmul(x.transpose(1, 2)), dim=2)  # [batch,lc,seq_len]
        #print ('alpha_shape:',alpha.shape)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)  # [256,lc,sum(num_filter_maps)]
        #print (alpha.shape)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)  # [batch,lc]
        return self.dropout(y)

class RemoveAttention(Classifier):
    def __init__(self, dataset, config):
        super(RemoveAttention, self).__init__(dataset, config)
        torch.manual_seed(1337)
        filter_sizes = config.MultiResCNN_Art.kernel_sizes
        self.convs = torch.nn.ModuleList()
        self.filter_num = len(filter_sizes)
        hidden = self.filter_num * config.MultiResCNN_Art.num_kernels
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = torch.nn.ModuleList()
            tmp = torch.nn.Conv1d(config.embedding.dimension, config.embedding.dimension, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            one_channel.add_module('baseconv', tmp)

            for idx in range(config.MultiResCNN_Art.conv_layer):
                tmp = ResidualBlock(config.embedding.dimension, config.MultiResCNN_Art.num_kernels, filter_size, 1, True,
                                    config.train.hidden_layer_dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.convs.add_module('channel-{}'.format(filter_size), one_channel)


        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = torch.nn.Linear(hidden, 183)  #202，183，95
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        #params = super(MultiResCNN_Art, self).get_parameter_optimizer_dict()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.final.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def resdual_block(self, embedding):
        conv_result = []
        for conv in self.convs:
            tmp = embedding
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.nn.functional.relu(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)  #[batch, seq_len,sum(num_filter_maps)]
        return x

    def forward(self, batch):
        embedding = self.token_embedding(
            batch[cDataset.DOC_TOKEN].to(self.config.device))
        embedding = embedding.transpose(1, 2)  #[batch,embedding_size,seq_length]
        x = self.resdual_block(embedding)
        m = torch.sum(x, dim=1)  #(batch,hidden)
        f = self.final(m)

        return self.dropout(f)

class TwoMultiRes(Classifier):
    def __init__(self, dataset, config):
        super(TwoMultiRes, self).__init__(dataset, config)
        torch.manual_seed(1337)
        filter_sizes = config.MultiResCNN_Art.kernel_sizes
        self.convs = torch.nn.ModuleList()
        self.filter_num = len(filter_sizes)
        hidden = self.filter_num * config.MultiResCNN_Art.num_kernels
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = torch.nn.ModuleList()
            tmp = torch.nn.Conv1d(config.embedding.dimension, config.MultiResCNN_Art.num_kernels, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            #xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            for idx in range(2):
                tmp = ResidualBlock(config.MultiResCNN_Art.num_kernels, config.MultiResCNN_Art.num_kernels, filter_size, 1, True,
                                    config.train.hidden_layer_dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.convs.add_module('channel-{}'.format(filter_size), one_channel)
        #encode the charge label description
        self.Article_encoder = ArticleEncoder(dataset,config)
        hidden_dimension = 2 * config.TextRNN.hidden_dimension
        self.bn1 = torch.nn.BatchNorm1d(hidden_dimension)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = torch.nn.Linear(self.filter_num * config.MultiResCNN_Art.num_kernels, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        #params = super(MultiResCNN_Art, self).get_parameter_optimizer_dict()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.Article_encoder.parameters()})
        params.append({'params': self.final.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def resdual_block(self, embedding):
        conv_result = []
        for conv in self.convs:
            tmp = embedding
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.nn.functional.relu(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)  #[batch, seq_len,sum(num_filter_maps)]
        return x

    def forward(self, batch):
        embedding = self.token_embedding(
            batch[cDataset.DOC_TOKEN].to(self.config.device))
        embedding = embedding.transpose(1, 2)  #[batch,embedding_size,seq_length]
        x = self.resdual_block(embedding)
        # apply relevant article description
        des = self.Article_encoder()  # [lc, article_num_filters*len(filter_size)][202,200]
        desc = torch.nn.functional.relu(self.bn1(des)) #(lc, hidden_dimension)
        # apply attention
        alpha = torch.nn.functional.softmax(desc.matmul(x.transpose(1, 2)), dim=2)  # [batch,lc,seq_len]
        #print ('alpha_shape:',alpha.shape)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        v = alpha.matmul(x)  # [256,lc,sum(num_filter_maps)]
        y = self.final.weight.mul(v).sum(dim=2).add(self.final.bias)

        return self.dropout(y)

#encode article content
class Average_article(Classifier):
    def __init__(self, dataset, config):
        super(Average_article, self).__init__(dataset, config)
        self.desc_label = dataset.desc_label
        self.bn = torch.nn.BatchNorm1d(200)
    def forward(self):
        embedding = self.token_embedding(
                self.desc_label.to(self.config.device))  #[202,len,200](202,50,200)
        doc_embedding = embedding.sum(dim=1)
        desc = torch.nn.functional.relu(self.bn(doc_embedding))
        return desc

class Bert_article(Classifier):
    def __init__(self, dataset, config):
        super(Bert_article, self).__init__(dataset, config)
        self.desc_label = dataset.desc_raw.to(self.config.device) #sentence encode,(202,768)
        self.Linear = torch.nn.Linear(768, 200)
        self.bn = torch.nn.BatchNorm1d(200)
    def forward(self):
        d = self.Linear(self.desc_label)
        desc = torch.nn.functional.relu(self.bn(d))
        return desc

class Average_WV(Classifier):
    def __init__(self, dataset, config):
        super(Average_WV, self).__init__(dataset, config)
        torch.manual_seed(1337)
        filter_sizes = config.MultiResCNN_Art.kernel_sizes
        self.filter_num = len(filter_sizes)
        hidden = self.filter_num * config.MultiResCNN_Art.num_kernels
        self.convs = torch.nn.ModuleList()
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = torch.nn.ModuleList()
            tmp = torch.nn.Conv1d(config.embedding.dimension, config.embedding.dimension, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            #xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            for idx in range(config.MultiResCNN_Art.conv_layer):
                tmp = ResidualBlock(config.embedding.dimension, config.MultiResCNN_Art.num_kernels, filter_size, 1, True,
                                    config.train.hidden_layer_dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.convs.add_module('channel-{}'.format(filter_size), one_channel)
        #encode the charge label description
        self.Article_encoder = Average_article(dataset,config)
        #hidden_dimension = 2 * config.TextRNN.hidden_dimension
        #self.bn1 = torch.nn.BatchNorm1d(hidden_dimension)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = torch.nn.Linear(hidden, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        #params = super(MultiResCNN_Art, self).get_parameter_optimizer_dict()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.Article_encoder.parameters()})
        params.append({'params': self.final.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        embedding = self.token_embedding(
            batch[cDataset.DOC_TOKEN].to(self.config.device))
        embedding = embedding.transpose(1, 2)  #[batch,embedding_size,seq_length]

        conv_result = []
        for conv in self.convs:
            tmp = embedding
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.nn.functional.relu(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)  #[batch, seq_len,sum(num_filter_maps)]

        # apply relevant article description
        desc = self.Article_encoder()  # [lc, article_num_filters*len(filter_size)][202,200]
        #desc = torch.nn.functional.relu(self.bn1(des)) #(lc, hidden_dimension)
        # apply attention
        alpha = torch.nn.functional.softmax(desc.matmul(x.transpose(1, 2)), dim=2)  # [batch,lc,seq_len]
        #print ('alpha_shape:',alpha.shape)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)  # [256,lc,sum(num_filter_maps)]
        #print (alpha.shape)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)  # [batch,lc]
        return self.dropout(y)

class Use_Bert(Classifier):
    def __init__(self, dataset, config):
        super(Use_Bert, self).__init__(dataset, config)
        torch.manual_seed(1337)
        filter_sizes = config.MultiResCNN_Art.kernel_sizes
        self.filter_num = len(filter_sizes)
        hidden = self.filter_num * config.MultiResCNN_Art.num_kernels
        self.convs = torch.nn.ModuleList()
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = torch.nn.ModuleList()
            tmp = torch.nn.Conv1d(config.embedding.dimension, config.embedding.dimension, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            #xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            for idx in range(config.MultiResCNN_Art.conv_layer):
                tmp = ResidualBlock(config.embedding.dimension, config.MultiResCNN_Art.num_kernels, filter_size, 1, True,
                                    config.train.hidden_layer_dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.convs.add_module('channel-{}'.format(filter_size), one_channel)
        #encode the charge label description
        self.Article_encoder = Bert_article(dataset,config)
        #hidden_dimension = 2 * config.TextRNN.hidden_dimension
        #self.bn1 = torch.nn.BatchNorm1d(hidden_dimension)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = torch.nn.Linear(hidden, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        #params = super(MultiResCNN_Art, self).get_parameter_optimizer_dict()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.Article_encoder.parameters()})
        params.append({'params': self.final.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        embedding = self.token_embedding(
            batch[cDataset.DOC_TOKEN].to(self.config.device))
        embedding = embedding.transpose(1, 2)  #[batch,embedding_size,seq_length]

        conv_result = []
        for conv in self.convs:
            tmp = embedding
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.nn.functional.relu(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)  #[batch, seq_len,sum(num_filter_maps)]

        # apply relevant article description
        desc = self.Article_encoder()  # [lc, article_num_filters*len(filter_size)][202,200]
        #desc = torch.nn.functional.relu(self.bn1(des)) #(lc, hidden_dimension)
        # apply attention
        alpha = torch.nn.functional.softmax(desc.matmul(x.transpose(1, 2)), dim=2)  # [batch,lc,seq_len]
        #print ('alpha_shape:',alpha.shape)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)  # [256,lc,sum(num_filter_maps)]
        #print (alpha.shape)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)  # [batch,lc]
        return self.dropout(y)

class RemoveMultiRes(Classifier):
    def __init__(self, dataset, config):
        super(RemoveMultiRes, self).__init__(dataset, config)
        torch.manual_seed(1337)
        filter_sizes = config.MultiResCNN_Art.kernel_sizes    #[3,5,7,9]
        self.convs = torch.nn.ModuleList()
        self.filter_num = len(filter_sizes)
        hidden = self.filter_num * config.MultiResCNN_Art.num_kernels   #4*50=200
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            tmp = torch.nn.Conv1d(config.embedding.dimension, config.MultiResCNN_Art.num_kernels, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            self.convs.append(tmp)
        #encode the charge label description
        self.Article_encoder = ArticleEncoder(dataset,config)
        self.bn1 = torch.nn.BatchNorm1d(hidden)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = torch.nn.Linear(hidden, len(dataset.label_map))  #202，183，95
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        #params = super(MultiResCNN_Art, self).get_parameter_optimizer_dict()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.final.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def resdual_block(self, embedding):
        conv_result = []
        for conv in self.convs:
            tmp = embedding
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.nn.functional.relu(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)  #[batch, seq_len,sum(num_filter_maps)]
        return x

    def forward(self, batch):
        embedding = self.token_embedding(
            batch[cDataset.DOC_TOKEN].to(self.config.device))
        embedding = embedding.transpose(1, 2)  #[batch,embedding_size,seq_length]

        conv_result = []
        for i, conv in enumerate(self.convs):
            tmp = torch.nn.functional.relu(conv(embedding))
            tmp = tmp.transpose(1, 2)  #[batch,seq_len,kernel_size=50]
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2) #[batch, seq_len,hidden]

        #apply article description
        des = self.Article_encoder() # [lc, article_num_filters*len(filter_size)][202,200]
        desc = torch.nn.functional.relu(self.bn1(des))  #[lc,hidden]

        #apply attention
        alpha = torch.nn.functional.softmax(desc.matmul(x.transpose(1,2)), dim=2) #[batch,lc,seq_len]
        m = alpha.matmul(x)  #[batch,lc,hidden]

        f = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)  #[batch,lc]

        return self.dropout(f)