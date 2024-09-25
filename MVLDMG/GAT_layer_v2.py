# -*- coding: utf-8 -*-
# @Time : 2022/5/23 | 20:07
# @Author : YangCheng
# @Email : yangchengyjs@163.com
# @File : GAT_layer_v2.py
# Software: PyCharm

import torch
import torch as th
from torch import nn

import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair


class GATv2Conv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False):
        super(GATv2Conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias)
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
            if share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=bias)
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.activation = activation
        self.share_weights = share_weights
        self.bias = bias
        self.reset_parameters()

    # 初始化模型参数，使用Xavier初始化方法来初始化全连接层（fc_src和fc_dst）的权重，以及注意力权重（attn）。
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src.bias, 0)
        if not self.share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.fc_dst.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    # 前向传播（forward方法）
    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            graph.srcdata.update({'el': feat_src})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            # 使用DGL的apply_edges方法计算边缘特征，
            # fn.u_add_v表示将源节点和目标节点的特征相加，然后应用LeakyReLU激活函数和注意力权重进行加权。
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # compute softmax
            # 使用edge_softmax方法进行softmax归一化，得到每条边的注意力权重。
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # (num_edge, num_heads)
            # message passing
            # 通过update_all方法进行消息传递，fn.u_mul_e表示将源节点特征和注意力权重相乘，
            # fn.sum表示对所有入边的特征进行求和。
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst