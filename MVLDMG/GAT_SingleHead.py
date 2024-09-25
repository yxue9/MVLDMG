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
                 num_heads,  # 这个参数将被保留但不使用
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False):
        super(GATv2Conv, self).__init__()
        self._num_heads = 1  # 内部将其设置为1，实现单头注意力
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # 只创建一个头的权重
        self.fc_src = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        if share_weights:
            self.fc_dst = self.fc_src
        else:
            self.fc_dst = nn.Linear(self._in_src_feats, out_feats, bias=bias)

        self.attn = nn.Parameter(torch.FloatTensor(size=(1, 1, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.activation = activation
        self.share_weights = share_weights
        self.bias = bias
        self.reset_parameters()

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

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph.')

            h_src = h_dst = self.feat_drop(feat)
            feat_src = self.fc_src(h_src)
            feat_dst = self.fc_dst(h_src) if not self.share_weights else feat_src
            graph.srcdata.update({'el': feat_src})
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=-1)
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e('el', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            if self.res_fc is not None:
                resval = self.res_fc(h_dst)
                rst = rst + resval
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
