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
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False,
                 activation=None, allow_zero_in_degree=False, bias=True, share_weights=False):
        super(GATv2Conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)
        if share_weights:
            self.fc_dst = self.fc_src
        else:
            self.fc_dst = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)

        # 初始化注意力权重为固定值，并停止梯度更新
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)), requires_grad=False)
        torch.nn.init.constant_(self.attn, 0.1)  # 初始化为0.1或其他固定值

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.activation = activation
        self.share_weights = share_weights
        self.bias = bias

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, output for those nodes will be invalid.')

            h_src = self.feat_drop(feat)
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            if self.share_weights:
                feat_dst = feat_src
            else:
                feat_dst = self.fc_dst(h_src).view(-1, self._num_heads, self._out_feats)
            graph.srcdata.update({'el': feat_src})
            graph.dstdata.update({'er': feat_dst})

            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # 使用静态注意力计算

            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e('el', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_src).view(h_src.shape[0], -1, self._out_feats)
                rst = rst + resval

            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
