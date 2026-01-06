from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import argparse, time
import numpy as np

from torch import Tensor


# 加载数据
def load_data(num_train):
    zh_sents = {}
    en_sents = {}
    for split in ['train', 'val', 'test']:
        zh_sents[split] = []
        en_sents[split] = []
        with open(f"data/zh_en_{split}.txt", encoding='utf-8') as f:
            for line in f.readlines():
                zh, en = line.strip().split("\t")
                zh = zh.split()
                en = en.split()
                zh_sents[split].append(zh)
                en_sents[split].append(en)
    num_train = len(zh_sents['train']) if num_train==-1 else num_train
    zh_sents['train'] = zh_sents['train'][:num_train]
    en_sents['train'] = en_sents['train'][:num_train]
    print("训练集 验证集 测试集大小分别为", len(zh_sents['train']), len(zh_sents['val']), len(zh_sents['test']))
    return zh_sents, en_sents


# 构建词表
class Vocab():
    def __init__(self):
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = []
        self.add_word("[BOS]")
        self.add_word("[EOS]")
        self.add_word("[UNK]")
        self.add_word("[PAD]")
    
    def add_word(self, word):
        """
        将单词word加入到词表中
        """
        if word not in self.word2idx:
            self.word2cnt[word] = 0
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        self.word2cnt[word] += 1
    
    def add_sent(self, sent):
        """
        将句子sent中的每一个单词加入到词表中
        sent是由单词构成的list
        """
        for word in sent:
            self.add_word(word)
    
    def index(self, word):
        """
        若word在词表中则返回其下标，否则返回[UNK]对应序号
        """
        return self.word2idx.get(word, self.word2idx["[UNK]"])
    
    def encode(self, sent, max_len):
        """
        在句子sent的首尾分别添加BOS和EOS之后编码为整数序列
        """
        encoded = [self.word2idx["[BOS]"]] + [self.index(word) for word in sent][:max_len] + [self.word2idx["[EOS]"]]
        return encoded
    
    def decode(self, encoded, strip_bos_eos_pad=False):
        """
        将整数序列解码为单词序列
        """
        return [self.idx2word[_] for _ in encoded if not strip_bos_eos_pad or self.idx2word[_] not in ["[BOS]", "[EOS]", "[PAD]"]]
    
    def __len__(self):
        """
        返回词表大小
        """
        return len(self.idx2word)


# 定义模型
class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, hidden_size)
        self.weight_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, hx):
        
        igates = self.weight_ih(input)
        hgates = self.weight_hh(hx)
        ret = torch.tanh(igates + hgates)

        return ret


class LSTMCell(nn.Module):
    """
    手写 LSTM 单元，不使用 nn.LSTM 或 nn.LSTMCell
    LSTM 公式：
        f_t = sigmoid(W_if * x_t + W_hf * h_{t-1} + b_f)  遗忘门
        i_t = sigmoid(W_ii * x_t + W_hi * h_{t-1} + b_i)  输入门
        g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_g)     候选细胞状态
        o_t = sigmoid(W_io * x_t + W_ho * h_{t-1} + b_o)  输出门
        c_t = f_t * c_{t-1} + i_t * g_t                   细胞状态
        h_t = o_t * tanh(c_t)                             隐状态
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入到四个门的线性变换 (合并为一个矩阵提高效率)
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        # 隐状态到四个门的线性变换
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, hx):
        """
        input: N * input_size
        hx: (h, c) 元组，h 和 c 都是 N * hidden_size
        返回: (h_new, c_new)
        """
        h_prev, c_prev = hx

        # 计算所有门的值
        gates = self.weight_ih(input) + self.weight_hh(h_prev)  # N * (4 * hidden_size)

        # 分割为四个门
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # 应用激活函数
        i_t = torch.sigmoid(i_gate)  # 输入门
        f_t = torch.sigmoid(f_gate)  # 遗忘门
        g_t = torch.tanh(g_gate)     # 候选细胞状态
        o_t = torch.sigmoid(o_gate)  # 输出门

        # 更新细胞状态和隐状态
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class Attention(nn.Module):
    """
    Bahdanau Attention (加性注意力)
    score(h_t, h_s) = v^T * tanh(W_h * h_t + W_s * h_s)
    """
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_hiddens, mask=None):
        """
        decoder_hidden: N * H (当前解码器隐状态)
        encoder_hiddens: N * L * H (编码器所有时间步的隐状态)
        mask: N * L (True 表示 PAD 位置，需要被屏蔽)

        返回:
            context: N * H (上下文向量)
            attn_weights: N * L (注意力权重)
        """
        # decoder_hidden: N * H -> N * 1 * H
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # 计算注意力分数
        # W_h(decoder_hidden): N * 1 * H
        # W_s(encoder_hiddens): N * L * H
        score = self.v(torch.tanh(self.W_h(decoder_hidden) + self.W_s(encoder_hiddens)))  # N * L * 1
        score = score.squeeze(2)  # N * L

        # 对 PAD 位置进行 mask，将其分数设为负无穷
        if mask is not None:
            score = score.masked_fill(mask, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(score, dim=1)  # N * L

        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_hiddens)  # N * 1 * H
        context = context.squeeze(1)  # N * H

        return context, attn_weights


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, cell_type='rnn'):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        if cell_type == 'lstm':
            self.rnn = LSTMCell(embedding_dim, hidden_size)
        else:
            self.rnn = RNNCell(embedding_dim, hidden_size)

    def forward(self, input, hidden):
        """
        input: N
        hidden: N * H (RNN) 或 (h, c) 元组 (LSTM)

        输出更新后的隐状态hidden
        """
        embedding = self.embed(input)
        hidden = self.rnn(embedding, hidden)
        return hidden

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, cell_type='rnn', use_attention=False):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.use_attention = use_attention

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        if cell_type == 'lstm':
            self.rnn = LSTMCell(embedding_dim, hidden_size)
        else:
            self.rnn = RNNCell(embedding_dim, hidden_size)

        if use_attention:
            self.attention = Attention(hidden_size)
            # 使用 attention 时，输出层的输入是 hidden + context，所以是 2 * hidden_size
            self.h2o = nn.Linear(2 * hidden_size, vocab_size)
        else:
            self.h2o = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_hiddens=None, attn_mask=None):
        """
        input: N
        hidden: N * H (RNN) 或 (h, c) 元组 (LSTM)
        encoder_hiddens: N * L * H (仅在 use_attention=True 时使用)
        attn_mask: N * L (True 表示 PAD 位置)

        输出对于下一个时间片的预测output（大小为N * V）和更新后的隐状态hidden
        """
        embedding = self.embed(input)
        hidden = self.rnn(embedding, hidden)

        # 获取当前隐状态用于 attention 和输出
        if self.cell_type == 'lstm':
            h_for_output = hidden[0]  # LSTM 返回 (h, c)，用 h 计算 attention
        else:
            h_for_output = hidden

        if self.use_attention and encoder_hiddens is not None:
            context, attn_weights = self.attention(h_for_output, encoder_hiddens, attn_mask)
            # 将隐状态和上下文向量拼接
            combined = torch.cat((h_for_output, context), dim=1)  # N * (2H)
            output = self.h2o(combined)
        else:
            output = self.h2o(h_for_output)

        output = self.softmax(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len, cell_type='rnn', use_attention=False):
        super(Seq2Seq, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.use_attention = use_attention
        self.encoder = EncoderRNN(len(src_vocab), embedding_dim, hidden_size, cell_type)
        self.decoder = DecoderRNN(len(tgt_vocab), embedding_dim, hidden_size, cell_type, use_attention)
        self.max_len = max_len

    def init_hidden(self, batch_size):
        """
        初始化编码器端隐状态
        RNN: 全0向量 (N * H)
        LSTM: (h, c) 元组，都是全0向量 (N * H)
        """
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        if self.cell_type == 'lstm':
            c = torch.zeros(batch_size, self.hidden_size).to(device)
            return (h, c)
        return h

    def init_tgt_bos(self, batch_size):
        """
        预测时，初始化解码器端输入为[BOS]（大小为batch_size）
        """
        device = next(self.parameters()).device
        return (torch.ones(batch_size)*self.tgt_vocab.index("[BOS]")).long().to(device)

    def forward_encoder(self, src):
        """
        src: N * L
        编码器前向传播，输出最终隐状态hidden和隐状态序列encoder_hiddens (N * L * H)
        """
        Bs, Ls = src.size()
        hidden = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        # 编码器端每个时间片，取出输入单词的下标，与上一个时间片的隐状态一起送入encoder，得到更新后的隐状态
        for i in range(Ls):
            input = src[:, i]
            hidden = self.encoder(input, hidden)
            # 对于 LSTM，encoder_hiddens 只存储 h，不存储 c
            if self.cell_type == 'lstm':
                encoder_hiddens.append(hidden[0])
            else:
                encoder_hiddens.append(hidden)
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        return hidden, encoder_hiddens

    def forward_decoder(self, tgt, hidden, encoder_hiddens=None, attn_mask=None):
        """
        tgt: N * Lt
        hidden: N * H (RNN) 或 (h, c) 元组 (LSTM)
        encoder_hiddens: N * L * H (用于 attention)
        attn_mask: N * L (True 表示 PAD 位置)

        解码器前向传播，用于训练，使用teacher forcing，输出预测结果outputs，大小为N * L * V
        """
        Bs, Lt = tgt.size()
        outputs = []
        for i in range(Lt):
            input = tgt[:, i]    # teacher forcing, 使用标准答案的单词作为输入，而非模型预测值
            output, hidden = self.decoder(input, hidden, encoder_hiddens, attn_mask)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


    def forward(self, src, tgt):
        """
            src: N * Ls
            tgt: N * Lt

            训练时的前向传播
        """
        hidden, encoder_hiddens = self.forward_encoder(src)
        # 生成 attention mask: PAD 位置为 True
        attn_mask = None
        if self.use_attention:
            attn_mask = (src == self.src_vocab.index("[PAD]"))
        outputs = self.forward_decoder(tgt, hidden, encoder_hiddens if self.use_attention else None, attn_mask)
        return outputs

    def predict(self, src):
        """
            src: 1 * Ls

            用于预测，解码器端初始输入为[BOS]，之后每个位置的输入为上个时间片预测概率最大的单词
            当解码长度超过self.max_len+2或预测出了[EOS]时解码终止
            输出预测的单词编号序列，大小为1 * L，L为预测长度
        """
        hidden, encoder_hiddens = self.forward_encoder(src)
        # 生成 attention mask: PAD 位置为 True
        attn_mask = None
        if self.use_attention:
            attn_mask = (src == self.src_vocab.index("[PAD]"))
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = [input]
        # max_len + 2 与训练时 tgt 长度一致（包含 BOS 和 EOS）
        while len(preds) < self.max_len + 2:
            output, hidden = self.decoder(input, hidden, encoder_hiddens if self.use_attention else None, attn_mask)
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        preds = torch.stack(preds, dim=-1)
        return preds

class PositionalEncoding(nn.Module):
    """
    位置编码: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [N, L, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    """
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.scale = d_k ** 0.5
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q: [N, num_heads, L_q, d_k]
        K: [N, num_heads, L_k, d_k]
        V: [N, num_heads, L_k, d_v]
        mask: [N, 1, 1, L_k] 或 [N, 1, L_q, L_k]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [N, heads, L_q, L_k]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        # 处理全 mask 情况：当某行全为 -inf 时，softmax 会产生 NaN，将其替换为 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # [N, heads, L_q, d_v]
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [N, L, d_model]
        mask: [N, 1, 1, L_k] 或 [N, 1, L_q, L_k]
        """
        N = q.size(0)

        # 线性投影 + reshape 成多头: [N, L, d_model] -> [N, num_heads, L, d_k]
        Q = self.W_q(q).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        out, attn = self.attention(Q, K, V, mask)

        # 拼接多头: [N, num_heads, L, d_k] -> [N, L, d_model]
        out = out.transpose(1, 2).contiguous().view(N, -1, self.d_model)
        out = self.W_o(out)

        return out, attn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer: Self-Attention + FFN
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, src_mask=None):
        # Self-Attention + Add & Norm
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer: Masked Self-Attention + Cross-Attention + FFN
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Masked Self-Attention + Add & Norm
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Cross-Attention + Add & Norm
        attn_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_out))

        # FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    """
    def __init__(self, src_vocab, tgt_vocab, d_model=256, num_heads=8, num_layers=3,
                 d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.d_model = d_model
        self.max_len = max_len

        # Embedding layers
        self.src_embedding = nn.Embedding(len(src_vocab), d_model)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(d_model, len(tgt_vocab))
        self.softmax = nn.LogSoftmax(dim=-1)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """
        src: [N, L]
        返回: [N, 1, 1, L] - PAD 位置为 0，其他为 1
        """
        src_mask = (src != self.src_vocab.index("[PAD]")).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        tgt: [N, L]
        返回: [N, 1, L, L] - 结合 PAD mask 和 causal mask
        """
        N, L = tgt.size()
        # PAD mask: [N, 1, 1, L]
        pad_mask = (tgt != self.tgt_vocab.index("[PAD]")).unsqueeze(1).unsqueeze(2)
        # Causal mask (下三角): [1, 1, L, L]
        causal_mask = torch.tril(torch.ones((1, 1, L, L), device=tgt.device)).bool()
        # 组合两个 mask
        tgt_mask = pad_mask & causal_mask
        return tgt_mask

    def encode(self, src, src_mask):
        """
        Encoder forward
        src: [N, L]
        """
        x = self.src_embedding(src) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, enc_output, tgt_mask, src_mask):
        """
        Decoder forward
        tgt: [N, L]
        """
        x = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        return x

    def forward(self, src, tgt):
        """
        训练时的前向传播
        src: [N, Ls]
        tgt: [N, Lt]
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt[:, :-1])  # 去掉最后一个 token

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt[:, :-1], enc_output, tgt_mask, src_mask)

        output = self.fc_out(dec_output)
        output = self.softmax(output)

        return output

    def predict(self, src):
        """
        预测时的自回归解码
        src: [1, Ls]
        """
        self.eval()
        device = src.device
        src_mask = self.make_src_mask(src)
        enc_output = self.encode(src, src_mask)

        # 初始化解码输入为 [BOS]
        tgt = torch.LongTensor([[self.tgt_vocab.index("[BOS]")]]).to(device)

        # 使用 while 循环，确保 tgt 长度不超过 max_len（避免位置编码越界）
        while tgt.size(1) < self.max_len:
            tgt_mask = self.make_tgt_mask(tgt)
            dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
            output = self.fc_out(dec_output[:, -1, :])  # 只取最后一个位置
            next_token = output.argmax(dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == self.tgt_vocab.index("[EOS]"):
                break

        return tgt


# 构建Dataloader
def collate(data_list):
    src = torch.stack([torch.LongTensor(_[0]) for _ in data_list])
    tgt = torch.stack([torch.LongTensor(_[1]) for _ in data_list])
    return src, tgt

def padding(inp_ids, max_len, pad_id):
    max_len += 2    # include [BOS] and [EOS]
    ids_ = np.ones(max_len, dtype=np.int32) * pad_id
    max_len = min(len(inp_ids), max_len)
    ids_[:max_len] = inp_ids
    return ids_

def create_dataloader(zh_sents, en_sents, max_len, batch_size, pad_id):
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = True if split=='train' else False
        datas = [(padding(zh_vocab.encode(zh, max_len), max_len, pad_id), padding(en_vocab.encode(en, max_len), max_len, pad_id)) for zh, en in zip(zh_sents[split], en_sents[split])]
        dataloaders[split] = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

# 训练、测试函数
def train_loop(model, optimizer, criterion, loader, device, is_transformer=False):
    model.train()
    epoch_loss = 0.0
    for src, tgt in tqdm(loader):
        src = src.to(device)
        tgt = tgt.to(device)
        outputs = model(src, tgt)

        if is_transformer:
            # Transformer forward 输入 tgt[:, :-1]，输出 [N, Lt-1, V]
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt[:, 1:].reshape(-1))
        else:
            # Seq2Seq forward 输出 [N, Lt, V]，需要去掉最后一个位置
            loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), tgt[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)     # 裁剪梯度，将梯度范数裁剪为1，使训练更稳定
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(loader)
    return epoch_loss

def test_loop(model, loader, tgt_vocab, device):
    model.eval()
    bleu = BLEU(force=True)
    hypotheses, references = [], []
    for src, tgt in tqdm(loader):
        B = len(src)
        for _ in range(B):
            _src = src[_].unsqueeze(0).to(device)     # 1 * L
            with torch.no_grad():
                outputs = model.predict(_src)         # 1 * L
            
            # 保留预测结果，使用词表vocab解码成文本，并删去BOS与EOS
            ref = " ".join(tgt_vocab.decode(tgt[_].tolist(), strip_bos_eos_pad=True))
            hypo = " ".join(tgt_vocab.decode(outputs[0].cpu().tolist(), strip_bos_eos_pad=True))
            references.append(ref)    # 标准答案
            hypotheses.append(hypo)   # 预测结果
    
    score = bleu.corpus_score(hypotheses, [references]).score      # 计算BLEU分数
    return hypotheses, references, score


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train', type=int, default=-1, help="训练集大小，等于-1时将包含全部训练数据")
    parser.add_argument('--max_len', type=int, default=10, help="句子最大长度")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005)
    # Seq2Seq 参数
    parser.add_argument('--cell_type', type=str, default='rnn', choices=['rnn', 'lstm'], help="RNN单元类型: rnn 或 lstm")
    parser.add_argument('--attention', action='store_true', help="是否使用注意力机制")
    # 模型选择
    parser.add_argument('--model', type=str, default='seq2seq', choices=['seq2seq', 'transformer'], help="模型类型")
    # Transformer 参数
    parser.add_argument('--num_layers', type=int, default=3, help="Transformer层数")
    parser.add_argument('--num_heads', type=int, default=8, help="注意力头数")
    parser.add_argument('--hidden_size', type=int, default=256, help="隐藏层大小")
    parser.add_argument('--ffn_hidden_size', type=int, default=512, help="FFN隐藏层大小")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout率")
    args = parser.parse_args()

    zh_sents, en_sents = load_data(args.num_train)

    zh_vocab = Vocab()
    en_vocab = Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    print("中文词表大小为", len(zh_vocab))
    print("英语词表大小为", len(en_vocab))

    trainloader, validloader, testloader = create_dataloader(zh_sents, en_sents, args.max_len, args.batch_size, pad_id=zh_vocab.word2idx['[PAD]'])

    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 根据参数选择模型
    is_transformer = (args.model == 'transformer')

    if is_transformer:
        print(f"使用 Transformer 模型: num_layers={args.num_layers}, num_heads={args.num_heads}, "
              f"hidden_size={args.hidden_size}, ffn_hidden_size={args.ffn_hidden_size}, dropout={args.dropout}")
        model = Transformer(
            src_vocab=zh_vocab,
            tgt_vocab=en_vocab,
            d_model=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.ffn_hidden_size,
            dropout=args.dropout,
            max_len=args.max_len + 2
        )
        model_name = "model_transformer.pt"
    else:
        print(f"使用 Seq2Seq 模型: cell_type={args.cell_type}, attention={args.attention}")
        model = Seq2Seq(zh_vocab, en_vocab, embedding_dim=256, hidden_size=256, max_len=args.max_len,
                        cell_type=args.cell_type, use_attention=args.attention)
        model_name = f"model_{args.cell_type}"
        if args.attention:
            model_name += "_attn"
        model_name += ".pt"

    model.to(device)
    if args.optim=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    weights = torch.ones(len(en_vocab)).to(device)
    weights[en_vocab.word2idx['[PAD]']] = 0 # set the loss of [PAD] to zero
    criterion = nn.NLLLoss(weight=weights)

    # 训练
    start_time = time.time()
    best_score = 0.0
    for _ in range(args.num_epoch):
        loss = train_loop(model, optimizer, criterion, trainloader, device, is_transformer)
        hypotheses, references, bleu_score = test_loop(model, validloader, en_vocab, device)
        # 保存验证集上bleu最高的checkpoint
        if bleu_score > best_score:
            torch.save(model.state_dict(), model_name)
            best_score = bleu_score
        print(f"Epoch {_}: loss = {loss}, valid bleu = {bleu_score}")
        print(references[0])
        print(hypotheses[0])
    end_time = time.time()

    #测试
    # 仅加载权重，避免 torch.load 默认 pickle 带来的安全警告（也更安全）
    try:
        state_dict = torch.load(model_name, map_location=device, weights_only=True)
    except TypeError:
        # 兼容旧版 PyTorch（不支持 weights_only 参数）
        state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)
    hypotheses, references, bleu_score = test_loop(model, testloader, en_vocab, device)
    print(f"Test bleu = {bleu_score}")
    print(references[0])
    print(hypotheses[0])
    print(f"Training time: {round((end_time - start_time)/60, 2)}min")