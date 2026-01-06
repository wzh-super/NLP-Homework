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
def train_loop(model, optimizer, criterion, loader, device):
    model.train()
    epoch_loss = 0.0
    for src, tgt in tqdm(loader):
        src = src.to(device)
        tgt = tgt.to(device)
        outputs = model(src, tgt)
        loss = criterion(outputs[:,:-1,:].reshape(-1, outputs.shape[-1]), tgt[:,1:].reshape(-1))

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
    parser.add_argument('--cell_type', type=str, default='rnn', choices=['rnn', 'lstm'], help="RNN单元类型: rnn 或 lstm")
    parser.add_argument('--attention', action='store_true', help="是否使用注意力机制")
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

    # 根据参数选择模型配置
    print(f"使用模型配置: cell_type={args.cell_type}, attention={args.attention}")
    model = Seq2Seq(zh_vocab, en_vocab, embedding_dim=256, hidden_size=256, max_len=args.max_len,
                    cell_type=args.cell_type, use_attention=args.attention)
    model.to(device)
    if args.optim=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    weights = torch.ones(len(en_vocab)).to(device)
    weights[en_vocab.word2idx['[PAD]']] = 0 # set the loss of [PAD] to zero
    criterion = nn.NLLLoss(weight=weights)

    # 根据模型配置生成 checkpoint 文件名
    model_name = f"model_{args.cell_type}"
    if args.attention:
        model_name += "_attn"
    model_name += ".pt"

    # 训练
    start_time = time.time()
    best_score = 0.0
    for _ in range(args.num_epoch):
        loss = train_loop(model, optimizer, criterion, trainloader, device)
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