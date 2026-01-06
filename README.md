# 基于 Seq2Seq 的中英机器翻译研究报告

> 课程：自然语言处理  
> 作业：作业 2（基于 Seq2Seq 的机器翻译）  
> 姓名：王政浩
> 学号：2200013134 
> 日期：2026.1.6

## 1. 任务描述
本任务为中英文机器翻译（Machine Translation, MT）：给定中文 token 序列作为输入，生成对应英文 token 序列。数据以“空格分词 + 制表符分隔”的平行语料形式提供：每行一条样本，格式为 `中文tokens\tEnglish tokens`（见 `data/zh_en_{train,val,test}.txt`）。

本次作业要求在普通 RNN Seq2Seq 参考代码基础上完成：
- 二选一：实现基于 **LSTM 或 GRU** 的 Seq2Seq（本项目实现 LSTM）。
- 实现 **Decoder 端对 Encoder 端的 Attention**。
- 限制：不得直接调用 `nn.LSTM` / `nn.LSTMCell`（或同类封装），需基于 `nn.Linear()` 或 `torch.mm()` 手写门控/计算（允许使用 `torch.sigmoid/tanh/softmax` 等激活函数）。

扩展（可选）：实现基于 **Transformer** 的 Seq2Seq 翻译模型（同样以线性层/矩阵乘为主）。

评测指标采用 BLEU（使用 `sacrebleu` 的 corpus BLEU）。

## 2. 模型原理介绍与代码实现思路
本节按“基线 → LSTM → Attention → Transformer”说明原理与实现。

### 2.1 基线：RNN Seq2Seq（Encoder-Decoder）
机器翻译可建模为条件概率分解：

`P(y|x) = ∏_t P(y_t | y_<t, x)`

- **Encoder**：逐 token 读取源句 `x_1...x_L`，递推得到隐状态序列 `h_1...h_L`（以及最终状态）。
- **Decoder**：以 `[BOS]` 为起点自回归生成；训练时采用 teacher forcing（用真实 `y_{t-1}` 作为输入），推理时用上一步预测作为输入。

实现对应：`seq2seq-rnn.py` 中 `Seq2Seq.forward_encoder / forward_decoder / predict`。

### 2.2 手写 LSTM：门控机制与长距离依赖（必答思考题）
普通 RNN 递推（以 tanh 为例）：

`h_t = tanh(W_x x_t + W_h h_{t-1} + b)`

在长序列上容易出现梯度消失/爆炸，导致难以保留远距离信息。

LSTM 引入细胞状态 `c_t` 与门控机制（forget/input/output gates），典型公式为：

- `f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)`（遗忘门）
- `i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)`（输入门）
- `g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)`（候选写入）
- `o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)`（输出门）
- `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`
- `h_t = o_t ⊙ tanh(c_t)`

哪些部分实现了“更长距离依赖”的机制：
- **遗忘门 `f_t`**：控制 `c_{t-1}` 传递比例；当 `f_t≈1` 时信息可以跨多步保留。
- **输入门 `i_t` 与候选 `g_t`**：控制写入强度，避免无关噪声持续污染记忆。
- **细胞状态的加法通路 `c_t = f_t*c_{t-1} + i_t*g_t`**：相对 RNN 的纯非线性叠加，提供更稳定的梯度传播路径，从而更利于长距离依赖学习。
- **输出门 `o_t`**：让“内部记忆”与“外部输出”解耦，保留信息但不必每步都暴露到 `h_t`。

实现思路（对应代码）：
- 在 `LSTMCell` 中将四个门的仿射变换合并为一次线性层输出 `4*hidden_size`，再 `chunk(4)` 切分得到 `i,f,g,o`，最后完成 `c_t/h_t` 更新。全程仅使用 `nn.Linear` 与基础激活函数，满足作业限制。

### 2.3 Decoder 端 Attention：缓解信息瓶颈并提升对齐
无 Attention 的 Seq2Seq 需要将整句信息压缩到单个最终状态，存在“信息瓶颈”。Attention 允许 Decoder 在每个时间步对 Encoder 全部隐状态进行加权汇聚：

- `score_{t,s} = v^T tanh(W_h h_t + W_s h_s)`
- `α_{t,:} = softmax(score_{t,:})`
- `context_t = Σ_s α_{t,s} h_s`

Decoder 输出可同时依赖当前解码状态与上下文：
- `logits_t = Linear([h_t; context_t])`

实现要点（对应代码）：
- `Attention.forward(decoder_hidden, encoder_hiddens, mask)`：返回 `context` 与 `attn_weights`。
- **PAD mask**：对源端 PAD 位置进行 mask（将其 score 置为 `-inf`），避免注意力权重分配到 padding。
- `DecoderRNN` 在 `use_attention=True` 时将 `h_t` 与 `context_t` 拼接后再映射到词表。

### 2.4 扩展：Transformer Seq2Seq（可选）
Transformer 以自注意力替代循环结构，核心组件包括：
- **位置编码（Positional Encoding）**：为 token 表示注入位置信息；
- **多头注意力（Multi-Head Attention）**：并行学习不同子空间的对齐；
- **残差连接 + LayerNorm + FFN**：稳定训练、增强非线性表达；
- **Decoder causal mask**：保证自回归生成时不能“看见未来”。

实现要点（对应代码）：
- `make_src_mask`：屏蔽源端 PAD（形状 `[N,1,1,L]`）。
- `make_tgt_mask`：组合 PAD mask 与下三角 causal mask（形状 `[N,1,L,L]`）。
- 训练对齐：输入 `tgt[:, :-1]`，预测 `tgt[:, 1:]`（输出形状 `[N, Lt-1, V]`）。
- 推理：从 `[BOS]` 开始逐步生成，直到 `[EOS]` 或长度上限。

## 3. 实验设置
### 3.1 数据与预处理
- 数据来源：Tatoeba 中英平行语料。
- 训练/验证/测试划分：默认使用提供的 `train/val/test` 文件。
- 词表：训练集上统计词表，特殊符号包含 `[BOS] [EOS] [UNK] [PAD]`。
- 句长截断与 padding：内容最大长度 `max_len`，最终序列长度为 `max_len + 2`（包含 BOS/EOS），不足用 PAD 补齐。

### 3.2 训练配置（以代码默认参数为准）
- 优化器：Adam
- 学习率：`lr=5e-4`
- 批大小：`batch_size=128`
- 训练轮数：`num_epoch=20`
- 损失函数：`NLLLoss`（log-softmax 输出），且对 `[PAD]` 的 loss 权重置 0
- 解码：贪心（argmax）

Transformer（默认推荐配置，亦与 README 中一致）：
- `num_layers=3`, `num_heads=8`, `hidden_size(d_model)=256`, `ffn_hidden_size=512`, `dropout=0.1`

### 3.3 运行方式
```bash
# Seq2Seq：RNN / LSTM + Attention
python seq2seq-rnn.py --model seq2seq --cell_type rnn --attention
python seq2seq-rnn.py --model seq2seq --cell_type lstm --attention

# Seq2Seq：RNN / LSTM（不带注意力）
python seq2seq-rnn.py --model seq2seq --cell_type rnn
python seq2seq-rnn.py --model seq2seq --cell_type lstm

# Transformer（可选扩展）
python seq2seq-rnn.py --model transformer
```

## 4. 实验结果与分析
实验结果记录在 `result.md`，此处直接引用汇总表（验证集 BLEU 与 GPU 训练时间）：

| Model | #train data | BLEU | Train Time (GPU) |
|---|---:|---:|---:|
| RNN | 26,187 | 2.527552615543479 | 2.66min |
| LSTM | 26,187 | 8.924460616881749 | 3.54min |
| RNN+Att | 26,187 | 13.409741501805792 | 3.91min |
| LSTM+Att | 26,187 | 17.254646597680818 | 6.24min |
| Transformer | 26,187 | 23.995671951614874 | 7.26min |

分析要点：
- **LSTM 相比 RNN**：BLEU 明显提升，说明门控与细胞状态有助于保留更长距离语义信息。
- **Attention 的增益**：对 RNN 与 LSTM 都带来显著提升，说明显式对齐（从 encoder states 中动态取信息）有效缓解信息瓶颈。
- **Transformer 最优**：多头自注意力可直接建模全局依赖，整体表达能力更强，因此 BLEU 最高。
- **效率权衡**：模型越复杂训练时间越长（RNN < LSTM < +Att < Transformer），但带来更高的翻译质量。

## 5. 分工
个人完成

## 6. 参考文献
1. Sutskever, I., Vinyals, O., Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*.
2. Bahdanau, D., Cho, K., Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*.
3. Vaswani, A., et al. (2017). *Attention Is All You Need*.
4. Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*.（sacreBLEU）