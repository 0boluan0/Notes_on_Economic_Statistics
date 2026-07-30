---
aliases:
  - "Transformer"
  - "Transformer 架构与注意力机制"
  - "Transformer Architecture and Attention"
status: source-checked
---

# Transformer 与注意力机制
<!-- bilingual-en:start -->
*Transformer Architecture and Attention*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 让序列中的每个 token 按当前任务直接选择并汇总其他 token 的信息，而不必像循环网络那样逐步传递。
> **具体锚点：** 在“银行没有批准贷款，因为它风险太高”中，“它”应重点读取“贷款”及其上下文；attention 学习这种数据依赖的读取权重。
> **核心难点：** attention 不是“理解力”本身，而是按 query–key 相似度加权 value；模型仍需位置、因果掩码、残差和训练目标才能成为语言模型。
> **为什么重要：** 预训练、长上下文、KV cache、多模态和大多数现代 LLM 都从这个计算骨架展开。
> **继续：** 先掌握一次 self-attention 的张量流，再读 [[LLM 预训练#自回归目标怎样产生训练信号|预训练]]；部署时转到 [[LLM 推理效率#Prefill 与 decoding|推理阶段]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Let every token in a sequence select and aggregate information from other tokens directly, instead of passing information step by step through a recurrent state.
> **Concrete anchor:** In “The bank did not approve the loan because it was too risky,” the representation of “it” should read strongly from “loan” and its context; attention learns such input-dependent reading weights.
> **Central difficulty:** Attention is not understanding by itself. It mixes values according to query–key compatibility; positions, masks, residual pathways, feed-forward computation, and a training objective are still required.
> **Why it matters:** Pretraining, long-context methods, the KV cache, multimodal models, and most modern language models build on this computational skeleton.
> **Continue with:** Trace one self-attention pass below, then use [[LLM 预训练#自回归目标怎样产生训练信号|LLM Pretraining]] for the learning objective and [[LLM 推理效率#Prefill 与 decoding|LLM Inference Efficiency]] for deployment.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
> <!-- bilingual-en:start -->
> - [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and original papers.
> - The corresponding original papers in Zotero verify the architecture, training methods, experimental conditions, and conclusions; paper notes do not replace the primary papers.
> <!-- bilingual-en:end -->

## Token 与表示
<!-- bilingual-en:start -->
*Tokens and Representations*
<!-- bilingual-en:end -->

文本先被 tokenizer 切成 token 并映射为向量。tokenization 决定序列长度和词表粒度，但 token 本身没有固定“词义”；上下文表示由多层变换形成。输入表示还要携带位置信息，否则 attention 对排列近似不敏感。
<!-- bilingual-en:start -->
Text is first divided into tokens and mapped to vectors. Tokenization determines sequence length and vocabulary granularity, but a token has no fixed meaning by itself; contextual representations emerge through successive layers. Inputs also need position information because attention without it is largely insensitive to token order.
<!-- bilingual-en:end -->

设批量维省略，输入为 $X\in\mathbb{R}^{n\times d_{model}}$：$n$ 是序列长度，$d_{model}$ 是每个 token 的表示维度。token embedding 与位置表示相加或通过旋转等方式结合后，才进入第一个 Transformer block。
<!-- bilingual-en:start -->
Ignoring the batch dimension, let the input be $X\in\mathbb{R}^{n\times d_{model}}$, where $n$ is sequence length and $d_{model}$ is the representation width. Token embeddings are combined with positional information, by addition or a mechanism such as rotation, before entering the first Transformer block.
<!-- bilingual-en:end -->

## Self-attention
<!-- bilingual-en:start -->
*Self-Attention*
<!-- bilingual-en:end -->

对表示矩阵 $X$ 做线性投影得到 $Q=XW_Q$、$K=XW_K$、$V=XW_V$：
<!-- bilingual-en:start -->
Linear projections of the representation matrix $X$ produce $Q=XW_Q$, $K=XW_K$, and $V=XW_V$:
<!-- bilingual-en:end -->

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

每个 query 与所有 key 比较，softmax 后形成读取 value 的权重。缩放项控制点积方差；$M$ 可屏蔽未来位置或 padding。权重能说明本层如何路由信息，但不能自动等同于完整、稳定的因果解释。
<!-- bilingual-en:start -->
Each query is compared with every key; softmax turns the scores into weights used to read the values. Scaling controls the variance of dot products, and $M$ can mask future or padded positions. The weights describe information routing in that layer, but they are not automatically a complete or stable causal explanation of the model's prediction.
<!-- bilingual-en:end -->

这三个角色不是三个固定语义槽。query 表示当前位置在本层想检索的特征，key 表示每个候选位置可怎样被匹配，value 是被汇总的内容；它们的含义由训练得到的投影矩阵共同决定。
<!-- bilingual-en:start -->
The three roles are not fixed semantic slots. A query expresses what the current position seeks in this layer, a key expresses how each candidate position can be matched, and a value carries the content to aggregate. Their meanings are jointly learned through the projection matrices.
<!-- bilingual-en:end -->

## Worked example：手算一次注意力读取
<!-- bilingual-en:start -->
*Worked Example: Compute One Attention Read*
<!-- bilingual-en:end -->

假设某个 query 与三个 key 的缩放点积为 $[2,1,0]$，且没有掩码。softmax 后权重约为 $[0.665,0.245,0.090]$。若三个一维 value 分别是 $[10,0,-5]$，输出约为 $0.665\times10+0.245\times0+0.090\times(-5)=6.20$。这说明 attention 输出是内容依赖的加权汇总，而不是复制得分最高的单个 token。
<!-- bilingual-en:start -->
Suppose the scaled dot products between one query and three keys are $[2,1,0]$, with no mask. Softmax gives weights of approximately $[0.665,0.245,0.090]$. If the three scalar values are $[10,0,-5]$, the output is about $0.665\times10+0.245\times0+0.090\times(-5)=6.20$. Attention therefore performs a content-dependent weighted aggregation rather than copying only the highest-scoring token.
<!-- bilingual-en:end -->

在 causal self-attention 中，位置 $t$ 对所有 $j>t$ 的分数加 $-\infty$，softmax 后这些未来权重为零。训练时整句已知，所有位置仍可在一个矩阵运算中并行计算；掩码只是阻止每个位置偷看其预测目标之后的 token。
<!-- bilingual-en:start -->
In causal self-attention, position $t$ receives a score of $-\infty$ for every $j>t$, so future weights become zero after softmax. During training the entire sequence is available and all positions can still be computed in one matrix operation; the mask merely prevents each position from seeing tokens beyond its prediction target.
<!-- bilingual-en:end -->

## Multi-head、残差与前馈层
<!-- bilingual-en:start -->
*Multi-Head Attention, Residual Paths, and Feed-Forward Layers*
<!-- bilingual-en:end -->

多头注意力在不同投影子空间并行读取，再拼接；逐位置前馈网络对每个位置施加相同的非线性变换。残差连接和 normalization 稳定深层训练。层层堆叠后，一个 token 可组合多步上下文关系。
<!-- bilingual-en:start -->
Multi-head attention performs parallel reads in different projected subspaces and concatenates their outputs. A position-wise feed-forward network applies the same nonlinear transformation at every position. Residual connections and normalization stabilize deep training. Across layers, a token can compose multi-step contextual relationships.
<!-- bilingual-en:end -->

attention 在 token 之间混合信息，前馈层在每个 token 内变换通道；残差路径保留旧表示并为梯度提供短路径。缺少其中任何一项，都不能仅凭公式中的 attention 矩阵复现完整 Transformer block。
<!-- bilingual-en:start -->
Attention mixes information across tokens, while the feed-forward network transforms channels within each token. Residual paths retain prior representations and provide short routes for gradients. The attention matrix alone is therefore insufficient to reproduce a complete Transformer block.
<!-- bilingual-en:end -->

## 位置与序列顺序
<!-- bilingual-en:start -->
*Position and Sequence Order*
<!-- bilingual-en:end -->

原始 Transformer 加正弦位置编码；后续模型常用学习位置、相对位置或旋转位置编码。位置方案影响长度外推，但“换一种位置编码”不能单独保证模型在训练窗外仍能正确利用远距离信息。
<!-- bilingual-en:start -->
The original Transformer adds sinusoidal positional encodings; later models often use learned, relative, or rotary position schemes. Position design affects length extrapolation, but changing the positional encoding alone does not guarantee correct use of distant information beyond the training window.
<!-- bilingual-en:end -->

一个诊断方法是比较“上下文能否装进去”和“模型是否真正使用它”。位置机制与上下文上限主要影响前者；后者还受训练长度分布、attention 模式、检索干扰和评测位置影响，详见 [[长上下文语言模型]]。
<!-- bilingual-en:start -->
A useful diagnostic separates whether context fits from whether the model actually uses it. Position mechanisms and the context limit mainly affect the former; the latter also depends on training-length distribution, attention patterns, retrieval interference, and evaluation position. See [[长上下文语言模型|Long-Context Language Models]].
<!-- bilingual-en:end -->

## Encoder、decoder 与因果掩码
<!-- bilingual-en:start -->
*Encoders, Decoders, and Causal Masks*
<!-- bilingual-en:end -->

encoder 可双向读取整个输入，适合理解或编码；decoder 用 causal mask，只能读取当前位置及过去，适合自回归生成；encoder–decoder 还用 cross-attention 从输入读取。BERT 类模型偏 encoder，GPT 类模型偏 decoder，T5 类模型使用 encoder–decoder。
<!-- bilingual-en:start -->
An encoder reads the full input bidirectionally and suits representation or understanding tasks. A decoder uses a causal mask and reads only the current and previous positions, supporting autoregressive generation. An encoder–decoder additionally uses cross-attention to read the encoded input. BERT-like models are encoder-oriented, GPT-like models decoder-oriented, and T5-like models use both.
<!-- bilingual-en:end -->

选择架构应从信息可见性出发：分类允许整段输入相互读取；续写必须避免未来泄漏；条件生成既需要完整读取输入，又要让输出保持自回归。架构名称只是这些约束的常见实现。
<!-- bilingual-en:start -->
Architecture choice should begin with information visibility. Classification permits the full input to interact; continuation must prevent future leakage; conditional generation needs full access to the input while keeping the output autoregressive. Architecture labels are common implementations of these constraints.
<!-- bilingual-en:end -->

## Next-token prediction
<!-- bilingual-en:start -->
*Next-Token Prediction*
<!-- bilingual-en:end -->

decoder 语言模型最大化 $\sum_t\log p(x_t\mid x_{<t})$。训练可并行计算所有位置的损失，生成却通常逐 token 进行。这个目标学习语言、知识与模式，但不直接保证事实性、服从指令或安全性，后续对齐阶段解决的是另一层问题。
<!-- bilingual-en:start -->
A decoder language model maximizes $\sum_t\log p(x_t\mid x_{<t})$. Training computes losses for all positions in parallel, whereas generation usually proceeds token by token. This objective learns language, knowledge, and patterns, but it does not directly guarantee factuality, instruction following, or safety; later alignment stages address a different layer of behavior.
<!-- bilingual-en:end -->

这里保留训练目标是为了闭合架构解释；数据、预算和训练动态属于 [[LLM 预训练]]，SFT 与偏好学习属于 [[LLM 后训练：SFT、RLHF 与 DPO]]。
<!-- bilingual-en:start -->
The objective appears here only to complete the architectural explanation. Data, budgets, and training dynamics belong in [[LLM 预训练|LLM Pretraining]], while supervised fine-tuning and preference learning belong in [[LLM 后训练：SFT、RLHF 与 DPO|LLM Post-Training: SFT, RLHF, and DPO]].
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 把 attention 权重直接当解释：检查结论是否跨层、跨头、跨扰动稳定，并用干预或归因方法补充，而不是只看一张热力图。
  <!-- bilingual-en:start -->
  Attention weights are treated as a complete explanation: test stability across layers, heads, and perturbations, and supplement visualization with interventions or attribution methods.
  <!-- bilingual-en:end -->
- 张量维度对不上：逐项写出 $Q,K,V$ 的形状，确认 $QK^\top$ 在序列维形成 $n\times n$ 分数矩阵，并检查 head 的拆分与拼接。
  <!-- bilingual-en:start -->
  Tensor dimensions do not match: write the shapes of $Q$, $K$, and $V$ explicitly, verify that $QK^\top$ forms an $n\times n$ score matrix over sequence positions, and inspect head splitting and concatenation.
  <!-- bilingual-en:end -->
- 训练损失异常低：检查 causal mask、label shift 与 padding mask，排除目标 token 或未来 token 泄漏。
  <!-- bilingual-en:start -->
  Training loss is suspiciously low: inspect the causal mask, label shift, and padding mask to rule out leakage of target or future tokens.
  <!-- bilingual-en:end -->
- 上下文变长但效果不升：区分位置上限、训练长度和远距离信息利用，不要把“接受更多 token”误写成“理解更多 token”。
  <!-- bilingual-en:start -->
  A larger context window does not improve results: distinguish positional capacity, training length, and actual distant-information use; accepting more tokens is not the same as understanding more tokens.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 用自己的话解释 query、key、value 各自做什么。
<!-- bilingual-en:start -->
*Explain in your own words what queries, keys, and values do.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> query 表示当前位置想找什么，key 表示每个位置可怎样被匹配，value 是匹配后实际被汇总的信息。
<!-- bilingual-en:start -->
> [!answer]- Answer
> A query represents what the current position seeks, a key represents how each position can be matched, and a value contains the information aggregated after matching.
<!-- bilingual-en:end -->

### 为什么 decoder 训练能并行而生成仍常逐 token？
<!-- bilingual-en:start -->
*Why can decoder training be parallel even though generation is usually token by token?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 训练时完整目标序列已知，可用 causal mask 同时算所有位置；生成时下一个输入依赖刚生成的 token。
<!-- bilingual-en:start -->
> [!answer]- Answer
> During training the complete target sequence is known, so a causal mask permits all positions to be computed together. During generation, the next input depends on the token just generated.
<!-- bilingual-en:end -->

### 只增加 attention 层为什么还不构成语言模型？
<!-- bilingual-en:start -->
*Why does adding attention alone not create a language model?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 还需要 token/位置表示、前馈与残差结构、输出分布和训练目标；attention 只是信息路由机制。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Token and position representations, feed-forward and residual structure, an output distribution, and a training objective are also required; attention is only the information-routing mechanism.
<!-- bilingual-en:end -->

### causal mask 应该在 softmax 前还是后应用，为什么？
<!-- bilingual-en:start -->
*Should a causal mask be applied before or after softmax, and why?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 在 softmax 前把未来位置分数设为 $-\infty$，归一化后其概率才严格为零；若之后再乘零，剩余权重不再保持正确归一化。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Future scores should become $-\infty$ before softmax so their normalized probabilities are exactly zero. Multiplying by zero afterward leaves the remaining weights incorrectly normalized.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
  <!-- bilingual-en:start -->
  [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
  <!-- bilingual-en:end -->
- 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
  <!-- bilingual-en:start -->
  The corresponding original papers in Zotero verify the architecture, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
  <!-- bilingual-en:end -->
- Vaswani et al. (2017), [[06_paper/LLM/papers/vaswaniAttentionAllYou|Attention Is All You Need]]：核验 Transformer 与多头注意力。
  <!-- bilingual-en:start -->
  Vaswani et al. (2017), [[06_paper/LLM/papers/vaswaniAttentionAllYou|Attention Is All You Need]], verifies the Transformer architecture and multi-head attention.
  <!-- bilingual-en:end -->
- [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762)：对照原论文核验 scaled dot-product attention、多头结构、位置编码、残差与 encoder–decoder 架构。
  <!-- bilingual-en:start -->
  [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762) is the primary source used to verify scaled dot-product attention, multi-head structure, positional encoding, residual pathways, and the encoder–decoder architecture.
  <!-- bilingual-en:end -->
