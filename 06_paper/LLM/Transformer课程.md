---
aliases:
  - Transformer：从输入表示到输出概率
  - Transformer 与注意力机制
  - Transformer 架构与注意力机制
  - Transformer Architecture and Attention
---

# Transformer：从输入表示到输出概率

<!-- bilingual-en:start -->
*Transformer: from input representations to output probabilities*
<!-- bilingual-en:end -->

读 Transformer 时，最容易走丢的地方不是某个公式太难，而是不清楚这个公式的输入从哪里来、输出又交给谁。本章沿着一次计算向前走：先把 token 变成向量，再看位置之间如何取信息、每个位置如何更新，最后把表示变成词表上的概率。你可以连续读完，也可以打开其中的共享原子，单独复习一个定义、推导或判断。
<!-- bilingual-en:start -->
The difficult part of reading a Transformer is often not an individual formula, but knowing where its inputs come from and where its output goes. This chapter follows the computation: token IDs become vectors, positions exchange information, each position is updated, and the resulting representations become probabilities over a vocabulary. Read straight through or open a shared atom to revisit one definition, derivation, or judgement.
<!-- bilingual-en:end -->

全局关系见 [[Transformer.canvas|Transformer 主题图]]。

## 1. 先看输入、计算和输出

<!-- bilingual-en:start -->
*1. Locate the input, computation, and output*
<!-- bilingual-en:end -->

先以原始翻译模型为例：源句经过编码器，得到一串可供查阅的表示；解码器根据译文前缀查阅这些表示，再输出下一个 token 的分布。注意力是这个过程中交换信息的部件，而 Transformer 是把这些部件组织起来的架构。
<!-- bilingual-en:start -->
Start with the original translation model. An encoder turns the source sentence into a sequence of representations. A decoder consults them using the target prefix, and an output head produces a distribution for the next token. Attention exchanges information inside this process; the Transformer is the architecture that organizes these components.
<!-- bilingual-en:end -->

![[Transformer]]

接下来把镜头移到一串输入。省略 batch 维后，记序列长度为 $n$、每个位置的表示宽度为 $d_{\mathrm{model}}$。矩阵的每一行对应一个位置；这些行最初是怎样得到的？
<!-- bilingual-en:start -->
Now focus on one input sequence. Omitting the batch dimension, let $n$ be its length and $d_{\mathrm{model}}$ the representation width. Each matrix row corresponds to a position. How are those rows first obtained?
<!-- bilingual-en:end -->

![[Token嵌入]]

重复出现的 token 最初可以查到同一个向量。它在当前句子里怎样与其他位置发生联系，则由后面的计算决定。先暂时把位置处理放在一边，弄清一次读取本身。
<!-- bilingual-en:start -->
Repeated token IDs can initially retrieve the same vector. Their relationships with other positions in this sentence are built by later computation. Set positional processing aside for a moment and examine one attention read.
<!-- bilingual-en:end -->

## 2. 手算一次读取

<!-- bilingual-en:start -->
*2. Work through one attention read*
<!-- bilingual-en:end -->

一次读取需要区分两个问题：怎样决定多读哪个位置，以及从那个位置实际读到什么。这就是 query、key 和 value 分工的起点。
<!-- bilingual-en:start -->
An attention read separates two questions: how much to read from each position, and what content to retrieve there. This is the starting point for queries, keys, and values.
<!-- bilingual-en:end -->

![[注意力读取]]

匹配分数还不是读取权重。为了比较同一个 query 面前的候选位置，需要把这一组分数一起归一化。这里的“同一组”很重要：每个 query 有自己的一行权重。
<!-- bilingual-en:start -->
Compatibility scores are not yet reading weights. The candidates facing one query must be normalized together. The grouping matters: each query has its own row of weights.
<!-- bilingual-en:end -->

![[Softmax]]

现在把三个步骤接起来：查询与键做点积、沿 key 轴归一化、用所得权重汇总 values。下面的例子保留实际向量和数值，你可以停下来手算一遍。核对时看清 $n_q$ 与 $n_k$：读取方和被读取方不必一样长。
<!-- bilingual-en:start -->
Join the three steps: query–key dot products, normalization along the key axis, and a weighted sum of values. The following example keeps the actual vectors and numbers so you can calculate it yourself. Track $n_q$ and $n_k$: the reading sequence and the sequence being read need not have the same length.
<!-- bilingual-en:end -->

![[缩放点积注意力]]

你已经能算出输出，但公式中还有一个需要解释的选择：为什么要除以 $\sqrt{d_k}$？这个理由依赖一组统计假设，不能把推导中的方差直接当成训练后模型的实测方差。
<!-- bilingual-en:start -->
You can now calculate the output, but one choice remains to be explained: division by $\sqrt{d_k}$. Its variance argument depends on statistical assumptions; the variance in that derivation is not automatically the measured variance of a trained model.
<!-- bilingual-en:end -->

![[点积缩放]]

## 3. 同一个算子，信息从哪里来

<!-- bilingual-en:start -->
*3. The same operator, different information sources*
<!-- bilingual-en:end -->

刚才从已经给定的 $Q,K,V$ 开始。回到网络中，它们通常由表示经过可学习投影得到。先看三者都取自同一表示序列的情况。
<!-- bilingual-en:start -->
The calculation above started with given $Q,K,V$. In a network, learned projections usually produce them from representations. Begin with the case where all three originate in the same sequence.
<!-- bilingual-en:end -->

![[自注意力]]

“self” 说的是来源，不是每个位置只看自己。若译文中的位置要查阅源句，把 query 的来源留在译文侧，把 key 和 value 的来源换成编码器输出，便得到另一种信息连接。
<!-- bilingual-en:start -->
“Self” describes the source, not a restriction to reading oneself. When a target position consults the source sentence, queries remain on the target side while keys and values come from encoder outputs. This gives a different connection between information sources.
<!-- bilingual-en:end -->

![[交叉注意力]]

一次投影给出一套匹配和读取方式。多头层允许多套这样的投影并行工作，再把读出的通道拼接、重新组合。它可以用于自注意力，也可以用于交叉注意力。
<!-- bilingual-en:start -->
One set of projections defines one way to match and read. A multi-head layer runs several such sets in parallel, then concatenates and recombines their output channels. It can be used for either self-attention or cross-attention.
<!-- bilingual-en:end -->

![[多头注意力]]

## 4. 从读取结果到下一层表示

<!-- bilingual-en:start -->
*4. From an attention read to the next layer's representations*
<!-- bilingual-en:end -->

注意力把其他位置的信息写入当前行。接下来，每一行还要经过非线性变换。这个变换的参数在同一层的各位置共享，但它本身不再读取另一行。
<!-- bilingual-en:start -->
Attention writes information from other positions into the current row. Each row then undergoes a nonlinear transformation. Positions share its parameters within a layer, but the transformation itself does not read another row.
<!-- bilingual-en:end -->

![[逐位置前馈网络]]

新计算的结果如何与进入子层前的表示结合？先看相加这一步，再看归一化；把两者分开，才能读准原图里 Add & Norm 的顺序。
<!-- bilingual-en:start -->
How does the new computation combine with the representation entering the sublayer? Examine addition first, then normalization. Separating them makes the order of Add & Norm in the original diagram clear.
<!-- bilingual-en:end -->

![[残差连接]]

![[LayerNorm]]

现在把两步重新放回原始 Transformer。这里讨论的是那篇论文的具体子层布局，读其他模型时应重新核对归一化的位置。
<!-- bilingual-en:start -->
Put the two steps back into the original Transformer. This is the specific sublayer arrangement in that paper; when reading another model, check the placement of normalization again.
<!-- bilingual-en:end -->

![[Transformer后归一化]]

## 5. 顺序信息在哪里进入

<!-- bilingual-en:start -->
*5. Where sequence order enters the computation*
<!-- bilingual-en:end -->

到这里，我们会对一组向量进行读取和变换了。但如果输入只有 token 身份，没有位置或顺序约束，把几行一起换个排列，会发生什么？下面的命题把这个问题说准确。
<!-- bilingual-en:start -->
We can now read and transform a collection of vectors. But if the input contains only token identities, without positional information or order constraints, what happens when its rows are permuted? The following proposition makes that question precise.
<!-- bilingual-en:end -->

![[自注意力置换等变性]]

因此，理解位置方案时，先问它把位置或距离引入了哪个计算环节。原始模型把向量相加，后续方法也可以修改注意力分数或旋转投影后的向量。
<!-- bilingual-en:start -->
To understand a positional scheme, first ask where it introduces position or distance into the computation. The original model adds vectors; later approaches may modify attention scores or rotate projected vectors.
<!-- bilingual-en:end -->

![[位置编码]]

先比较两种给绝对位置一个向量的办法：一种由固定公式生成，另一种由训练学得的表给出。它们和 token 嵌入的查表不是同一组参数。
<!-- bilingual-en:start -->
Compare two ways to assign a vector to an absolute position: a fixed formula and a learned table. Their positional representations are distinct from the token-embedding parameters.
<!-- bilingual-en:end -->

![[正弦位置编码]]

![[可学习位置嵌入]]

RoPE 的位置则不同：先获得 query 和 key 的投影，再按位置旋转。读下面的推导时，注意它改用单个 token 的列向量记号；内积里的相对位置来自两个旋转矩阵的乘积。
<!-- bilingual-en:start -->
RoPE enters elsewhere: queries and keys are projected first, then rotated by position. The derivation below explicitly switches to column vectors for individual tokens. Relative position in the inner product comes from multiplying the two rotation matrices.
<!-- bilingual-en:end -->

![[RoPE]]

## 6. 编码器和解码器怎样组织读取

<!-- bilingual-en:start -->
*6. How encoders and decoders organize information access*
<!-- bilingual-en:end -->

位置机制说明位置关系，可见性约束说明哪些位置允许被读取。先把最常见的因果掩码写清，再看它怎样进入完整架构。
<!-- bilingual-en:start -->
Positional mechanisms represent positional relationships; visibility constraints specify which positions may be read. Define the usual causal mask first, then place it in the architecture.
<!-- bilingual-en:end -->

![[因果注意力掩码]]

屏蔽未来位置还涉及一个运算细节：权重必须在允许的候选集合上归一化。只把某些已经算出的权重乘零，会改变剩余权重之和。
<!-- bilingual-en:start -->
Blocking future positions has a computational detail: normalization must use the allowed candidate set. Simply zeroing selected weights after they have been calculated changes the sum of the remaining weights.
<!-- bilingual-en:end -->

![[注意力掩码归一化]]

现在回到原始翻译模型。编码器先处理源句，保留每个位置的上下文表示；解码器再使用目标前缀和这些源表示。两边的子层顺序与信息来源都已在前面准备好。
<!-- bilingual-en:start -->
Return to the original translation model. The encoder processes the source sentence and retains contextual representations at every position. The decoder then uses the target prefix together with those source representations. The sublayers and information sources needed to follow both modules are now in place.
<!-- bilingual-en:end -->

![[Transformer编码器]]

![[Transformer解码器]]

读 BERT、GPT 或 T5 时，不应只凭“编码”或“解码”两个词猜测用途。把输入分成来源序列和待预测序列，再画出允许读取的关系，就能比较这些常见结构。
<!-- bilingual-en:start -->
When reading BERT, GPT, or T5, do not infer the task from the words “encoder” and “decoder” alone. Identify the source sequence and the sequence to be predicted, then compare which information each position may access.
<!-- bilingual-en:end -->

![[Transformer架构可见性]]

## 7. 隐藏状态怎样变成预测，预测怎样进入训练

<!-- bilingual-en:start -->
*7. From hidden states to predictions and training*
<!-- bilingual-en:end -->

经过若干层后，仍然得到一串向量。要给某个 token 一个概率，需要另外的输出接口。注意这里 Softmax 的归一化轴变成了词表，而不是输入位置。
<!-- bilingual-en:start -->
After several layers, the result is still a sequence of vectors. Assigning a probability to a token requires an output interface. Here Softmax normalizes over the vocabulary, not over input positions.
<!-- bilingual-en:end -->

![[语言模型输出头]]

接下来的三个共享原子以 decoder-only 语言模型为主要语境。第一个回答训练在优化什么，第二个回答训练时的前缀从哪里来，第三个回答并行计算时怎样避免目标泄漏。它们分别约束概率目标、输入协议和实际信息流。
<!-- bilingual-en:start -->
The next three shared atoms primarily concern decoder-only language models. The first defines the training objective, the second identifies where training prefixes come from, and the third explains how parallel computation avoids target leakage. They address the probability objective, input protocol, and actual information flow, respectively.
<!-- bilingual-en:end -->

![[自回归目标]]

![[Teacher forcing]]

![[Transformer自回归约束]]

这就接上了训练与生成的区别：训练时可以利用已知真实序列组织并行计算，自由生成时后续输入依赖已经生成的 token。继续研究训练数据与预算，可进入 [[LLM 预训练.canvas|预训练主题图]]；真实前缀与模型前缀的差异见 [[暴露偏差]]。至于示范回答、偏好等信号怎样改变训练目标，另见 [[训练信号差异]]。
<!-- bilingual-en:start -->
This connects training to generation: known target sequences permit parallel training computation, whereas later inputs in free generation depend on tokens already generated. Continue to the [[LLM 预训练.canvas|pretraining map]] for data and budgets, and to [[暴露偏差|exposure bias]] for differences between observed and model-generated prefixes. [[训练信号差异|Training-signal distinctions]] explains how demonstrations and preferences introduce other training signals.
<!-- bilingual-en:end -->

## 8. 从计算机制判断成本与解释范围

<!-- bilingual-en:start -->
*8. Use the computation to judge cost and explanatory scope*
<!-- bilingual-en:end -->

最后回到论文里常见的两个判断：长序列为什么贵，以及注意力热图能说明什么。前者先数成对运算，再区分怎样保存中间结果。
<!-- bilingual-en:start -->
Finish with two recurring questions in papers: why long sequences are expensive, and what an attention heatmap can tell us. For cost, count pairwise computation first, then distinguish how intermediate results are stored.
<!-- bilingual-en:end -->

![[稠密注意力二次成本]]

![[注意力计算与显存]]

理解这两条后，再进入 [[预填充|prefill]] 与[[增量解码|逐步解码]] 或 [[长上下文语言模型|长上下文的有效利用]]。一次能装入更多 token、一次需要多少运算、模型实际能否利用其中信息，是需要分别检查的事。
<!-- bilingual-en:start -->
With those distinctions in place, continue to [[预填充|prefill]] and [[增量解码|incremental decoding]] or [[长上下文语言模型|effective use of long contexts]]. How many tokens fit, how much computation they require, and whether the model uses their information are separate questions.
<!-- bilingual-en:end -->

热图则只展示了部分计算。回想最初的读取公式：权重相同，并不意味着被加权的 value 相同；一层写入的向量也还会进入后续网络。
<!-- bilingual-en:start -->
A heatmap displays only part of the computation. Recall the initial attention formula: identical weights do not imply identical values, and the vector written by one layer still passes through the rest of the network.
<!-- bilingual-en:end -->

![[注意力权重解释边界]]

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

本章的公式、条件、推导与最小例子由上方共享原子的来源节逐项承载。Vaswani 原文用于原始架构；残差、LayerNorm、位置变体、存储算法和注意力解释分别回到相应原论文。课程中的过渡段负责连接这些明确的计算接口，未执行模型训练或性能复现。
<!-- bilingual-en:start -->
The source sections of the embedded atoms support their formulas, conditions, derivations, and minimal examples. The original architecture is traced to Vaswani and colleagues; residual connections, LayerNorm, positional variants, memory-efficient algorithms, and interpretation are checked against their respective primary papers. The chapter's transitions connect these explicit computational interfaces. No model training or performance reproduction was performed.
<!-- bilingual-en:end -->

论文与原始附件入口：[[06_paper/LLM/papers/vaswaniAttentionAllYou|Attention Is All You Need]] · [[06_paper/LLM/papers/devlin2019BERTPretrainingDeep|BERT]] · [[06_paper/LLM/papers/raffel2023ExploringLimitsTransfer|T5]] · [[06_paper/LLM/papers/su2023RoFormerEnhancedTransformer|RoFormer]]。完整来源路线见 [[06_paper/LLM/LLM Map Index|LLM 论文索引]]。
