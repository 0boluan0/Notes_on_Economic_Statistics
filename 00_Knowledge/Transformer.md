---
aliases:
  - "Transformer 是以堆叠注意力和逐位置前馈网络构造序列表示的神经网络架构"
  - Transformer architecture
student_os: knowledge-atom
atom_id: LLM-TF-013
atom_type: definition
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# Transformer 是以堆叠注意力和逐位置前馈网络构造序列表示的神经网络架构

<!-- bilingual-en:start -->
*The Transformer is a neural network architecture that builds sequence representations through stacked attention and position-wise feed-forward networks.*
<!-- bilingual-en:end -->

Transformer 是一种以注意力组织序列位置之间的信息交互、以逐位置前馈网络变换各位置表示的神经网络架构。它以 token 表示为输入，在处理序列时注入位置信息，经带有残差连接和归一化的多层模块逐步更新，再把最后表示送入任务所需的输出接口。原始 Transformer 将这些模块组织成编码器与解码器，完成输入序列到输出序列的条件生成。

<!-- bilingual-en:start -->
The Transformer is a neural network architecture that uses attention to exchange information between sequence positions and position-wise feed-forward networks to transform each position's representation. It takes token representations as input, introduces positional information during sequence processing, updates the representations through stacked modules with residual connections and normalization, and passes the final representations to a task-specific output interface. The original Transformer organizes these modules into an encoder and decoder for conditional generation from an input sequence to an output sequence.
<!-- bilingual-en:end -->

## 从输入表示到输出分布

<!-- bilingual-en:start -->
*From input representations to an output distribution.*
<!-- bilingual-en:end -->

原始架构先由 [[Token嵌入]] 和 [[位置编码]] 形成源端表示 $X_{\mathrm{src}}$ 与目标端表示 $X_{\mathrm{tgt}}$。后者由已经提供给解码器的目标前缀构成；训练时可以把各位置所需的前缀放在同一个带因果掩码的张量中。把完整数据流写成模块接口，就是

<!-- bilingual-en:start -->
The original architecture combines [[Token嵌入|token embeddings]] with [[位置编码|positional encoding]] to form source representations $X_{\mathrm{src}}$ and target representations $X_{\mathrm{tgt}}$. The target input contains the prefix supplied to the decoder. During training, one causally masked tensor can represent the prefixes needed at all positions. At the module level, the data flow is
<!-- bilingual-en:end -->

$$
Z=\operatorname{Encoder}(X_{\mathrm{src}}),\qquad
H=\operatorname{Decoder}(X_{\mathrm{tgt}},Z),\qquad
P=\operatorname{Head}(H).
$$

[[Transformer编码器]] 把输入变成上下文表示序列 $Z$；[[Transformer解码器]] 同时使用目标前缀与 $Z$ 构造隐藏状态 $H$；[[语言模型输出头]] 将这些状态映射到词表分布 $P$。例如，假想源端有“我／读书”两个 token，编码器产生两个向量；目标端已有一个译文前缀，解码器利用前缀和这两个源向量构造下一个 token 的分布。这个例子只说明接口间传递的对象。

<!-- bilingual-en:start -->
The [[Transformer编码器|Transformer encoder]] produces a sequence of contextual representations $Z$. The [[Transformer解码器|Transformer decoder]] uses both the target prefix and $Z$ to construct hidden states $H$, and the [[语言模型输出头|language-model output head]] converts these states into vocabulary distributions $P$. For a toy source tokenized into two tokens meaning “I” and “read books,” the encoder produces two vectors. Given a translated target prefix, the decoder uses that prefix and the source vectors to form a distribution for the next token. This example illustrates the objects passed between interfaces.
<!-- bilingual-en:end -->

## 堆叠层里的分工

<!-- bilingual-en:start -->
*How components divide the work within stacked layers.*
<!-- bilingual-en:end -->

每层以 [[多头注意力]] 读取可见位置的信息，再以 [[逐位置前馈网络]] 处理各位置的通道。[[残差连接]] 把子层变换与已有表示相加；原始架构还按 [[Transformer后归一化]] 的顺序施加 LayerNorm。重复堆叠让后层使用前层已经形成的上下文表示。

<!-- bilingual-en:start -->
Each layer uses [[多头注意力|multi-head attention]] to read from visible positions and a [[逐位置前馈网络|position-wise feed-forward network]] to transform each position's channels. [[残差连接|Residual connections]] add sublayer transformations to existing representations; the original architecture also applies LayerNorm in the order specified by [[Transformer后归一化|original Transformer post-normalization]]. Stacking lets later layers operate on contextual representations built by earlier ones.
<!-- bilingual-en:end -->

Transformer 后来形成 encoder-only、decoder-only 等组织方式。阅读具体模型时，沿 [[Transformer架构可见性]] 确认哪些位置能互相读取，再对应其模块结构和训练目标。

<!-- bilingual-en:start -->
Later Transformer variants include encoder-only and decoder-only organizations. When reading a particular model, use [[Transformer架构可见性|Transformer architecture visibility]] to identify which positions can read one another, then relate that pattern to its modules and training objective.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification.*
<!-- bilingual-en:end -->

- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/pdf/1706.03762)，§3、Figure 1、§3.1–3.5：支持原始架构的输入表示、编码器与解码器接口、子层组成和输出分布。接口公式与词例是按原图整理的教学表达。
  <!-- bilingual-en:start -->
  Section 3, Figure 1, and Sections 3.1–3.5 establish the original input representations, encoder and decoder interfaces, sublayer components, and output distribution. The interface notation and toy sequence are teaching constructions based on that diagram.
  <!-- bilingual-en:end -->
- [Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)，§3.1、式 (2)：支持 decoder-only Transformer 是这一架构的具体变体，详细可见性由链接卡承载。
  <!-- bilingual-en:start -->
  Section 3.1 and Equation (2) establish a decoder-only Transformer variant; the linked visibility note specifies the detailed architectural distinction.
  <!-- bilingual-en:end -->
- [Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://aclanthology.org/N19-1423.pdf)，§3 的 Model Architecture：支持基于多层双向 Transformer 编码器的变体；可见性比较由链接卡展开。
  <!-- bilingual-en:start -->
  Model Architecture in Section 3 establishes a variant based on a multilayer bidirectional Transformer encoder. The linked visibility note develops the comparison.
  <!-- bilingual-en:end -->
