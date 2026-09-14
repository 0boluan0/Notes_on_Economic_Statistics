---
aliases:
  - 常见 Transformer 架构通过双向自读、因果自读及跨序列读取区分信息流
  - Common Transformer architectures differ in bidirectional self-attention, causal self-attention, and cross-sequence access
  - Transformer architecture visibility
student_os: knowledge-atom
atom_id: LLM-TF-026
atom_type: distinction
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# 常见 Transformer 架构通过双向自读、因果自读及跨序列读取区分信息流
<!-- bilingual-en:start -->
*Common Transformer architectures differ in bidirectional self-attention, causal self-attention, and cross-sequence access*
<!-- bilingual-en:end -->

比较常见的 encoder-only、decoder-only 与 encoder–decoder [[Transformer]] 时，要同时检查**有几套序列表示，以及每个 query 可以读哪些 key/value**。下面以 BERT 型双向编码器、常见因果语言模型和原始序列到序列 Transformer 为参照。

<!-- bilingual-en:start -->
When comparing common encoder-only, decoder-only, and encoder–decoder [[Transformer|Transformers]], inspect both **which sequence representations exist and which keys/values each query can read**. The comparison below uses a BERT-style bidirectional encoder, a conventional causal language model, and the original sequence-to-sequence Transformer.
<!-- bilingual-en:end -->

| 架构 | 同序列自注意力可见性 | 跨序列读取 |
|---|---|---|
| Encoder-only | [[Transformer编码器]] 内每个位置可读当前输入的全部有效位置 | 没有独立目标 decoder 的读取路径 |
| Decoder-only | 单栈采用 [[因果注意力掩码]]，每个输入位置可读自身与左侧输入 | 常见因果 LM 没有独立 encoder 的 cross-attention 路径；提示与续写进入同一序列 |
| Encoder–decoder | 源序列 encoder 双向；[[Transformer解码器]] 的目标自注意力采用因果可见性 | 目标侧经 [[交叉注意力]] 读取全部有效源表示 |

<!-- bilingual-en:start -->
| Architecture | Self-attention visibility | Cross-sequence access |
|---|---|---|
| Encoder-only | Each position in the [[Transformer编码器\|encoder]] can read all valid positions in the supplied input | No separate target-decoder read path |
| Decoder-only | The single stack uses a [[因果注意力掩码\|causal attention mask]]; each input position can read itself and earlier inputs | A conventional causal LM has no cross-attention path to a separate encoder; prompt and continuation form one sequence |
| Encoder–decoder | Bidirectional source encoder; causal target self-attention in the [[Transformer解码器\|decoder]] | Target queries use [[交叉注意力\|cross-attention]] to read all valid source representations |
<!-- bilingual-en:end -->

例如源序列是 $(s_1,s_2)$，目标侧当前输入是 $(\mathrm{BOS},y_1)$。在 encoder–decoder 中，处理 $y_1$ 的 query 可以在目标自注意力中读取 $\mathrm{BOS},y_1$，再通过 cross-attention 读取 $s_1,s_2$ 的编码表示；与该位置对齐的下一个目标是 $y_2$，而非 $y_1$。原始 Transformer 通过目标错位与目标侧因果掩码保持这一预测合法；同类约束在 decoder-only 中的实现见 [[Transformer自回归约束]]。

<!-- bilingual-en:start -->
Suppose the source sequence is $(s_1,s_2)$ and the current target inputs are $(\mathrm{BOS},y_1)$. In an encoder–decoder model, the query processing $y_1$ can read $\mathrm{BOS},y_1$ through target self-attention, then read the encoded source through cross-attention. Its next target is $y_2$. The original Transformer combines target shifting with target-side causal masking to make this prediction valid. The corresponding decoder-only implementation is described in [[Transformer自回归约束|Transformer autoregressive constraints]].
<!-- bilingual-en:end -->

“双向”是读取**当前送入模型的输入**，并不保证看到被预训练任务移除的真实答案。“Decoder-only”也不足以单独确定每一块 mask：prefix LM 可以让同一栈中的提示前缀内部双向可见，再让续写部分因果可见。遇到新模型，应查看其实际注意力可见性规则和训练目标。

<!-- bilingual-en:start -->
“Bidirectional” means access to the **input actually supplied to the model**, not to original answers removed by a pretraining task. Nor does “decoder-only” uniquely determine every mask: a prefix LM can use bidirectional attention within the prompt prefix and causal attention for continuation in the same stack. For an unfamiliar model, inspect its actual attention visibility rules and training objective.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/html/1706.03762v7)，§3.1、§3.2.3：支持原始 encoder–decoder 的两栈结构、目标因果自注意力与源—目标交叉注意力。
- [Devlin et al. (2019), *BERT*](https://aclanthology.org/N19-1423.pdf)，§3 “Model Architecture”：支持 BERT 的双向 Transformer encoder。
- [Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://arxiv.org/html/1910.10683v4#S3.SS2.SSS1)，§3.2.1、Fig. 3–4：支持 full、causal、prefix 的可见性差异及单栈/两栈比较。源与目标的短例子按这些规则构造。

<!-- bilingual-en:start -->
- Vaswani et al., §3.1 and §3.2.3, defines the original encoder–decoder information paths.
- Devlin et al., §3 “Model Architecture,” specifies BERT's bidirectional Transformer encoder.
- Raffel et al., §3.2.1 and Figures 3–4, compares fully visible, causal, and prefix masks across one- and two-stack architectures. The short source–target example applies those rules directly.
<!-- bilingual-en:end -->
