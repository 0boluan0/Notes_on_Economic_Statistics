---
aliases:
  - "Token 数是预处理文本经指定 tokenizer 编码后的序列长度"
  - "A token count is the encoded sequence length produced by a specified tokenizer over preprocessed text"
student_os: knowledge-atom
atom_id: LLM-PRE-008
atom_set: llm-pretraining
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[预训练分布]]"
  - "[[数据混合]]"
  - "[[混合权重与Token暴露]]"
  - "[[计算最优分配]]"
leads_to:
  - "[[困惑度可比性]]"
part_of:
  - "[[LLM 预训练.canvas|LLM 预训练]]"
---

# Token 数是预处理文本经指定 tokenizer 编码后的序列长度
<!-- bilingual-en:start -->
*A token count is the encoded sequence length produced by a specified tokenizer over preprocessed text*
<!-- bilingual-en:end -->

> [!summary] 定义
> 对原始文档集合 $D$、预处理函数 $P$ 与 tokenizer $\tau$，token 数定义为
> $$
> N_{\tau,P}(D)=\sum_{d\in D}\left|\tau(P(d))\right|.
> $$
> token 不是原始文本自带的单位；它是文本经过明确管线后得到的离散序列元素。只给一个 token 总数而不指定 $P$ 与 $\tau$，不能复现计数对象。
>
> <!-- bilingual-en:start -->
> For raw documents $D$, preprocessing function $P$, and tokenizer $\tau$, the token count is $N_{\tau,P}(D)=\sum_{d\in D}|\tau(P(d))|$. Tokens are elements of the sequence produced by a declared pipeline, not units already attached to raw text. A total without $P$ and $\tau$ does not identify a reproducible counting object.
> <!-- bilingual-en:end -->

## 预处理与 tokenizer 各自决定什么
<!-- bilingual-en:start -->
*What preprocessing and the tokenizer determine*
<!-- bilingual-en:end -->

$P$ 决定哪一段文本进入编码，例如 HTML 抽取、Unicode 与空白规范化、文档截断、边界符插入和重复处理；$\tau$ 决定这段文本怎样切成词、子词、字节或特殊符号。两者任一改变，token 序列及其长度都可能改变。

<!-- bilingual-en:start -->
Preprocessing determines what text is encoded, including extraction, normalization, truncation, boundaries, and repetition. The tokenizer determines how that text is segmented into words, subwords, bytes, or special symbols. Changing either can change the sequence and its length.
<!-- bilingual-en:end -->

例如，假想 tokenizer A 把 `unbelievable` 编码为 `un`、`believ`、`able` 三个 token，tokenizer B 把它编码为一个 token。同一可见字符串的计数分别为 3 与 1；这说明编码粒度不同，不说明 A 获得三倍语义信息。

<!-- bilingual-en:start -->
For example, tokenizer A may encode `unbelievable` as three tokens while tokenizer B encodes it as one. The counts show different encoding granularity, not three times as much semantic information.
<!-- bilingual-en:end -->

## Token 计数参与哪些量
<!-- bilingual-en:start -->
*Quantities that use token counts*
<!-- bilingual-en:end -->

- 在 [[混合权重与Token暴露]] 中，来源权重只有结合抽样单位与加载规则才能换算为期望 token 暴露。
- 在 [[计算最优分配]] 中，训练 token 数是预算分配变量。
- 在 [[困惑度]] 中，有效 token 数是平均负对数似然的分母。

这些量都可以在各自明确的管线内使用 token 作为单位。定义成立不等于不同 tokenizer 已经拥有共同口径；需要跨模型比较 raw PPL 或换用 bits per byte（BPB）时，进入 [[困惑度可比性]]。

<!-- bilingual-en:start -->
Token counts enter the conditional conversion from [[混合权重与Token暴露|mixture weights to source exposure]], serve as a budget variable in compute-optimal allocation, and form the denominator of mean NLL in perplexity. Tokens are valid units within each declared pipeline, but different tokenizers do not thereby share a comparison unit. Raw-PPL comparison and BPB are handled in [[困惑度可比性|perplexity comparability]].
<!-- bilingual-en:end -->

## 可复现的计数声明
<!-- bilingual-en:start -->
*A reproducible counting declaration*
<!-- bilingual-en:end -->

“训练了 1T tokens”至少应绑定 tokenizer 名称与版本、关键预处理和 special-token 规则、计数是 unique 还是包含重复暴露的 total，以及重复采样如何影响总暴露。若要说明原始文本覆盖，还应另报 byte、character 或 document 统计，不能把 token 总数当作它们的同义词。

<!-- bilingual-en:start -->
“Trained on 1T tokens” should identify the tokenizer and version, material preprocessing and special-token rules, whether the count is unique data or total exposure, and how resampling affects exposure. Raw-text coverage needs separate byte, character, or document statistics.
<!-- bilingual-en:end -->

> [!example] 同一 token 预算覆盖不同字符数
> Llama 3 在其英语样本上报告新 tokenizer 为 3.94 characters/token，Llama 2 为 3.17 characters/token。在该样本与处理口径内，同样 1T tokens 对应的字符覆盖不同；这些比率不能无条件外推到其他语言、代码或清洗规则。

> [!question]- 自检
> tokenizer A 把同一 1 GB 文本编码成 2.5 亿 token，tokenizer B 编码成 4 亿 token。两个模型都训练 4 亿 token，能否仅据此说它们看过相同原始文本量？
>
> **答案：** 不能。必须查看预处理、采样与重复规则，并另报 byte、character 或 document 覆盖。

## 来源与核验

- [Grattafiori et al. (2024), *The Llama 3 Herd of Models*](https://arxiv.org/pdf/2407.21783)，第 3.2 节：128K vocabulary 与英语样本上的 characters/token 变化。
