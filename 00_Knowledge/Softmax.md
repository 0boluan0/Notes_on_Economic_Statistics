---
aliases:
  - "Softmax 是把一组实数分数指数归一化为概率向量的函数"
  - "Softmax maps real-valued scores to a probability vector by exponentiating and normalizing them"
  - "Softmax function"
student_os: knowledge-atom
atom_id: LLM-TF-003
atom_type: definition
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# Softmax 是把一组实数分数指数归一化为概率向量的函数

<!-- bilingual-en:start -->
*Softmax maps real-valued scores to a probability vector by exponentiating and normalizing them*
<!-- bilingual-en:end -->

Softmax 对一组有限实数分数分别取指数，再除以这些指数的总和，得到概率向量。对于非空分数组 $s=(s_1,\ldots,s_m)\in\mathbb R^m$，定义为

<!-- bilingual-en:start -->
Softmax exponentiates each score in a finite real-valued vector and divides by the sum of those exponentials, producing a probability vector. For a nonempty vector $s=(s_1,\ldots,s_m)\in\mathbb R^m$, it is defined by
<!-- bilingual-en:end -->

$$
p_j=\operatorname{softmax}(s)_j
=\frac{e^{s_j}}{\sum_{\ell=1}^{m}e^{s_\ell}},
\qquad j=1,\ldots,m.
$$

在精确实数运算下，每项 $p_j>0$，且 $\sum_jp_j=1$。这两个性质来自指数为正及共同分母；它们说明输出可作为归一化权重使用，本身不证明这些权重已经表达真实或校准过的事件概率。

<!-- bilingual-en:start -->
In exact real arithmetic, every $p_j$ is positive and $\sum_jp_j=1$, because the exponentials are positive and share the same denominator. These properties allow the output to serve as normalized weights; they do not establish that the weights are true or calibrated event probabilities.
<!-- bilingual-en:end -->

## 从分数到权重

<!-- bilingual-en:start -->
*From scores to weights*
<!-- bilingual-en:end -->

取 $s=(\log2,0)$，两项指数是 2 和 1，所以

<!-- bilingual-en:start -->
For $s=(\log2,0)$, the exponentials are 2 and 1, giving
<!-- bilingual-en:end -->

$$
\operatorname{softmax}(\log2,0)=(2/3,1/3).
$$

第一个分数比第二个高 $\log2$，于是它的权重是第二个的两倍。一般地，从定义消去共同分母即可得到 $p_i/p_j=e^{s_i-s_j}$：权重比由分数差决定，而不是直接等于分数之比。

<!-- bilingual-en:start -->
The first score exceeds the second by $\log2$, so its weight is twice as large. Cancelling the common denominator gives the general relation $p_i/p_j=e^{s_i-s_j}$: weight ratios depend exponentially on score differences, rather than equalling score ratios.
<!-- bilingual-en:end -->

## 矩阵输入必须指明归一化轴

<!-- bilingual-en:start -->
*A matrix input requires a normalization axis*
<!-- bilingual-en:end -->

[[缩放点积注意力]]的分数矩阵有 $n_q$ 行 query、$n_k$ 列 key。每一行单独沿 key 轴做 Softmax，所以每个 query 在自己的候选集合内分配总和为 1 的权重；不同 query 的权重不会共用一个总和。[[语言模型输出头]]若对词表 logits 使用 Softmax，归一化轴则是词表条目。

<!-- bilingual-en:start -->
The score matrix in [[缩放点积注意力|scaled dot-product attention]] has $n_q$ query rows and $n_k$ key columns. Softmax runs separately across the key axis of each row, so every query distributes unit total weight over its own candidates. Different queries do not share a normalization sum. When a [[语言模型输出头|language-model output head]] applies Softmax to vocabulary logits, the normalization axis consists of vocabulary entries instead.
<!-- bilingual-en:end -->

[[注意力掩码归一化]]会把被排除位置视作 $-\infty$ 分数，并以极限意义令其指数为零；此时至少要留有一个有限分数，才能得到有效的归一化分布。整行都是 $-\infty$ 不属于上面的有限实数定义，也不能直接用该公式归一化。

<!-- bilingual-en:start -->
[[注意力掩码归一化|Masked attention normalization]] treats excluded positions as having score $-\infty$, with zero exponential understood as a limit. At least one finite score must remain for a valid normalized distribution. A row containing only $-\infty$ lies outside the finite-real definition above and cannot be normalized directly by that formula.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Blanchard, Higham & Higham, *Accurately computing the log-sum-exp and softmax functions*](https://webhomes.maths.ed.ac.uk/~dhigham/Publications/P149.pdf)，§1 式(1.2)，[DOI: 10.1093/imanum/draa038](https://doi.org/10.1093/imanum/draa038)：支持有限实数向量上的 Softmax 定义。正性、归一化、权重比和两分数例子由定义直接推得；全屏蔽行的限制来自分母为零。
  <!-- bilingual-en:start -->
  Section 1, equation (1.2), defines Softmax on finite real vectors. Positivity, normalization, the weight ratio, and the two-score example follow directly from that definition. The fully masked-row restriction follows from a zero denominator.
  <!-- bilingual-en:end -->
- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/html/1706.03762v7)，§3.2.1、§3.2.3、§3.4：分别支持注意力权重归一化、softmax 前屏蔽连接，以及输出层的词表概率应用。
  <!-- bilingual-en:start -->
  Sections 3.2.1, 3.2.3, and 3.4 support attention-weight normalization, masking before Softmax, and vocabulary probabilities at the output layer, respectively.
  <!-- bilingual-en:end -->
