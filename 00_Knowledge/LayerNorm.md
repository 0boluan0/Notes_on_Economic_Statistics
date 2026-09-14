---
aliases:
  - "LayerNorm 用单个样本的一组层内特征计算归一化统计，再施加可学习的逐特征仿射变换"
  - Layer normalization
  - 层归一化
student_os: knowledge-atom
atom_id: LLM-TF-011
atom_type: definition
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# LayerNorm 用单个样本的一组层内特征计算归一化统计，再施加可学习的逐特征仿射变换

<!-- bilingual-en:start -->
*LayerNorm computes normalization statistics from a group of within-layer features of one example, then applies a learned affine transformation to each feature.*
<!-- bilingual-en:end -->

LayerNorm（层归一化）是一种对单个样本的层内特征进行归一化的方法：先从指定特征组计算均值与方差，把各特征中心化并按标准差缩放，再施加可学习的逐特征缩放和偏移。在 Transformer 的常见表示 $X\in\mathbb R^{n\times d_{\mathrm{model}}}$ 中，每行是一个 token，LayerNorm 分别在每行的 $d_{\mathrm{model}}$ 个特征上进行这一运算。

<!-- bilingual-en:start -->
LayerNorm normalizes within-layer features of a single example. It computes a mean and variance over a specified feature group, centers and rescales those features, and applies a learned scale and shift to each feature. For a typical Transformer representation $X\in\mathbb R^{n\times d_{\mathrm{model}}}$, each row is one token, and LayerNorm operates separately on the $d_{\mathrm{model}}$ features in each row.
<!-- bilingual-en:end -->

## 统计量沿哪个轴计算

<!-- bilingual-en:start -->
*The axis over which statistics are computed.*
<!-- bilingual-en:end -->

记 $d=d_{\mathrm{model}}$、$x_{ia}$ 为第 $i$ 个 token 的第 $a$ 个特征：

<!-- bilingual-en:start -->
Write $d=d_{\mathrm{model}}$, and let $x_{ia}$ denote feature $a$ of token $i$:
<!-- bilingual-en:end -->

$$
\mu_i=\frac1d\sum_{a=1}^d x_{ia},\qquad
\sigma_i^2=\frac1d\sum_{a=1}^d(x_{ia}-\mu_i)^2,
$$

$$
\operatorname{LN}(x_i)
=\gamma\odot\frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}}+\beta.
$$

$\gamma,\beta\in\mathbb R^d$ 分别是可学习的逐特征缩放与偏移；$\odot$ 表示逐元素乘法。$\varepsilon>0$ 是实现中加入的数值稳定项。同一 LayerNorm 模块在各 token 位置共享 $\gamma,\beta$，但每个位置由自己的这一行算出 $\mu_i,\sigma_i^2$。计算一行的统计量不需要其他 token 或其他样本。

<!-- bilingual-en:start -->
The learned vectors $\gamma,\beta\in\mathbb R^d$ provide a scale and shift for each feature, and $\odot$ denotes element-wise multiplication. The implementation adds $\varepsilon>0$ for numerical stability. One LayerNorm module shares $\gamma,\beta$ across token positions, while each position computes $\mu_i,\sigma_i^2$ from its own row. Its statistics require no other tokens or examples.
<!-- bilingual-en:end -->

## 两个 token 各算各的统计量

<!-- bilingual-en:start -->
*Two tokens compute separate statistics.*
<!-- bilingual-en:end -->

令两行输入为 $x_1=[1,3]$、$x_2=[2,6]$。第一行的均值为 $2$、方差为 $1$；第二行的均值为 $4$、方差为 $4$。取 $\gamma=[1,1]$、$\beta=[0,0]$ 时：

<!-- bilingual-en:start -->
For two input rows $x_1=[1,3]$ and $x_2=[2,6]$, the first row has mean $2$ and variance $1$, while the second has mean $4$ and variance $4$. With $\gamma=[1,1]$ and $\beta=[0,0]$:
<!-- bilingual-en:end -->

$$
\operatorname{LN}(x_1)=\frac{[-1,1]}{\sqrt{1+\varepsilon}},\qquad
\operatorname{LN}(x_2)=\frac{[-2,2]}{\sqrt{4+\varepsilon}}.
$$

该算例先固定仿射参数，展示中心化与缩放怎样进行。学习得到的 $\gamma,\beta$ 会继续调节每个输出特征。在 [[Transformer后归一化]] 中，LayerNorm 的输入就是 [[残差连接]] 相加后的向量，按同样的特征轴计算。

<!-- bilingual-en:start -->
This example fixes the affine parameters to isolate centering and scaling. Learned $\gamma,\beta$ subsequently adjust each output feature. In [[Transformer后归一化|original Transformer post-normalization]], LayerNorm receives the vector produced by [[残差连接|residual addition]] and computes along the same feature axis.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification.*
<!-- bilingual-en:end -->

- [Ba, Kiros and Hinton (2016), *Layer Normalization*](https://arxiv.org/pdf/1607.06450)，§3、式 (3)，§3.1、式 (4)，§5.1、式 (5)：支持单个样本内部的统计量、逐特征 gain/bias，以及不同位置分别计算统计量的定义。
  <!-- bilingual-en:start -->
  Section 3, Equation (3), Section 3.1, Equation (4), and Section 5.1, Equation (5), establish within-example statistics, per-feature gain and bias, and separate statistics across sequence positions.
  <!-- bilingual-en:end -->
- [PyTorch, LayerNorm 官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)，算子公式与 NLP example：支持含 $\varepsilon$ 的实现公式、以 $d$ 为分母的方差，以及对 token 表示末尾特征轴的应用。此数值稳定项是实现表达中的补充。
  <!-- bilingual-en:start -->
  The operator formula and NLP example document the implementation's $\varepsilon$ term, variance with denominator $d$, and normalization over the final feature axis of token representations. The stability term supplements the original paper's mathematical expression.
  <!-- bilingual-en:end -->
