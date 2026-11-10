---
aliases:
  - "Gram-Schmidt orthogonalization"
  - "Classical Gram-Schmidt"
  - "经典 Gram-Schmidt 正交化"
student_os: knowledge-atom
atom_id: LA-PROJ-015
atom_set: orthogonal-projection-least-squares
atom_type: procedure
status: source-checked
requires:
  - "[[标准正交组]]"
  - "[[标准正交基投影公式]]"
leads_to:
  - "[[薄QR分解]]"
  - "[[Gram-Schmidt数值失稳]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---

<!-- bilingual-en:start -->
# Gram–Schmidt orthogonalization
<!-- bilingual-en:end -->

*Gram–Schmidt 正交化*

<!-- bilingual-en:start -->
A set of directions may describe the right plane or subspace but make coordinates awkward because the directions overlap. **Gram–Schmidt orthogonalization** replaces an ordered list of independent vectors with perpendicular unit vectors describing the same space. It removes the part of each new vector that earlier directions already explain, then rescales the remaining new direction to length one. The result is an [[标准正交组|orthonormal family]], meaning pairwise perpendicular vectors of unit length.
<!-- bilingual-en:end -->

一组方向可以描述正确的平面或子空间，却因方向彼此重叠而不便计算坐标。**Gram–Schmidt 正交化**把有顺序的线性无关向量换成描述同一空间的垂直单位向量：从每个新向量中减掉已有方向能够解释的部分，再把剩下的新方向缩放到长度一。结果是[[标准正交组]]，即两两垂直、各自长度为一的向量组。

> [!example] Two overlapping directions · 两个有重叠的方向
> <!-- bilingual-en:start -->
> Start with $a_1=(1,1,0)^T$ and $a_2=(1,0,1)^T$. They span a plane, but their dot product is $1$, so they are not perpendicular. The first unit vector is $q_1=a_1/\sqrt2$. The [[标准正交基投影公式|projection]] of $a_2$ onto this unit direction is
>
> $$q_1(q_1^Ta_2)=q_1/\sqrt2=(1/2,1/2,0)^T.$$
>
> Subtracting it gives $v_2=(1/2,-1/2,1)^T$, with length $\sqrt{3/2}$. Hence $q_2=(1,-1,2)^T/\sqrt6$. Now $q_1^Tq_2=0$ and both lengths are one. No part of the original plane was discarded: $a_2=q_1/\sqrt2+\sqrt{3/2}\,q_2$.
> <!-- bilingual-en:end -->
>
> 从 $a_1=(1,1,0)^T$、$a_2=(1,0,1)^T$ 出发，两者张成一个平面，但点积为 $1$，并不垂直。先把第一向量除以长度 $\sqrt2$ 得 $q_1$。$a_2$ 在这个单位方向上的[[标准正交基投影公式|投影]]为 $(1/2,1/2,0)^T$。减掉它，余量是 $v_2=(1/2,-1/2,1)^T$，长度为 $\sqrt{3/2}$；归一化后 $q_2=(1,-1,2)^T/\sqrt6$。两单位向量的点积为零，而上述 $a_2$ 的重建式说明原平面的信息没有丢失。

<!-- bilingual-en:start -->
## Remove the known part, then normalize
<!-- bilingual-en:end -->

*投影系数与新方向*

<!-- bilingual-en:start -->
Let $A=[a_1\ \cdots\ a_n]$ be an $m\times n$ real or complex matrix with [[线性无关|linearly independent columns]], so $m\ge n\ge1$. We use the [[欧氏范数|Euclidean norm]] $\|v\|_2$, the square root of the sum of squared absolute coordinates. Write $q^*$ for conjugate transpose: transpose and change $a+bi$ into $a-bi$, where $i^2=-1$. For real vectors it is just $q^T$.

At column $k$, the earlier $q_1,\ldots,q_{k-1}$ have already been constructed. Classical Gram–Schmidt computes

$$r_{ik}=q_i^*a_k\quad(i<k),\qquad
v_k=a_k-\sum_{i=1}^{k-1}q_i r_{ik},\qquad
r_{kk}=\|v_k\|_2,\quad q_k=v_k/r_{kk}.$$

The scalar $r_{ik}$ measures the amount of $a_k$ along the unit direction $q_i$; $q_i r_{ik}$ is the vector removed. The remainder $v_k$ contains what the earlier directions do not explain. The positive number $r_{kk}$ is its length, not another projection coefficient. At $k=1$, the sum is empty and equals zero, giving $q_1=a_1/\|a_1\|_2$.
<!-- bilingual-en:end -->

设 $A=[a_1\ \cdots\ a_n]$ 为 $m\times n$ 实或复矩阵，各列[[线性无关]]，因此 $m\ge n\ge1$。采用[[欧氏范数]] $\|v\|_2$，即各坐标绝对值平方和的平方根。$q^*$ 是共轭转置：转置后把 $a+bi$ 换成 $a-bi$，其中 $i^2=-1$；实向量时就是 $q^T$。

处理第 $k$ 列时，已有 $q_1,\ldots,q_{k-1}$，经典 Gram–Schmidt 按上述公式计算。标量 $r_{ik}$ 是 $a_k$ 沿单位方向 $q_i$ 的分量，$q_i r_{ik}$ 才是被减掉的向量。余量 $v_k$ 保留已有方向不能解释的部分；正数 $r_{kk}$ 是它的长度，并不是又一个投影系数。第一步的求和为空，按零处理，所以 $q_1=a_1/\|a_1\|_2$。

<!-- bilingual-en:start -->
For any earlier index $j<k$, orthonormality gives $q_j^*v_k=q_j^*a_k-r_{jk}=0$: every unwanted component cancels. Dividing by a nonzero length preserves these zero inner products and makes the new vector unit length. If one works with unnormalized perpendicular vectors instead, the projection coefficient must be divided by that vector's squared norm; the simpler coefficient above relies on unit length. In complex arithmetic the conjugation is essential: $(1,i)^T$ has squared length $2$, although its ordinary transpose product with itself is zero.
<!-- bilingual-en:end -->

对每个先前的 $j<k$，标准正交性给出 $q_j^*v_k=q_j^*a_k-r_{jk}=0$，恰好消去对应的旧分量。再除以非零长度，既保持这些零内积，也把新向量变成单位长度。若使用尚未归一化的垂直向量，投影系数还须除以该向量的范数平方；上面的简单系数依赖单位长度。复数情形不能省略共轭，例如 $(1,i)^T$ 的长度平方是 $2$，但与自身的普通转置乘积却为零。

<!-- bilingual-en:start -->
## Why each prefix spans the same space
<!-- bilingual-en:end -->

## 为什么张成空间不变

<!-- bilingual-en:start -->
The [[张成|span]] is the set of all linear combinations. Assume the earlier $q_i$ span the same space as $a_1,\ldots,a_{k-1}$. Subtracting their combination from $a_k$ puts $v_k$, and therefore $q_k$, in the span of the first $k$ original vectors. Conversely,

$$a_k=\sum_{i<k}q_i r_{ik}+r_{kk}q_k$$

reconstructs the original column from the new directions. These two inclusions give

$$\operatorname{span}(q_1,\ldots,q_k)=\operatorname{span}(a_1,\ldots,a_k).$$

Independence also explains why normalization is legal: if $v_k=0$, then $a_k$ was already a combination of earlier columns, contradicting the hypothesis.
<!-- bilingual-en:end -->

[[张成|张成空间]]是全部线性组合的集合。假定先前 $q_i$ 与 $a_1,\ldots,a_{k-1}$ 张成同一空间，从 $a_k$ 减去它们的组合，所得 $v_k$ 及其归一化 $q_k$ 仍在原前 $k$ 列的张成空间内。反过来，显示的重建式把 $a_k$ 写成新方向的组合。两个方向的包含关系，给出每一步前缀张成空间相等。

线性无关还保证归一化合法：若 $v_k=0$，则 $a_k$ 已是先前列的线性组合，与假设矛盾。

> [!warning] A zero remainder needs a different action · 余量为零时不能继续相除
> <!-- bilingual-en:start -->
> If dependent input vectors are allowed, an exactly zero remainder means that this column contributes no new direction. For example, appending $a_3=a_1+a_2$ to the opening example gives zero remainder. One may skip that column to build a basis of the span, but one must not divide by zero or claim to have produced $n$ orthonormal vectors. A very small computed remainder is a separate [[Gram-Schmidt数值失稳|floating-point reliability]] question, not automatic proof of exact dependence.
> <!-- bilingual-en:end -->
>
> 若允许输入相关向量，精确的零余量说明该列没有贡献新方向。例如在开头两列后添加 $a_3=a_1+a_2$，余量就是零。为了构造张成空间的基，可以跳过它，但不能除以零，也不能声称得到 $n$ 个标准正交向量。计算所得余量很小，则还涉及[[Gram-Schmidt数值失稳|浮点可靠性]]，并不自动证明精确线性相关。

<!-- bilingual-en:start -->
## How the coefficients form a QR factorization
<!-- bilingual-en:end -->

## 与 QR 分解的关系

<!-- bilingual-en:start -->
Set $Q=[q_1\ \cdots\ q_n]$ and collect the coefficients into an $n\times n$ matrix $R$, with $r_{ik}=0$ for $i>k$. It is upper triangular because a column uses only its own and earlier directions. The reconstruction formulas become

$$A=QR,\qquad Q^*Q=I_n,$$

where $I_n$ is the [[单位矩阵|identity matrix]]. This constructs a [[薄QR分解|thin QR factorization]] with positive diagonal entries in $R$. The order of the input columns determines the sequence of directions; preserving the final span does not make every ordering produce the same $Q$. The exact construction and its [[Gram-Schmidt数值失稳|numerical implementation]] should be assessed separately.
<!-- bilingual-en:end -->

令 $Q=[q_1\ \cdots\ q_n]$，把系数收进 $n\times n$ 矩阵 $R$，对 $i>k$ 令 $r_{ik}=0$。每列只使用自身及先前方向，因此 $R$ 为上三角。全部重建式合起来就是 $A=QR$、$Q^*Q=I_n$，其中 $I_n$ 为[[单位矩阵]]。这样构造出对角元为正的[[薄QR分解]]。输入列的顺序决定逐步得到的方向；最终张成空间不变，并不表示任何排列都会得到同一 $Q$。精确构造与[[Gram-Schmidt数值失稳|数值实现]]需要分别评估。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|MIT 18.06SC Session 2.4]], pp. 2–3: subtraction of projections, normalization and the QR interpretation.
- [Sheldon Axler, *Linear Algebra Done Right*, fourth edition](https://linear.axler.net/LADR4e.pdf), Theorem 6.32, printed p. 201 / PDF p. 215; complex inner-product conventions on printed pp. 183–184: prefix-span preservation and nonzero remainders. Here the projection coefficient is written as $q_i^*a_k$, equivalent to Axler's inner product linear in its first slot. The coordinate example is calculated directly.
<!-- bilingual-en:end -->

- MIT Session 2.4 第 2–3 页核对减投影、归一化及 QR 联系。
- Axler 第 6.32 条（印刷第 201 页 / PDF 第 215 页）核对前缀张成空间与非零余量；印刷第 183–184 页核对复内积约定。正文用 $q_i^*a_k$ 表示投影系数，对应书中第一槽线性的内积写法；小坐标例直接计算。
