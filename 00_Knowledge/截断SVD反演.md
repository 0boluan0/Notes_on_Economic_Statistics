---
aliases:
  - Truncated SVD inversion
  - TSVD inverse solution
student_os: knowledge-atom
atom_id: LA-PINV-012
atom_set: pseudoinverse-one-sided-inverses
atom_type: procedure
status: source-checked
requires:
  - "[[伪逆SVD公式]]"
  - "[[数值秩]]"
  - "[[小奇异值放大噪声]]"
related:
  - "[[截断SVD]]"
  - "[[秩亏最小二乘算法]]"
leads_to:
  - "[[截断SVD与岭正则化]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
---

<!-- bilingual-en:start -->
# Truncated SVD inversion
<!-- bilingual-en:end -->

*截断 SVD 反演*

<!-- bilingual-en:start -->
Recovering an input from a weak output requires dividing by a small number. That also enlarges errors. **Truncated SVD inversion** chooses a set of sufficiently strong singular directions to invert and sets the recovered coefficients in the other directions to zero. It suppresses some noise amplification by also giving up some possible signal.
<!-- bilingual-en:end -->

从微弱输出恢复输入，需要除以很小的数，也会放大误差。**截断 SVD 反演**选择足够强的一组奇异方向进行反演，其余方向的恢复系数设为零。它抑制部分噪声放大的同时，也放弃了部分可能存在的信号。

<!-- bilingual-en:start -->
## A weak measurement direction
<!-- bilingual-en:end -->

## 最小例子

<!-- bilingual-en:start -->
Let $A=\operatorname{diag}(1,10^{-4})$, a [[对角矩阵|diagonal matrix]]. It leaves the first input coordinate unchanged and multiplies the second by $10^{-4}$. A true input $(1,1)^T$ therefore produces noiseless output $b_0=(1,10^{-4})^T$. If an error $10^{-3}$ is added to the second measurement, the exact inverse returns $(1,11)^T$. Here $T$ denotes [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

令 $A=\operatorname{diag}(1,10^{-4})$，这是一个[[对角矩阵]]：第一输入坐标不变，第二坐标乘 $10^{-4}$。真实输入 $(1,1)^T$ 产生无噪声输出 $b_0=(1,10^{-4})^T$；第二个测量若增加误差 $10^{-3}$，精确反演便返回 $(1,11)^T$。这里 $T$ 表示[[矩阵转置|转置]]。

> [!example] Keep the first direction, discard the second / 保留第一方向，舍去第二方向
> <!-- bilingual-en:start -->
> With threshold $\tau=10^{-3}$, the stretches $1$ and $10^{-4}$ fall on opposite sides of the cutoff. The truncated answer is $(1,0)^T$, with or without that measurement error. It avoids the second direction's $10^4$ noise multiplier, but it also loses its true coefficient 1. With perfect data the exact inverse was already correct; truncation then makes the recovery worse.
> <!-- bilingual-en:end -->
>
> 取阈值 $\tau=10^{-3}$，伸缩量 $1$ 与 $10^{-4}$ 分别落在截断线两侧。无论是否加入上述误差，截断答案都为 $(1,0)^T$：第二方向的 $10^4$ 倍噪声放大被消除，但真实系数1也丢失了。如果数据完全准确，原精确反演本来正确，截断反而使恢复变差。

<!-- bilingual-en:start -->
## Define the retained directions
<!-- bilingual-en:end -->

*写清保留的方向*

<!-- bilingual-en:start -->
Fix a real or complex $m\times n$ matrix $A$, with $m,n\ge1$, and an output vector $b$ with $m$ coordinates. For [[矩阵秩|rank]] $r>0$, its [[紧SVD|compact SVD]] is
<!-- bilingual-en:end -->

固定实或复 $m\times n$ 矩阵 $A$，$m,n\ge1$，以及含 $m$ 个坐标的输出向量 $b$。若[[矩阵秩]]为 $r>0$，其[[紧SVD]]为

$$A=\sum_{i=1}^r\sigma_i u_i v_i^*,\qquad \sigma_1\ge\cdots\ge\sigma_r>0.$$

<!-- bilingual-en:start -->
The unit vectors $v_i$ are perpendicular input directions; $u_i$ are the corresponding perpendicular output directions. Each positive [[奇异值|singular value]] satisfies $Av_i=\sigma_i u_i$. The star is conjugate transpose: ordinary transpose for real matrices, with complex conjugation added otherwise. Thus $u_i^*b$ reads the output coordinate along $u_i$. Choose an absolute threshold $\tau\ge0$ and retain exactly the values strictly greater than it:
<!-- bilingual-en:end -->

单位向量 $v_i$ 是彼此垂直的输入方向，$u_i$ 是对应的垂直输出方向；正[[奇异值]]满足 $Av_i=\sigma_i u_i$。星号表示共轭转置：实矩阵时就是普通转置，复数时再作共轭。因此 $u_i^*b$ 读取 $b$ 在输出方向 $u_i$ 上的坐标。选择绝对阈值 $\tau\ge0$，恰好保留严格大于它的值：

$$
A_\tau=\sum_{\sigma_i>\tau}\sigma_i u_i v_i^*,\qquad
\hat x_\tau=A_\tau^+b=\sum_{\sigma_i>\tau}\frac{u_i^*b}{\sigma_i}v_i.
$$

<!-- bilingual-en:start -->
The first sum defines a modified matrix; the second applies its [[伪逆SVD公式|pseudoinverse]] to the data. Values equal to the threshold are discarded. Empty sums give zero, so this also covers $A=0$ and thresholds that discard every direction. A relative threshold can be specified as in [[数值秩|numerical rank]], but its actual value and units must be clear.
<!-- bilingual-en:end -->

第一个求和定义修改后的矩阵，第二个用它的[[伪逆SVD公式|伪逆]]处理数据。恰好等于阈值的方向也舍去；空求和为零，所以定义同时覆盖 $A=0$ 及舍去全部方向的情形。也可按[[数值秩]]采用相对阈值，但须说明实际阈值与单位。

<!-- bilingual-en:start -->
## Which problem does the answer solve?
<!-- bilingual-en:end -->

## 它与精确伪逆的边界

<!-- bilingual-en:start -->
The vector $\hat x_\tau$ is the [[最小范数最小二乘解|unique minimum-norm least-squares solution]] for $A_\tau$: first fit the data as closely as that modified matrix allows, then choose the shortest input among ties. Equivalently, it minimizes the original fitting error $\|Ax-b\|_2$ while restricting $x$ to the span of the retained $v_i$. Here $\|\cdot\|_2$ is [[欧氏范数|Euclidean length]].
<!-- bilingual-en:end -->

$\hat x_\tau$ 是 $A_\tau$ 的[[最小范数最小二乘解|唯一最小范数最小二乘解]]：先在修改后的矩阵允许的范围内尽量拟合，再从并列解中选最短输入。等价地，限制 $x$ 只能落在保留的 $v_i$ 所张成的空间内，再最小化原拟合误差 $\|Ax-b\|_2$。$\|\cdot\|_2$ 表示[[欧氏范数|欧氏长度]]。

<!-- bilingual-en:start -->
If every positive singular value is retained, $A_\tau=A$ and $\hat x_\tau=A^+b$. For nonzero $A$, this occurs when $\tau<\sigma_r$. If a positive value is discarded, $A_\tau^+$ is a different inverse operator from $A^+$, although the two answers can still agree for a particular $b$ with no component in the discarded output directions. In general, the truncated answer need not minimize the unrestricted original fitting error.
<!-- bilingual-en:end -->

如果保留全部正奇异值，就有 $A_\tau=A$ 和 $\hat x_\tau=A^+b$；对非零 $A$，条件是 $\tau<\sigma_r$。一旦舍去正值，$A_\tau^+$ 就是不同于 $A^+$ 的反演算子，不过某个特定 $b$ 若在被舍去的输出方向没有分量，两种答案仍可能相同。一般而言，截断解不必最小化不受限制的原拟合误差。

<!-- bilingual-en:start -->
## Signal loss and noise suppression
<!-- bilingual-en:end -->

*舍去的既可能是噪声，也可能是信号*

<!-- bilingual-en:start -->
If $b=Ax_{\rm true}+e$, where $e$ is measurement error, a retained direction has recovered coordinate $v_i^*x_{\rm true}+(u_i^*e)/\sigma_i$. A discarded direction has recovered coordinate zero, regardless of its true signal. This explains both effects without assuming a noise distribution. The original [[零空间|nullspace]] component of $x_{\rm true}$ was never identifiable from $Ax_{\rm true}$, even before truncation.
<!-- bilingual-en:end -->

若 $b=Ax_{\rm true}+e$，$e$ 为测量误差，保留方向上的恢复坐标为 $v_i^*x_{\rm true}+(u_i^*e)/\sigma_i$；被舍去方向的恢复坐标则恒为零，不论真实信号多少。无需假设噪声分布，就能看清这两种作用。$x_{\rm true}$ 原本位于[[零空间]]中的分量，即使不截断，也无法从 $Ax_{\rm true}$ 识别。

<!-- bilingual-en:start -->
The threshold is therefore an accuracy or modelling choice. A smaller inverse gain does not guarantee a smaller error relative to the true input. Selecting a useful cutoff requires information about data error and the recovery goal, not just a glance at the singular values. This card fixes the cutoff; it does not supply a universal selection rule.
<!-- bilingual-en:end -->

所以阈值是精度或建模选择：反演增益变小，不保证相对于真实输入的误差也变小。选择合适截断位置需要数据误差与恢复目标的信息，不能只看奇异值大小。本卡解释给定阈值后的运算，不提供普遍适用的选阈值规则。

<!-- bilingual-en:start -->
## Matrix approximation and inverse recovery
<!-- bilingual-en:end -->

## 不要与低秩矩阵近似混同

<!-- bilingual-en:start -->
A [[截断SVD|rank-$k$ truncated SVD]] constructs an approximation $A_k$ to the matrix itself. Here the retained matrix is $A_\tau$, but the requested output is a coefficient vector $A_\tau^+b$. Constructing the approximation and applying its inverse are different operations: one multiplies by retained singular values, the other divides by them. [[岭的SVD滤波公式|Ridge filtering]] offers a different recovery rule, with continuous shrinkage instead of this hard cutoff.
<!-- bilingual-en:end -->

[[截断SVD|秩 $k$ 截断 SVD]]构造矩阵本身的近似 $A_k$；这里保留的矩阵是 $A_\tau$，最终求的却是系数向量 $A_\tau^+b$。构造近似与使用其逆是两种运算：前者乘保留的奇异值，后者除以它们。[[岭的SVD滤波公式|岭滤波]]采用另一种恢复规则，用连续收缩代替硬截断。

## 来源与核验

<!-- bilingual-en:start -->
- [Hansen, *Intro to Inverse Problems*, Chapter 4, PDF pp.5–8](https://www2.imm.dtu.dk/~pcha/DIP/chap4.pdf#page=5): TSVD expansion, minimum-norm fitting for the truncated matrix, parameter selection and signal/noise tradeoff. The threshold form and rectangular/complex extension follow by the stated SVD calculation.
- [LAPACK, linear least squares](https://netlib.org/lapack/lug/node27.html) and [SVD conventions](https://netlib.org/lapack/lug/node32.html): minimum-norm solving, real/complex directions and rectangular dimensions. [[03_Positive Definite Matrices and Applications#3.8.8 边界、反例与易错点|Course §3.8.8]] supplies the local inverse-problem scope. The two-coordinate example is calculated directly.
<!-- bilingual-en:end -->

- Hansen 第4章 PDF5–8页核对 TSVD 展开、截断矩阵的最小范数拟合、参数选择及信号／噪声取舍；阈值口径和长方形／复数扩展由正文的 SVD 运算直接推出。
- LAPACK 最小二乘与 SVD 章节核对最小范数求解、实复方向和长方形尺寸；课程第3.8.8节确定本地反问题范围。二坐标例子直接计算。
