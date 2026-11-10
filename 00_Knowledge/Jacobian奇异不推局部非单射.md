---
aliases:
  - A singular Jacobian does not imply local non-injectivity
  - Singular Jacobian and injectivity
student_os: knowledge-atom
atom_id: CALC-MV-019
atom_set: multivariable-differentiation
atom_type: implication-boundary
status: source-checked
requires:
  - "[[逆函数定理]]"
related:
  - "[[Jacobian行列式]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

<!-- bilingual-en:start -->
# A singular Jacobian and local injectivity
<!-- bilingual-en:end -->

*Jacobian 奇异与局部单射*

<!-- bilingual-en:start -->
A zero derivative does not necessarily mean that nearby inputs become indistinguishable. The function $F(t)=t^3$ has $F'(0)=0$, yet different real inputs always give different outputs. What the zero derivative loses is an invertible **first-order approximation**. This distinction separates recovering an input from recovering it with a differentiable inverse.
<!-- bilingual-en:end -->

导数为零，并不一定表示附近输入变得无法区分。函数 $F(t)=t^3$ 满足 $F'(0)=0$，不同实数输入却总能得到不同输出。零导数失去的是可逆的**一阶近似**。因此，“能够找回输入”和“能够用可微的逆映射找回输入”需要区分。

<!-- bilingual-en:start -->
Let $F:U\to\mathbb R^n$ be differentiable at $a$, with $U\subseteq\mathbb R^n$ [[开集|open]] and integer $n\ge1$. Its [[Jacobian矩阵|Jacobian]] represents the total derivative $DF(a)$. “Singular” means this square matrix has no inverse, equivalently its [[Jacobian行列式|determinant]] is zero. “Locally injective at $a$” means some open neighborhood of $a$ contains no two distinct points with the same output.
<!-- bilingual-en:end -->

设 $F:U\to\mathbb R^n$ 在 $a$ 可微，$U\subseteq\mathbb R^n$ 为[[开集]]，维数 $n\ge1$ 有限。它的[[Jacobian矩阵]]表示全微分 $DF(a)$。“奇异”是指这个方阵没有逆，等价地其[[Jacobian行列式|行列式]]为零。“在 $a$ 附近局部单射”则表示存在 $a$ 的某个开邻域，其中任意两个不同的点都不会得到相同输出。

<!-- bilingual-en:start -->
## The cubic keeps distinct inputs distinct
<!-- bilingual-en:end -->

*三次函数仍能区分不同输入*

<!-- bilingual-en:start -->
For $s<t$, factor the output difference:
<!-- bilingual-en:end -->

对 $s<t$，将输出差因式分解：

$$t^3-s^3=(t-s)(t^2+ts+s^2)>0.$$

<!-- bilingual-en:start -->
The second factor is $(t+s/2)^2+3s^2/4$ and cannot be zero for distinct $s,t$. Thus the cubic is strictly increasing and injective on the entire real line. It also takes every real output once, with continuous inverse $G(y)=\sqrt[3]{y}$. Nevertheless, at zero its inverse difference quotient is
<!-- bilingual-en:end -->

第二个因子等于 $(t+s/2)^2+3s^2/4$，当 $s,t$ 不同时不可能为零。所以三次函数在整个实线上严格递增、单射，而且每个实数输出都恰好出现一次，连续逆函数为 $G(y)=\sqrt[3]{y}$。然而，在零点处逆函数的差商为

$$\frac{G(h)-G(0)}h=|h|^{-2/3}\longrightarrow+\infty\qquad(h\to0,\ h\ne0).$$

<!-- bilingual-en:start -->
There is no finite derivative of $G$ at zero. In several dimensions, $F(x_1,\ldots,x_n)=(x_1^3,x_2,\ldots,x_n)$ shows the same phenomenon: it is globally bijective, but its Jacobian is singular wherever $x_1=0$.
<!-- bilingual-en:end -->

因此 $G$ 在零点处没有有限导数。在多维中，$F(x_1,\ldots,x_n)=(x_1^3,x_2,\ldots,x_n)$ 显示同一现象：它全局双射，但凡 $x_1=0$，Jacobian 都奇异。

<!-- bilingual-en:start -->
## What singularity really rules out
<!-- bilingual-en:end -->

*奇异性确实排除了什么*

<!-- bilingual-en:start -->
Suppose there were an inverse $G$ between open neighborhoods, differentiable at $b=F(a)$. The identity $G(F(x))=x$ and the [[多元链式法则|chain rule]] would imply
<!-- bilingual-en:end -->

若两个开邻域之间存在逆映射 $G$，且它在 $b=F(a)$ 可微，那么由 $G(F(x))=x$ 和[[多元链式法则]]，必有

$$DG(b)\,DF(a)=I_n.$$

<!-- bilingual-en:start -->
Here $I_n$ is the [[单位矩阵|identity matrix]]. A singular $DF(a)$ has a nonzero vector $v$ in its [[零空间|null space]], meaning $DF(a)v=0$. Multiplying the displayed identity by $v$ would give $0=v$, a contradiction. Thus, when $F$ is differentiable at $a$, a singular derivative **rules out a differentiable local inverse at $b$**, not merely the ability to apply a particular theorem. It does not rule out a continuous inverse or local injectivity.
<!-- bilingual-en:end -->

这里 $I_n$ 是[[单位矩阵]]。奇异的 $DF(a)$ 的[[零空间]]含有某个非零向量 $v$，即 $DF(a)v=0$。用上式作用于 $v$，便得到 $0=v$，矛盾。因此，只要 $F$ 在 $a$ 可微，奇异导数就**确实排除了在 $b$ 可微的局部逆**，并不只是让某条定理暂时不能用。它没有排除连续逆或局部单射。

> [!important] Keep the conclusions separate · 分开不同结论
> <!-- bilingual-en:start -->
> Under the $C^1$ hypothesis, the [[逆函数定理|inverse function theorem]] guarantees a $C^1$ local inverse from an invertible derivative. Conversely, a differentiable local inverse forces that derivative to be invertible by the chain rule. Mere local injectivity is weaker: the cubic shows that it does not force a nonzero determinant.
> <!-- bilingual-en:end -->
>
> 在 $C^1$ 假设下，[[逆函数定理]]由导数可逆保证 $C^1$ 局部逆。反过来，可微局部逆也会由链式法则强迫导数可逆。单独的局部单射更弱：三次函数说明，它不强迫行列式非零。

<!-- bilingual-en:start -->
Nor does singularity prove injectivity. The smooth map $H(t)=t^2$ also has $H'(0)=0$, but every neighborhood of zero contains opposite nonzero inputs with equal output. The two functions $t^3$ and $t^2$ have the same derivative at zero and opposite answers to local injectivity. Examine the actual map, not only its first-order matrix.
<!-- bilingual-en:end -->

奇异性同样不能证明单射。光滑映射 $H(t)=t^2$ 也满足 $H'(0)=0$，但零点的每个邻域都含有互为相反数的非零输入，且二者输出相同。$t^3$ 与 $t^2$ 在零点的导数相同，局部单射性的答案却相反；需要查看原映射，而不只是一阶矩阵。

## 来源与核验

<!-- bilingual-en:start -->
### Sources and verification

- [Harvard Math 25b, inverse-function discussion](https://people.math.harvard.edu/~elkies/M25b.13/index.html): explicitly distinguishes a differentiable inverse, which requires an invertible derivative, from a continuous inverse such as the cube root.
- [Lebl, Basic Analysis II, §8.5, PDF pp.51–54](https://www.jirka.org/ra/realanal2.pdf#page=51): the local $C^1$ theorem and the cubic boundary example. The signed cube-root quotient, the quadratic comparison, and the chain-rule necessity argument are checked explicitly here.
- [Stanford Math 174A, Theorem 1](https://math.stanford.edu/~andras/174A-2.pdf): the precise smoothness conclusion to distinguish from injectivity alone.
<!-- bilingual-en:end -->

- Harvard Math25b 的逆函数讨论：明确区分需要可逆导数的可微逆，与立方根这样的连续逆。
- Lebl《Basic Analysis II》第8.5节，PDF51–54页：局部 $C^1$ 定理与三次函数边界例；双侧立方根差商、平方函数对照及链式法则必要性均在正文核验。
- Stanford Math174A 定理1：核对需要与单射性区分的光滑逆结论。
