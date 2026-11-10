---
aliases:
  - SVD filter formula for ridge regularization
  - Ridge spectral filter factors
student_os: knowledge-atom
atom_id: LA-PINV-017
atom_set: pseudoinverse-one-sided-inverses
atom_type: formula
status: source-checked
requires:
  - "[[岭正则化]]"
  - "[[奇异值分解]]"
  - "[[小奇异值放大噪声]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
leads_to:
  - "[[截断SVD与岭正则化]]"
related:
  - "[[数值秩]]"
---

<!-- bilingual-en:start -->
# Ridge filtering in singular directions
<!-- bilingual-en:end -->

*岭的 SVD 滤波公式*

<!-- bilingual-en:start -->
The exact inverse of a weak measurement can amplify noise sharply. [[岭正则化|Ridge regularization]] accepts some fitting error in return for smaller coefficients. In singular coordinates, this tradeoff becomes especially clear: each direction receives its own smooth gain, and weak directions are shrunk more strongly relative to exact inversion.
<!-- bilingual-en:end -->

微弱测量的精确反演可能大幅放大噪声。[[岭正则化]]允许牺牲部分拟合，以换取更小的系数。在奇异方向坐标中，这种取舍很清楚：每个方向获得一个平滑增益，相对于精确反演，弱方向被收缩得更多。

<!-- bilingual-en:start -->
Fix a real or complex $m\times n$ matrix $A$, with $m,n\ge1$, and data $b$ with $m$ coordinates. This card uses the objective
<!-- bilingual-en:end -->

固定实或复 $m\times n$ 矩阵 $A$，$m,n\ge1$，以及含 $m$ 个坐标的数据 $b$。本卡采用目标

$$\hat x_\lambda=\arg\min_x\{\|Ax-b\|_2^2+\lambda\|x\|_2^2\},\qquad\lambda>0.$$

<!-- bilingual-en:start -->
The [[欧氏范数|Euclidean squared length]] $\|x\|_2^2$ adds squared absolute values of all $n$ coordinates. **Every coordinate of $x$ is penalized here.** The first term measures fitting error; $\lambda$ controls the cost of coefficient size. No probability or Gaussian assumption is needed for this minimization.
<!-- bilingual-en:end -->

[[欧氏范数|欧氏长度平方]] $\|x\|_2^2$ 是全部 $n$ 个坐标绝对值的平方和，**这里惩罚 $x$ 的每个坐标**。第一项度量拟合误差，$\lambda$ 控制系数大小的代价。求这个最小值无需概率或 Gaussian 假设。

<!-- bilingual-en:start -->
## Separate input and output directions
<!-- bilingual-en:end -->

*分开输入与输出方向*

<!-- bilingual-en:start -->
For [[矩阵秩|rank]] $r>0$, use the [[紧SVD|compact SVD]] $A=U_r\Sigma_rV_r^*$. Its positive [[奇异值|singular values]] are $\sigma_1,\ldots,\sigma_r$. The columns $v_i$ of $V_r$ are perpendicular unit input directions; the columns $u_i$ of $U_r$ are their corresponding output directions, with $Av_i=\sigma_i u_i$. The star is conjugate transpose, equal to ordinary [[矩阵转置|transpose]] in the real case. Thus $u_i^*b$ is the measured output coordinate in direction $u_i$.
<!-- bilingual-en:end -->

当[[矩阵秩]]为 $r>0$，使用[[紧SVD]] $A=U_r\Sigma_rV_r^*$，其中正[[奇异值]]为 $\sigma_1,\ldots,\sigma_r$。$V_r$ 的列 $v_i$ 是相互垂直的单位输入方向，$U_r$ 的列 $u_i$ 是配对输出方向，满足 $Av_i=\sigma_i u_i$。星号表示共轭转置，在实数域中就是普通[[矩阵转置|转置]]；$u_i^*b$ 因而是测量数据沿 $u_i$ 的输出坐标。

> [!important] Smooth inverse gains / 平滑的反演增益
> <!-- bilingual-en:start -->
> The unique ridge solution is
>
> $$\hat x_\lambda=\sum_{i=1}^r
> \underbrace{\frac{\sigma_i}{\sigma_i^2+\lambda}}_{\text{inverse gain}}
> (u_i^*b)v_i.$$
>
> Zero singular directions receive coefficient zero; if $A=0$, the sum is empty and the answer is zero. Unlike [[截断SVD反演|TSVD]], ridge does not discard a positive direction by a hard cutoff.
> <!-- bilingual-en:end -->
>
> 唯一岭解如上。零奇异方向的系数取零；若 $A=0$，求和为空，答案为零。与[[截断SVD反演]]不同，岭不会通过硬阈值把某个正奇异方向整体舍去。

<!-- bilingual-en:start -->
To see why, write $x=\sum_i z_i v_i+x_0$, where $x_0$ belongs to the [[零空间|nullspace]] $N(A)$ and is perpendicular to all $v_i$. Set $c_i=u_i^*b$ and let $b_\perp$ be the part of $b$ perpendicular to the [[列空间|attainable output space]]. By perpendicularity, the objective becomes
<!-- bilingual-en:end -->

为理解公式，写成 $x=\sum_i z_i v_i+x_0$，其中 $x_0$ 属于[[零空间]] $N(A)$，且垂直于全部 $v_i$。令 $c_i=u_i^*b$，$b_\perp$ 是 $b$ 垂直于[[列空间|可达输出空间]]的部分。利用正交性，目标变成

$$\sum_{i=1}^r\bigl(|\sigma_i z_i-c_i|^2+\lambda|z_i|^2\bigr)
+\|b_\perp\|_2^2+\lambda\|x_0\|_2^2.$$

<!-- bilingual-en:start -->
Each $z_i$ can be chosen separately. Completing the square gives $z_i=\sigma_i c_i/(\sigma_i^2+\lambda)$; the positive penalty makes $x_0=0$. This proves the formula and uniqueness even when $A$ is rank deficient. Equivalently,
<!-- bilingual-en:end -->

各 $z_i$ 可以分别选择；配方得到 $z_i=\sigma_i c_i/(\sigma_i^2+\lambda)$，正惩罚又使 $x_0=0$。这同时证明公式与唯一性，即使 $A$ 秩亏也成立。等价地，

$$(A^*A+\lambda I_n)\hat x_\lambda=A^*b,$$

<!-- bilingual-en:start -->
where $I_n$ is the [[单位矩阵|identity]]. On each $v_i$, the matrix on the left multiplies by $\sigma_i^2+\lambda$, and on the nullspace it multiplies by $\lambda$. In compact notation it equals $V_r(\Sigma_r^2+\lambda I_r)V_r^*+\lambda P_{N(A)}$, where $P_{N(A)}$ is [[正交投影|orthogonal projection]] onto the nullspace. All these multipliers are positive, so the matrix is invertible.
<!-- bilingual-en:end -->

$I_n$ 是[[单位矩阵]]。左侧矩阵在每个 $v_i$ 方向乘 $\sigma_i^2+\lambda$，在零空间方向乘 $\lambda$；紧记法为 $V_r(\Sigma_r^2+\lambda I_r)V_r^*+\lambda P_{N(A)}$，其中 $P_{N(A)}$ 是到零空间的[[正交投影]]。这些乘数全部为正，所以矩阵可逆。

<!-- bilingual-en:start -->
## Gain and retained fraction are different quantities
<!-- bilingual-en:end -->

*反演增益与保留比例是两个量*

<!-- bilingual-en:start -->
The [[伪逆SVD公式|exact pseudoinverse]] uses gain $1/\sigma_i$. Ridge multiplies that gain by the dimensionless retained fraction
<!-- bilingual-en:end -->

[[伪逆SVD公式|精确伪逆]]的增益是 $1/\sigma_i$。岭相当于再乘一个无量纲的保留比例

$$f_i=\frac{\sigma_i^2}{\sigma_i^2+\lambda},\qquad
\frac{\sigma_i}{\sigma_i^2+\lambda}=f_i\frac1{\sigma_i}.$$

<!-- bilingual-en:start -->
For positive $\sigma_i$, $0<f_i<1$, and smaller singular values have smaller retained fractions. The inverse gain itself need not be below 1: for $\sigma_i=0.1$ and $\lambda=0.01$, it is 5, while exact inversion has gain 10. Thus “shrinkage” is relative to exact inversion in that direction, not a claim that every output coordinate is numerically reduced below its original size.
<!-- bilingual-en:end -->

正 $\sigma_i$ 满足 $0<f_i<1$，较小奇异值对应较小的保留比例。反演增益本身却不必小于1：当 $\sigma_i=0.1$、$\lambda=0.01$，它为5，精确反演增益则为10。所以“收缩”是相对同方向的精确反演而言，并非断言输出坐标经过运算后一定比原数值小。

<!-- bilingual-en:start -->
## A weak direction is shrunk strongly
<!-- bilingual-en:end -->

## 最小例子

<!-- bilingual-en:start -->
Consider the [[对角矩阵|diagonal matrix]] $A=\operatorname{diag}(1,10^{-4})$ and choose $\lambda=10^{-4}$. The inverse gains are
<!-- bilingual-en:end -->

考虑[[对角矩阵]] $A=\operatorname{diag}(1,10^{-4})$，取 $\lambda=10^{-4}$。两个反演增益为

$$g_1=\frac{1}{1+10^{-4}}=\frac{10000}{10001},\qquad
g_2=\frac{10^{-4}}{10^{-8}+10^{-4}}=\frac{10000}{10001}.$$

<!-- bilingual-en:start -->
Both are about 1, but exact inverse gains are 1 and $10^4$. The retained fractions are therefore about 1 and $10^{-4}$. For noiseless data $b=(1,10^{-4})^T$ from input $(1,1)^T$, ridge returns $(10000/10001,1/10001)^T$: the weak direction's signal is nearly erased along with its noise amplification. A better gain bound does not by itself guarantee a better recovery error for every true input.
<!-- bilingual-en:end -->

二者都约为1，但精确反演增益分别为1与 $10^4$，因此保留比例约为1和 $10^{-4}$。真实输入 $(1,1)^T$ 产生无噪声数据 $b=(1,10^{-4})^T$ 时，岭返回 $(10000/10001,1/10001)^T$：弱方向的信号与噪声放大一起几乎被消除。增益上界改善，本身不保证对每个真实输入都恢复得更准。

<!-- bilingual-en:start -->
## Parameter conventions and limiting cases
<!-- bilingual-en:end -->

## 参数化边界

<!-- bilingual-en:start -->
If a source writes the penalty as $\alpha^2\|x\|_2^2$, replace $\lambda$ here by $\alpha^2$; its denominator is $\sigma_i^2+\alpha^2$. If the fitting loss is averaged, $\|Ax-b\|_2^2/m+\lambda\|x\|_2^2$, multiplication by $m$ shows that the corresponding denominator is $\sigma_i^2+m\lambda$. Read the objective, not just the parameter name.
<!-- bilingual-en:end -->

如果来源把惩罚写成 $\alpha^2\|x\|_2^2$，本卡的 $\lambda$ 就应替换为 $\alpha^2$，分母为 $\sigma_i^2+\alpha^2$。若拟合损失取平均，即 $\|Ax-b\|_2^2/m+\lambda\|x\|_2^2$，整体乘 $m$ 后可知相应分母为 $\sigma_i^2+m\lambda$。需要看目标函数，不能只看参数名。

<!-- bilingual-en:start -->
For fixed $\lambda>0$, the gain $\sigma/(\sigma^2+\lambda)$ tends to zero as $\sigma\to0$, so it does not diverge like $1/\sigma$. Conversely, for a **fixed matrix and fixed data**, $\hat x_\lambda\to A^+b$ as $\lambda\downarrow0$: every positive direction approaches its reciprocal, while nullspace coefficients stay zero. At $\lambda=0$ itself, a rank-deficient least-squares problem has multiple minimizers; the limit selects the one of smallest norm. These are different limits with different quantities fixed.
<!-- bilingual-en:end -->

固定 $\lambda>0$ 时，$\sigma\to0$ 会使增益 $\sigma/(\sigma^2+\lambda)$ 趋于零，不像 $1/\sigma$ 那样发散。反过来，**固定矩阵和数据**，令 $\lambda\downarrow0$，有 $\hat x_\lambda\to A^+b$：每个正方向趋近倒数增益，零空间系数始终为零。但直接取 $\lambda=0$，秩亏最小二乘仍有多个最小化解，极限选出其中范数最小者。这是固定不同量的两种极限。

> [!warning] An unpenalized intercept changes the setup / 截距不受惩罚时要另作处理
> <!-- bilingual-en:start -->
> The formula assumes an isotropic penalty on **all** coordinates. If an intercept is exempt, do not apply it unchanged to the SVD of the full design including the constant column. For ordinary unweighted real regression, center the response and predictor columns, apply the formula to the penalized slopes, then recover the unpenalized intercept from the means. Other penalty matrices likewise need their own derivation. Changing predictor units also changes the meaning of the squared-coefficient penalty.
> <!-- bilingual-en:end -->
>
> 公式假定**全部**坐标受到同样的平方惩罚。若截距免罚，不能直接把含常数列的完整设计矩阵代入同一 SVD 公式。对通常未加权的实数回归，可先中心化响应与解释变量列，对受罚斜率使用公式，再由均值恢复免罚截距；其他惩罚矩阵也需重新推导。改变解释变量单位，同样会改变系数平方惩罚的含义。

<!-- bilingual-en:start -->
For nonzero $A$, a positive penalty changes the inverse operator, even if its result happens to agree with $A^+b$ for some data such as $b=0$. This is a changed optimization rule, not merely more accurate evaluation of the original pseudoinverse.
<!-- bilingual-en:end -->

对非零 $A$，正惩罚改变反演算子，即使对某些数据（如 $b=0$）碰巧仍得到 $A^+b$。这是改变优化规则，并非只是更准确地计算原伪逆。

## 来源与核验

<!-- bilingual-en:start -->
- [Hansen, *Intro to Inverse Problems*, Chapter 4, PDF pp.12–18](https://www2.imm.dtu.dk/~pcha/DIP/chap4.pdf#page=12): squared-parameter Tikhonov objective, normal equations and relative SVD filter factors. This card uses $\lambda=\alpha^2$ and derives the rectangular/complex and zero-direction cases by separating coordinates.
- [Penn State STAT508, §6.1 Ridge Regression](https://online.stat.psu.edu/stat508/Lesson06.html): centered predictors/response, the squared-slope penalty, and the unpenalized-intercept convention. Only that ridge setup is used here.
- [[03_Positive Definite Matrices and Applications#3.8.8 边界、反例与易错点|Course §3.8.8]]: local noise-amplification and regularization scope. The gain examples, limiting cases and loss-rescaling identity are direct calculations from the displayed formula.
<!-- bilingual-en:end -->

- Hansen 第4章 PDF12–18页核对平方参数的 Tikhonov 目标、正规方程与相对 SVD 滤波因子；本卡取 $\lambda=\alpha^2$，通过分方向计算处理长方形／复数矩阵及零方向。
- Penn State STAT508 第6.1节核对响应／解释变量中心化、斜率平方惩罚与截距免罚约定；此处只采用该岭设定。
- 课程第3.8.8节确定噪声放大和正则化的本地范围；增益例子、极限及损失缩放恒等式均由显示公式直接计算。
