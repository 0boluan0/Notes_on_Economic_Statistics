---
aliases:
- Truncated SVD inversion versus ridge regularization
- TSVD versus Tikhonov regularization
student_os: knowledge-atom
atom_id: LA-PINV-008
atom_set: pseudoinverse-one-sided-inverses
atom_type: method-distinction
status: source-checked
requires:
- '[[截断SVD反演]]'
- '[[岭正则化]]'
- '[[岭的SVD滤波公式]]'
part_of:
- '[[广义逆与最小范数解.canvas]]'
related:
- '[[小奇异值放大噪声]]'
- '[[数值秩]]'
- '[[正则化判别]]'
---

<!-- bilingual-en:start -->
# Truncated SVD versus ridge regularization
<!-- bilingual-en:end -->

*截断 SVD 与岭正则化*

<!-- bilingual-en:start -->
A weakly measured direction creates a choice: invert it completely, ignore it, or keep only part of its contribution. [[截断SVD反演|Truncated SVD inversion]] (TSVD) chooses between complete inversion and deletion. [[岭正则化|Ridge regularization]] instead gives each positive singular direction a nonzero, reduced gain relative to exact inversion. The difference concerns how information is retained, not simply which program solves a matrix equation.
<!-- bilingual-en:end -->

一个测量很弱的方向带来三种选择：完整反演、忽略它，或只保留部分贡献。[[截断SVD反演]]（TSVD）在完整反演与删除之间选择；[[岭正则化]]则对每个正奇异方向使用非零、但相对精确反演减小的增益。区别在于如何保留信息，而不仅是用哪段程序解矩阵方程。

> [!example] Keep, discard or shrink / 保留、删除或收缩
> <!-- bilingual-en:start -->
> Take $A=\operatorname{diag}(1,0.1)$ and $b=(1,0.1)^T$. The [[对角矩阵|diagonal matrix]] observes the first input directly but reduces the second by a factor of ten. Exact inversion gives $(1,1)^T$. TSVD with cutoff $\tau=0.2$ keeps only the first direction and gives $(1,0)^T$. Ridge with penalty $\lambda=0.01$ gives $(100/101,1/2)^T$: it retains part of the weak direction while also slightly shrinking the strong one.
> <!-- bilingual-en:end -->
>
> 取 $A=\operatorname{diag}(1,0.1)$、$b=(1,0.1)^T$。这个[[对角矩阵]]直接观测第一输入，却把第二输入缩小为十分之一。精确反演得到 $(1,1)^T$；阈值 $\tau=0.2$ 的 TSVD 只保留第一方向，得到 $(1,0)^T$；惩罚系数 $\lambda=0.01$ 的岭得到 $(100/101,1/2)^T$，保留弱方向的一部分，同时也略微收缩强方向。

<!-- bilingual-en:start -->
## Compare the inverse gains and the retained fractions
<!-- bilingual-en:end -->

## 滤波规则的直接对照

<!-- bilingual-en:start -->
Fix a real or complex $m\times n$ matrix $A$, $m,n\ge1$, and an output vector $b$. For positive [[矩阵秩|rank]] $r$, its [[紧SVD|compact SVD]] is $A=U_r\Sigma_rV_r^*$. The positive [[奇异值|singular values]] $\sigma_i$ measure directional stretching; $v_i$ and $u_i$ are paired unit input and output directions satisfying $Av_i=\sigma_i u_i$. The star means conjugate transpose. The scalar $c_i=u_i^*b$ is the measured component of $b$ in output direction $u_i$. Both estimates have the form
<!-- bilingual-en:end -->

固定实或复 $m\times n$ 矩阵 $A$，$m,n\ge1$，以及输出向量 $b$。当[[矩阵秩|秩]] $r>0$ 时，[[紧SVD]]为 $A=U_r\Sigma_rV_r^*$。正[[奇异值]] $\sigma_i$ 度量方向伸缩；$v_i$、$u_i$ 是配对的单位输入、输出方向，满足 $Av_i=\sigma_i u_i$。星号表示共轭转置；标量 $c_i=u_i^*b$ 是 $b$ 沿输出方向 $u_i$ 的测量分量。两种估计都可写为

$$\hat x=\sum_{i=1}^r g_i c_i v_i.$$

<!-- bilingual-en:start -->
The inverse gain $g_i$ converts an output coordinate into an input coefficient. With a declared cutoff $\tau\ge0$ and the all-coordinate ridge penalty $\lambda\|x\|_2^2$, $\lambda>0$, the gains are
<!-- bilingual-en:end -->

反演增益 $g_i$ 把输出坐标转换成输入系数。声明阈值 $\tau\ge0$，并使用全部坐标受罚的岭惩罚 $\lambda\|x\|_2^2$、$\lambda>0$ 时，增益为

$$
g_i^{\mathrm{TSVD}}(\tau)=\begin{cases}1/\sigma_i,&\sigma_i>\tau,\\0,&\sigma_i\le\tau,\end{cases}
\qquad
g_i^{\mathrm{ridge}}(\lambda)=\frac{\sigma_i}{\sigma_i^2+\lambda}.
$$

<!-- bilingual-en:start -->
**Equality belongs to the discarded side.** In the opening example, even $\tau=0.1$ removes the second direction. For a positive cutoff, crossing it changes the TSVD rule abruptly; ridge varies smoothly. The [[岭的SVD滤波公式|ridge formula and its derivation]] assume every coefficient is penalized. An exempt intercept must first be handled as described in the ridge definition; one cannot apply this gain unchanged to the full design including an unpenalized intercept.
<!-- bilingual-en:end -->

**等于阈值时也删除。** 开头的例子即使取 $\tau=0.1$，仍删除第二方向。对于正阈值，跨过阈值会让 TSVD 规则突变，岭则平滑变化。[[岭的SVD滤波公式|岭公式及其推导]]假定全部系数受罚；截距免罚时，要先按岭定义中的办法处理，不能直接对含免罚截距的完整设计套用同一增益。

<!-- bilingual-en:start -->
To compare with [[伪逆SVD公式|exact pseudoinversion]], divide each gain by $1/\sigma_i$. This gives the relative retained fraction
<!-- bilingual-en:end -->

若要与[[伪逆SVD公式|精确伪逆]]比较，把每个增益除以 $1/\sigma_i$，得到相对保留比例

$$f_i^{\mathrm{TSVD}}=\begin{cases}1,&\sigma_i>\tau,\\0,&\sigma_i\le\tau,\end{cases}
\qquad f_i^{\mathrm{ridge}}=\frac{\sigma_i^2}{\sigma_i^2+\lambda}\in(0,1).$$

<!-- bilingual-en:start -->
TSVD keeps all or none of each positive direction; ridge keeps a fraction strictly between zero and one. For the weak direction in the example, exact gain is 10, ridge gain is 5, and the retained fraction is $1/2$. “Shrinkage” does not mean the inverse gain is below 1. Also, a nonzero gain does not guarantee a nonzero estimated coefficient: the measured component $c_i$ may itself be zero. Zero singular directions receive zero in both methods; if $A=0$, both answers are zero.
<!-- bilingual-en:end -->

TSVD 对每个正方向全留或全删，岭则保留严格介于零与一之间的比例。例子中的弱方向，精确增益为10、岭增益为5，而保留比例为 $1/2$。“收缩”并不表示反演增益小于1。非零增益也不保证估计系数非零，因为测量分量 $c_i$ 本身可能为零。两种方法都把零奇异方向取为零；若 $A=0$，二者都返回零。

<!-- bilingual-en:start -->
## A cutoff and a penalty weight make different choices
<!-- bilingual-en:end -->

## 两个参数回答不同问题

<!-- bilingual-en:start -->
The cutoff $\tau$ selects a subspace: retain exactly the directions with $\sigma_i>\tau$. Their count is the [[数值秩|numerical rank under this SVD convention]]. TSVD minimizes the original fitting error over that retained input subspace; equivalently it applies the pseudoinverse of the matrix with discarded singular values replaced by zero. It need not minimize the unrestricted original least-squares problem.
<!-- bilingual-en:end -->

阈值 $\tau$ 选择一个子空间：恰好保留 $\sigma_i>\tau$ 的方向，其个数是[[数值秩|该 SVD 口径下的数值秩]]。TSVD 在保留的输入子空间中最小化原拟合误差；等价地，它使用把已删除奇异值换成零后的矩阵伪逆。它不必最小化不受限制的原始最小二乘问题。

<!-- bilingual-en:start -->
Ridge permits all input directions but charges for coefficient length. Its $\lambda$ sets a smooth degree of retention, not a hard count of directions. A positive singular value equal to $\sqrt\lambda$ has retained fraction $1/2$; this gives a useful scale comparison with a cutoff, but a 0-or-1 decision cannot reproduce the strictly intermediate ridge fraction on a positive direction. Larger $\lambda$ yields smaller fractions when $A$ is fixed.
<!-- bilingual-en:end -->

岭允许全部输入方向，但对系数长度收费。$\lambda$ 控制平滑的保留程度，而不是硬性计数方向。正奇异值恰为 $\sqrt\lambda$ 时，保留比例为 $1/2$；这可帮助比较它与阈值的尺度，但0或1的二分选择不能复制正方向上严格介于两者之间的岭保留比例。固定 $A$，增大 $\lambda$ 会减小保留比例。

<!-- bilingual-en:start -->
Read the objective when translating parameters. A source penalty $\alpha^2\|x\|_2^2$ corresponds to $\lambda=\alpha^2$ here, and its halfway scale is $\sigma_i=\alpha$ for $\alpha>0$. An averaged fitting loss $\|Ax-b\|_2^2/m$ with penalty weight $\lambda$ corresponds to weight $m\lambda$ in the sum-loss formula. A singular-value cutoff and a penalty weight therefore do not have the same scale. Predictor units and the rule used to choose the parameter must also be stated.
<!-- bilingual-en:end -->

换算参数时应看目标函数。来源若用 $\alpha^2\|x\|_2^2$，就对应本卡的 $\lambda=\alpha^2$，当 $\alpha>0$ 时半保留尺度为 $\sigma_i=\alpha$。若拟合损失为平均值 $\|Ax-b\|_2^2/m$、惩罚权重为 $\lambda$，换到求和损失公式时权重是 $m\lambda$。因此奇异值阈值与惩罚权重并非同一尺度；解释变量单位和选参规则也需说明。

<!-- bilingual-en:start -->
## Changed recovery rules do not guarantee better recovery
<!-- bilingual-en:end -->

## 与精确伪逆的共同边界

<!-- bilingual-en:start -->
If TSVD discards no positive singular value, it equals the original exact pseudoinverse. If it discards even one positive direction, the operator changes. For nonzero $A$, ridge with finite $\lambda>0$ changes every positive direction's gain and hence changes the operator. Different operators may still return the same answer for particular data, such as $b=0$. These distinctions concern the rule, not whether a single computed vector happens to coincide.
<!-- bilingual-en:end -->

若 TSVD 没有删除任何正奇异值，它就等于原矩阵的精确伪逆；只要删除一个正方向，算子就改变。对非零 $A$，有限的 $\lambda>0$ 会让岭改变每个正方向的增益，因此改变算子。不过，不同算子对特定数据仍可能给出相同答案，例如 $b=0$。这里区分的是规则，而不是某一次计算结果是否碰巧一致。

> [!warning] The same data can contain signal or noise / 同一份数据可能包含信号或噪声
> <!-- bilingual-en:start -->
> Return to $A=\operatorname{diag}(1,0.1)$ and $b=(1,0.1)^T$. If the true input is $(1,1)^T$ and measurement is exact, the pseudoinverse recovers it perfectly; TSVD loses the second signal and ridge shrinks it. But if the true input is $(1,0)^T$ and the data contain error $(0,0.1)^T$, the same TSVD output is now exact. The matrix and observed vector alone do not reveal which story generated the weak component.
> <!-- bilingual-en:end -->
>
> 仍取 $A=\operatorname{diag}(1,0.1)$、$b=(1,0.1)^T$。若真实输入为 $(1,1)^T$、测量完全准确，伪逆恢复无误，TSVD 丢失第二方向信号，岭则收缩该信号。若真实输入为 $(1,0)^T$、数据含误差 $(0,0.1)^T$，同一个 TSVD 输出却恰好正确。单凭矩阵与观测向量，无法判断弱分量来自哪种情况。

<!-- bilingual-en:start -->
Suppressing a weak direction can reduce [[小奇异值放大噪声|noise amplification]], but may remove useful signal at the same time. With a specified random-noise model and target, compare [[MSE偏差方差分解|bias and variance through their combined mean squared error]]; without that model, neither rule has a universal statistical advantage. Parameter choice must reflect the noise scale, desired output and a suitable validation design. The two methods express different compromises, not a guarantee that changing the original problem improves every solution.
<!-- bilingual-en:end -->

抑制弱方向可能降低[[小奇异值放大噪声|噪声放大]]，也可能同时删除有用信号。有明确随机噪声模型和目标时，应通过[[MSE偏差方差分解|偏差与方差共同构成的均方误差]]比较；缺少该模型，不能给任何一种规则宣称普遍统计优势。选参必须考虑噪声尺度、所需输出及适当验证设计。两种方法表达不同取舍，而非保证改变原问题就能改善每个解。

## 来源与核验

<!-- bilingual-en:start -->
- [Hansen, *Intro to Inverse Problems*, Chapter 4, PDF pp.5–8 and 12–18](https://www2.imm.dtu.dk/~pcha/DIP/chap4.pdf#page=5): filtered SVD expansion, truncated-matrix solution, Tikhonov objective and relative filter factors. The source's squared parameter is translated explicitly; the zero/rectangular/complex cases follow from the displayed directional formulas.
- [LAPACK DGELSD, `RCOND` and `RANK` documentation](https://netlib.org/lapack/lapack_routine/dgelsd.f): for the ordinary relative-cutoff convention $0<\mathrm{RCOND}<1$, values at or below the declared cutoff are discarded; retained values are strictly greater. Algorithm-specific defaults are handled in [[数值秩|numerical rank]].
- [[03_Positive Definite Matrices and Applications#3.8.8 边界、反例与易错点|Course §3.8.8]]: local regularization and noise-amplification scope. The two explanations of the same data, parameter conversions and exact-pseudoinverse boundary are checked directly.
<!-- bilingual-en:end -->

- Hansen 第4章 PDF5–8、12–18页核对滤波 SVD 展开、截断矩阵解、Tikhonov 目标及相对保留因子；平方参数已显式换算，零矩阵／长方形／复数情况由显示的方向公式处理。
- LAPACK DGELSD 的 `RCOND` 与 `RANK` 说明核对普通相对阈值 $0<\mathrm{RCOND}<1$ 的口径：等于或低于声明阈值时舍去，严格大于时保留；算法特有默认值由[[数值秩]]负责。
- 课程第3.8.8节确定正则化与噪声放大范围；同一数据的两种解释、参数换算和精确伪逆边界均直接核算。
