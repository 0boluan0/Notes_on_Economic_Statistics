---
aliases:
- Four classical MANOVA statistics
- MANOVA criteria
- Wilks、Pillai、Hotelling-Lawley 与 Roy
student_os: knowledge-atom
atom_id: STAT-MAN-003
atom_set: manova
atom_type: definition
status: source-checked
part_of:
- '[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]'
requires:
- '[[MANOVA SSCP分解]]'
- '[[误差SSP秩边界]]'
implies:
- '[[MANOVA检验校准]]'
related:
- '[[Pillai稳健性边界]]'
- '[[多响应联合显著边界]]'
- '[[响应尺度与线性变换]]'
---

<!-- bilingual-en:start -->
# Four classical MANOVA statistics
<!-- bilingual-en:end -->

*MANOVA 的四种经典统计量*

<!-- bilingual-en:start -->
A MANOVA calculation produces a matrix of variation attributed to the tested effect and a matrix of variation left as error. To summarize how large the effect looks **relative to the error**, we need a scalar. Wilks' lambda, Pillai's trace, the Hotelling–Lawley trace and Roy's greatest root are four different ways to make this summary. They use the same directional comparisons but combine them differently.
<!-- bilingual-en:end -->

MANOVA 计算会产生一张归于待检验效应的变动矩阵，以及一张剩余误差的变动矩阵。要用一个数概括效应**相对于误差**有多大，可以使用 Wilks' lambda、Pillai trace、Hotelling–Lawley trace 或 Roy 最大根。它们使用同一组方向比较，只是汇总方式不同。

<!-- bilingual-en:start -->
Write $H$ for the hypothesis SSCP and $E$ for the error SSCP. In [[MANOVA SSCP分解|one-way MANOVA]], these are between-group and within-group squared-deviation matrices, with $T=E+H$; the one-way null says that all population mean vectors are equal. For the classical formulas here, both are real $p\times p$ matrices with $p\ge1$, $H$ is [[半正定矩阵|positive semidefinite]], and $E$ is [[正定矩阵|positive definite]]. Positive definiteness means every nonzero response direction has positive error variation, so $E^{-1}$ exists. Use the **unscaled SSCP matrices**, not matrices separately divided by their degrees of freedom.
<!-- bilingual-en:end -->

用 $H$ 表示假设 SSCP，$E$ 表示误差 SSCP。在[[MANOVA SSCP分解|单因素 MANOVA]]中，它们分别是组间与组内平方偏差矩阵，满足 $T=E+H$；这个单因素原假设说的是各组总体均值向量全部相等。本卡经典公式要求二者为实 $p\times p$ 矩阵、$p\ge1$，$H$ [[半正定矩阵|半正定]]，$E$ [[正定矩阵|正定]]。正定性意味着每个非零响应方向都具有正的误差变动，因此 $E^{-1}$ 存在。这里使用**未按自由度缩放的 SSCP**，不能把分别除过自由度的矩阵直接代入而仍沿用同一定义。

<!-- bilingual-en:start -->
## First compare directions on a common error scale
<!-- bilingual-en:end -->

*先把不同方向放到共同误差尺度上*

<!-- bilingual-en:start -->
With one response, the comparison would simply be $H/E$. With several responses, we remove the scale and correlation recorded by $E$. Its [[正定逆平方根|inverse square root]] $E^{-1/2}$ is the symmetric positive-definite matrix satisfying $E^{-1/2}EE^{-1/2}=I$. In these transformed coordinates, error variation is the identity matrix, and the hypothesis matrix is
$$B=E^{-1/2}HE^{-1/2}.$$
Because $B$ is symmetric and positive semidefinite, its [[特征对|eigenvalues]] are real and nonnegative. List them in decreasing order, including repeated and zero values:
$$\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_p\ge0.$$
These are the MANOVA roots. A large root describes a direction with substantial hypothesis variation relative to error variation.
<!-- bilingual-en:end -->

只有一个响应时，可以直接比较 $H/E$。有多个响应时，先消去 $E$ 记录的尺度与相关结构。[[正定逆平方根|逆平方根]] $E^{-1/2}$ 是满足 $E^{-1/2}EE^{-1/2}=I$ 的对称正定矩阵。换到这种坐标后，误差矩阵成为单位矩阵，假设矩阵成为 $B$。$B$ 对称且半正定，所以[[特征对|特征值]]都是非负实数；按从大到小排列，重复值与零值也全部列入。这些就是 MANOVA 特征根：根越大，对应方向上的假设变动相对于误差越大。

<!-- bilingual-en:start -->
The frequently used matrix $E^{-1}H$ need not itself be symmetric, but it has exactly the same roots, since
$$E^{-1}H=E^{-1/2}B E^{1/2}.$$
Here $E^{1/2}=(E^{-1/2})^{-1}$. This is a [[相似矩阵|similarity transformation]], which preserves eigenvalues. At most $\operatorname{rank}(H)$ roots can be nonzero. In a one-way design with $g$ nonempty groups, there are at most $\min(p,g-1)$ such roots. They are not the eigenvalues of $H$ alone: $E$ provides the comparison scale.
<!-- bilingual-en:end -->

这里 $E^{1/2}=(E^{-1/2})^{-1}$。常见写法 $E^{-1}H$ 本身未必对称，但通过上述[[相似矩阵|相似变换]]与 $B$ 具有完全相同的特征根。非零根至多有 $\operatorname{rank}(H)$ 个；单因素设计有 $g$ 个非空组时，至多为 $\min(p,g-1)$ 个。它们不是单独 $H$ 的特征值，因为比较尺度由 $E$ 提供。

> [!important] Four summaries of the same roots · 同一组根的四种汇总
> <!-- bilingual-en:start -->
> The [[行列式|determinant]] is written $\det$, and the trace $\operatorname{tr}$ is the sum of a matrix's diagonal entries. The criteria are
>
> | Criterion | Root formula | Matrix formula | More evidence against the null |
> |---|---|---|---|
> | Wilks' lambda | $\Lambda=\prod_{j=1}^p(1+\lambda_j)^{-1}$ | $\det(E)/\det(E+H)$ | Smaller $\Lambda$ |
> | Pillai's trace | $V=\sum_{j=1}^p\lambda_j/(1+\lambda_j)$ | $\operatorname{tr}\{H(E+H)^{-1}\}$ | Larger $V$ |
> | Hotelling–Lawley trace | $U=\sum_{j=1}^p\lambda_j$ | $\operatorname{tr}(E^{-1}H)$ | Larger $U$ |
> | Roy's greatest root | $\Theta=\lambda_1$ | Largest eigenvalue of $E^{-1}H$ | Larger $\Theta$ |
> <!-- bilingual-en:end -->
>
> $\det$ 表示[[行列式]]，$\operatorname{tr}$（迹）表示对角元之和。Wilks 把每个 $1+\lambda_j$ 的倒数相乘，**越小**越偏向反对原假设；其余三个都是**越大**越偏向反对。Pillai 先把每个根变成 $\lambda_j/(1+\lambda_j)$ 再相加；Hotelling–Lawley 直接相加；Roy 只保留最大根。

<!-- bilingual-en:start -->
Taking determinants in $E+H=E^{1/2}(I+B)E^{1/2}$ yields the determinant ratio. The [[迹与特征值总和|trace–eigenvalue identity]] yields the two sums. In particular, normalizing by total rather than error variation gives roots
$$\eta_j=\frac{\lambda_j}{1+\lambda_j}\in[0,1)$$
of $(E+H)^{-1}H$. These are a different root convention; they recover $\lambda_j=\eta_j/(1-\eta_j)$. Roy's $\Theta$ in this card uses the error-normalized root $\lambda_1$, not the bounded root $\eta_1$.
<!-- bilingual-en:end -->

从 $E+H=E^{1/2}(I+B)E^{1/2}$ 的行列式可得 Wilks 的比值，再由[[迹与特征值总和|迹等于特征值之和]]得到两个求和式。特别地，若用总变动而非误差变动归一化，$(E+H)^{-1}H$ 的根是 $\eta_j=\lambda_j/(1+\lambda_j)\in[0,1)$。这是一种不同记号口径，可以用 $\lambda_j=\eta_j/(1-\eta_j)$ 换回。本卡 Roy 的 $\Theta$ 使用误差归一化的 $\lambda_1$，不是有界的 $\eta_1$。

<!-- bilingual-en:start -->
Thus $0<\Lambda\le1$, $0\le V<p$, and $U,\Theta\ge0$. If $H$ is the zero matrix, the values are $1,0,0,0$. This describes **no observed hypothesis variation**. It is not what the population null requires in every sample: equal population means can still produce different sample means through sampling noise.
<!-- bilingual-en:end -->

因此 $0<\Lambda\le1$、$0\le V<p$，而 $U,\Theta\ge0$。若 $H$ 为零矩阵，四者为 $1,0,0,0$，表示**没有观测到的假设变动**。总体原假设并不要求每次样本都如此：总体均值相等时，抽样噪声仍会使样本均值不同。

> [!example] Continue the six-observation example · 接着六观测例子计算
> <!-- bilingual-en:start -->
> The [[MANOVA SSCP分解|worked decomposition]] gave
> $$E=H=\begin{bmatrix}4&2\\2&4\end{bmatrix},\qquad T=\begin{bmatrix}8&4\\4&8\end{bmatrix}.$$
> Since $E^{-1}H=I_2$, the two roots are $1,1$. Therefore
> $$\Lambda=\tfrac14,\qquad V=1,\qquad U=2,\qquad\Theta=1.$$
> You can check Wilks directly: $\det E=12$, $\det T=48$, so the ratio is $1/4$. Pillai is $\operatorname{tr}(I_2/2)=1$. These four numbers are statistics; none of them is a p-value.
> <!-- bilingual-en:end -->
>
> [[MANOVA SSCP分解|前一张卡的例子]]给出上述 $E,H,T$。因为 $E^{-1}H=I_2$，根为 $1,1$，所以四个值依次为 $1/4,1,2,1$。还可直接核对 $\det E=12$、$\det T=48$，以及 Pillai 的 $\operatorname{tr}(I_2/2)=1$。这些都是统计量，不是 p 值。

<!-- bilingual-en:start -->
## One strong direction or several moderate directions?
<!-- bilingual-en:end -->

*一个很强方向，还是多个中等方向？*

<!-- bilingual-en:start -->
For a nonzero response coefficient vector $a$, the ratio $a^THa/(a^TEa)$ compares hypothesis and error variation in that linear combination. Setting $z=E^{1/2}a$ turns it into $z^TBz/(z^Tz)$, whose maximum is $\lambda_1$: expand $z$ in an orthonormal eigenbasis and the ratio is a weighted average of the roots. Roy therefore keeps the strongest direction and ignores the remaining roots. The other criteria combine several directions, but with different weights.
<!-- bilingual-en:end -->

对非零响应系数向量 $a$，$a^THa/(a^TEa)$ 比较这一线性组合里的假设变动与误差变动。令 $z=E^{1/2}a$，比值成为 $z^TBz/(z^Tz)$；把 $z$ 展开到标准正交特征基后，该比值就是各根的加权平均，因此最大值为 $\lambda_1$。Roy 保留最强方向、忽略其他根；另外三者会以不同方式汇总多个方向。

<!-- bilingual-en:start -->
For example, compare root patterns $(3,0)$ and $(1,1)$. Both give Wilks $\Lambda=1/4$. The first gives Roy $3$ and Hotelling–Lawley $3$, versus $1$ and $2$ for the second. Pillai instead gives $3/4$ versus $1$: its bounded contributions give more relative weight to the spread across two directions. These are not four interchangeable numerical scales or a universal ordering of the same datasets.
<!-- bilingual-en:end -->

例如比较根结构 $(3,0)$ 与 $(1,1)$：两者的 Wilks 都是 $1/4$；前者 Roy 为 $3$、Hotelling–Lawley 为 $3$，后者则为 $1,2$。Pillai 却分别为 $3/4,1$，因为每个方向的贡献有上界，它对两个方向分散出现的信号给予不同的相对权重。因此四者不是可以直接互换的数值尺度，也未必给不同数据集排出相同顺序。

<!-- bilingual-en:start -->
With only one nonzero root, all four are monotone functions of that root, with Wilks decreasing and the others increasing. They then carry the same ordering information. This still does not turn a root into an F statistic. For $p=1$ in a one-way design with $g\ge2$ and $N>g$, the usual univariate F ratio is $[H/(g-1)]/[E/(N-g)]$, which includes degrees-of-freedom factors absent from the root $H/E$.
<!-- bilingual-en:end -->

只有一个非零根时，四者都是该根的单调函数：Wilks 递减，另外三个递增，因而提供相同的排序信息。但根仍不自动等于 F 统计量。例如 $p=1$、$g\ge2$、$N>g$ 的单因素设计，普通单变量 F 比是 $[H/(g-1)]/[E/(N-g)]$，其中有特征根 $H/E$ 没有的自由度因子。

<!-- bilingual-en:start -->
## Calculating a criterion is not calibrating a test
<!-- bilingual-en:end -->

*算出统计量，还没有完成检验校准*

<!-- bilingual-en:start -->
The matrix formulas require the stated algebraic conditions. A rejection threshold or p-value additionally needs a sampling model, hypothesis dimension and residual degrees of freedom; see [[MANOVA检验校准|MANOVA calibration]]. Software often reports an approximate F transformation, whose name does not make the raw statistic F-distributed. Choose the criterion before inspecting competing p-values. Any comparative robustness argument for [[Pillai稳健性边界|Pillai's trace]] has its own assumptions and is separate from the definitions here.
<!-- bilingual-en:end -->

矩阵公式只要求上述代数条件；拒绝阈值或 p 值还需要抽样模型、假设维数和残差自由度，见[[MANOVA检验校准]]。软件常报告近似 F 转换，名称里有 F 并不表示原始统计量本身服从 F 分布。应在比较多个 p 值之前选定统计量。[[Pillai稳健性边界|Pillai 的比较稳健性]]也有自己的适用条件，独立于本卡定义。

<!-- bilingual-en:start -->
If $E$ is singular, $E^{-1}H$ is undefined. Some other displayed expressions may still be algebraically calculable when $E+H$ is invertible, but this does not restore the common error-normalized root construction or its classical reference distributions. A pseudoinverse, response reduction or regularization needs an explicit new definition and justification; consult the [[误差SSP秩边界|error-SSCP rank boundary]]. A joint rejection also does not establish that every individual response differs, as explained in [[多响应联合显著边界|the scope of a joint significance conclusion]].
<!-- bilingual-en:end -->

若 $E$ 奇异，$E^{-1}H$ 没有定义。即使 $E+H$ 可逆、某些其他表达式仍能算出，也不能据此恢复统一的误差归一化根构造或经典参考分布。换成伪逆、减少响应或正则化，都需要明确的新定义和论证，见[[误差SSP秩边界]]。联合拒绝也不证明每一个响应都不同，见[[多响应联合显著边界]]。

## 来源与核验

<!-- bilingual-en:start -->
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm): the four criteria as functions of $E^{-1}H$ or $(E+H)^{-1}H$, the matrix formulas, Roy's error-normalized root convention, and the distinction from F-based p-value calculation.
- [Penn State STAT505, Lesson8 §8.3](https://online.stat.psu.edu/stat505/Lesson08): determinant/trace/root definitions and the opposite rejection direction for Wilks. Section8.1 gives the scalar mean-square F ratio; §8.2 supplies $H,E$ and degrees of freedom.
- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.5.4. Wilks Lambda 检验|Course §1.5.4]]: Wilks' determinant ratio and course notation. The symmetric-root argument, root-convention conversion, ranges and numerical examples are verified directly above using spectral decomposition, determinants and traces.
<!-- bilingual-en:end -->

- SAS 核对四者的矩阵定义、两类根记号、Roy 使用误差归一化根，以及统计量与 F 型 p 值计算的区别。
- Penn State 第8课8.3节核对定义与 Wilks 相反的拒绝方向；8.1节核对单变量均方 F 比，8.2节核对 $H,E$ 与自由度。
- 课程第1.5.4节核对 Wilks 比值及课程记号；对称化后的根、记号转换、取值范围与数值例子在正文通过谱分解、行列式和迹直接核验。
