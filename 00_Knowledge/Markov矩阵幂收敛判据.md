---
aliases:
  - "有限 Markov 矩阵的幂收敛当且仅当除一以外的全部特征值模小于一"
  - Spectral convergence criterion for Markov matrix powers
  - Markov 矩阵幂收敛判据
student_os: knowledge-atom
atom_id: LA-EIG-042
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov矩阵谱边界]]"
  - "[[Markov单位圆特征值半单]]"
  - "[[Markov矩阵必有特征值一]]"
  - "[[矩阵幂趋零判据]]"
related:
  - "[[稳态唯一不推收敛]]"
  - "[[有限链逐步收敛]]"
  - "[[周期链Cesaro平均]]"
leads_to:
  - "[[Markov幂的秩一稳态极限]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 有限 Markov 矩阵的幂收敛当且仅当除一以外的全部特征值模小于一
<!-- bilingual-en:start -->
*Powers of a finite Markov matrix converge exactly when every eigenvalue other than one has modulus less than one*
<!-- bilingual-en:end -->

> [!summary] 精确的矩阵幂判据
> 对有限行随机或列随机矩阵 $P$，序列 $P^k$ 收敛，当且仅当每个 $\lambda\ne1$ 的特征值都满足 $|\lambda|<1$。极限是到固定点空间 $E_1(P)$ 的谱投影。
> <!-- bilingual-en:start -->
> Powers of a finite stochastic matrix converge if and only if every eigenvalue other than one lies strictly inside the unit circle. The limit is the spectral projection onto the fixed-point space.
> <!-- bilingual-en:end -->

由[[Markov单位圆特征值半单]]，特征值 $1$ 没有非平凡 Jordan 块。因此空间可分成固定点空间 $E_1(P)$ 与其余广义特征空间的直和：$P$ 在前者上就是恒等映射；在后者上的矩阵幂由[[矩阵幂趋零判据]]判断。结合[[Markov矩阵谱边界]]，只要每个 $\lambda\ne1$ 都严格位于单位圆内，后半部分趋于零，于是 $P^k$ 留下到 $E_1(P)$ 的谱投影。

反过来，若还有 $\lambda\ne1$ 且 $|\lambda|=1$，对应模式会持续旋转或振荡，$P^k$ 不能收敛。这里不要求 $1$ 是单根：$P=I$ 时，$P^k=I$ 从一开始就收敛，而整个空间都是固定点空间。这个例子也说明“一般矩阵幂收敛”与“从任意初态收敛到同一个稳态”不是同一句话。

若还要求所有初态收敛到同一个稳态，必须再加入稳态唯一性；由此得到的秩一投影及其对概率初态的作用见[[Markov幂的秩一稳态极限]]。反过来，只知道稳态唯一却没有排除其他单位根，仍不能推出 $P^k$ 收敛，见[[稳态唯一不推收敛]]。
<!-- bilingual-en:start -->
Semisimplicity at one leaves a fixed-point spectral projection. Every other spectral component decays exactly when its eigenvalue lies strictly inside the unit circle. Turning that projection into a common rank-one stationary limit is a separate result requiring uniqueness.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> $P=I_2$ 的矩阵幂是否收敛？这是否意味着它有唯一稳态？
>
> **答案：** 收敛，因为 $P^k=I_2$；但稳态不唯一，每个二维概率向量都是固定点。极限是二维固定点空间上的恒等投影，而不是秩一投影。

## 来源与核验

- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/)：核对随机矩阵幂收敛的单位圆谱条件。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]]：核对稳态特征向量和次主特征值衰减。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|MIT Lecture 24 transcript]]：核对列随机例与模式分解。
