---
aliases:
  - "t Copula 由共享随机尺度的多元 Student t 向量定义"
  - "t copula definition"
  - "t-Copula"
student_os: knowledge-atom
atom_id: RM-DEP-009
atom_set: dependence-and-copulas
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Copula]]"
  - "[[相关矩阵]]"
related:
  - "[[Gaussian Copula]]"
  - "[[Gaussian与t Copula尾部]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# t Copula 由共享随机尺度的多元 Student t 向量定义
<!-- bilingual-en:start -->
*A t copula is defined by a multivariate Student-t vector with a shared random scale*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 令 $Z\sim N_d(0,R)$、$W\sim\chi^2_\nu$ 相互独立，并让所有坐标共享同一个 $W$：
> $$T=\frac{Z}{\sqrt{W/\nu}},\qquad U_i=t_\nu(T_i).$$
> 向量 $U$ 的 copula 就是自由度为 $\nu$、参数矩阵为 $R$ 的 t Copula。
> <!-- bilingual-en:start -->
> Let $Z\sim N_d(0,R)$ and an independent $W\sim\chi^2_\nu$, with the same $W$ shared by every coordinate. The copula of $U_i=t_\nu(Z_i/\sqrt{W/\nu})$ is the t copula.
> <!-- bilingual-en:end -->

等价地，若 $t_{\nu,R}$ 是标准多元 t CDF，

$$
C_{\nu,R}^{t}(u_1,\ldots,u_d)
=t_{\nu,R}\!\left(t_\nu^{-1}(u_1),\ldots,t_\nu^{-1}(u_d)\right).
$$

## 共享尺度为什么关键

共同的 $W$ 会让所有坐标在同一时刻一起被放大，从而形成联合极端。只把某一个共同因子换成 t 分布、却让其余特质项保持独立正态，并不自动得到标准 t Copula。

$R$ 必须对角为 1 且半正定；使用通常的非退化密度时通常需要正定。对任意 $\nu>0$，$R$ 都是椭圆分布的依赖参数；只有 $\nu>2$ 时潜在 t 向量的 Pearson 相关存在并等于 $R$。即使 $R=I$，有限 $\nu$ 下共享尺度仍使坐标不独立。

具体尾部依赖公式见 [[Gaussian与t Copula尾部]]。

> [!question]- 自检
> 在有限自由度下令 $R=I$，能否因此断言 t Copula 是独立 copula？
>
> **答案：** 不能。共享随机尺度仍会让坐标共同出现极端值。

## 来源与核验

- Stefano Demarta and Alexander J. McNeil (2005), [“The t Copula and Related Copulas”](https://doi.org/10.1111/j.1751-5823.2005.tb00254.x)：核对多元 t、共同尺度混合构造及参数解释。
- 作者逐式复核日：2026-09-01；$\nu>0$ 的模型定义与 $\nu>2$ 的 Pearson 相关边界已分开表述。
