---
aliases:
  - "Gaussian Copula 由潜在多元正态向量的边际概率变换定义"
  - "Gaussian copula definition"
  - "高斯Copula"
student_os: knowledge-atom
atom_id: RM-DEP-008
atom_set: dependence-and-copulas
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Copula]]"
  - "[[相关矩阵]]"
related:
  - "[[Gaussian仿射闭包]]"
  - "[[协方差矩阵半正定性]]"
  - "[[Gaussian与t Copula尾部]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# Gaussian Copula 由潜在多元正态向量的边际概率变换定义
<!-- bilingual-en:start -->
*A Gaussian copula is defined by marginal probability transforms of a latent multivariate normal vector*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 令 $Z\sim N_d(0,R)$，其中 $R$ 是合法相关矩阵，并令 $U_i=\Phi(Z_i)$。向量 $U=(U_1,\ldots,U_d)$ 的 copula 就是参数为 $R$ 的 Gaussian Copula。
> <!-- bilingual-en:start -->
> If $Z\sim N_d(0,R)$ and $U_i=\Phi(Z_i)$, the copula of $U$ is the Gaussian copula with parameter matrix $R$.
> <!-- bilingual-en:end -->

其分布函数可写成

$$
C_R^G(u_1,\ldots,u_d)
=\Phi_R\!\left(\Phi^{-1}(u_1),\ldots,\Phi^{-1}(u_d)\right),
$$

其中 $\Phi_R$ 是相关矩阵为 $R$ 的标准多元正态 CDF。

## 参数与边际

$R$ 必须对角为 1 且半正定；若要使用通常的非退化密度与似然，通常还需要正定。$R$ 描述潜在 Gaussian 坐标的相关结构，不等于任意原始边际数据的 Pearson 相关矩阵。

Sklar 组合可以把这个 copula 与任意连续边际 $F_i$ 配合，所以“使用 Gaussian copula”不等于“原变量必须服从正态分布”。若 $R=I$，Gaussian copula 退化为乘积 copula。

它的渐近尾部边界见 [[Gaussian与t Copula尾部]]。

> [!question]- 自检
> 给收益使用 Gaussian copula，是否意味着每只资产收益都被设成正态分布？
>
> **答案：** 不意味着。边际分布可另行指定；Gaussian 只描述潜在概率尺度上的依赖结构。

## 来源与核验

- Stefano Demarta and Alexander J. McNeil (2005), [“The t Copula and Related Copulas”](https://doi.org/10.1111/j.1751-5823.2005.tb00254.x)：核对椭圆 copula 的概率变换定义与 Gaussian 特例。
- Roger B. Nelsen, *An Introduction to Copulas*, 2nd ed.：[出版社页面](https://link.springer.com/book/10.1007/0-387-28678-0)；核对 copula 与边际的分离。
