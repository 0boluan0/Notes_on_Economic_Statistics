---
aliases:
  - "经典 MANOVA 校准要求独立多元正态误差与共同组内协方差"
  - "classical MANOVA assumptions"
  - "MANOVA assumption boundary"
student_os: knowledge-atom
atom_id: STAT-MAN-005
atom_set: manova
atom_type: assumption-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
requires:
  - "[[MANOVA]]"
  - "[[矩阵正态误差结构]]"
  - "[[Wishart抽样假设]]"
related:
  - "[[误差SSP秩边界]]"
  - "[[相关行不容独立推断]]"
  - "[[多响应联合显著边界]]"
  - "[[多响应因果边界]]"
---

# 经典 MANOVA 校准要求独立多元正态误差与共同组内协方差
<!-- bilingual-en:start -->
*Classical MANOVA calibration requires independent multivariate-normal errors with a common within-group covariance*
<!-- bilingual-en:end -->

> [!summary] 原子假设边界
> 单因素经典模型通常写成
> $$Y_{ij}\overset{ind}{\sim}N_p(\mu_i,\Sigma),\qquad \Sigma\succ0,$$
> 其中不同观测单位独立，所有组共享同一个组内协方差矩阵 $\Sigma$，均值允许随组变化。要使用依赖 $E^{-1}$ 的全响应经典检验，还需设计中的目标效应可估、残差自由度为正，并且误差 SSCP 在有效响应空间满秩。
> <!-- bilingual-en:start -->
> The classical one-way model assumes independent $N_p(\mu_i,\Sigma)$ observations with one common within-group covariance. Estimability and sufficient residual rank are separate prerequisites for the usual inverse-based tests.
> <!-- bilingual-en:end -->

这些条件各自处理不同问题：

- **独立观测单位：** 同一人重复测量、班级内聚类、空间或时序依赖会破坏行独立；把每次测量当作新受试者并不会被 MANOVA 自动修复，见 [[相关行不容独立推断]]；
- **联合多元正态：** 每个响应边际看起来近似正态并不足以证明联合 Gaussian；经典小样本 Wishart 和参考分布需要更强的联合模型，见 [[Wishart抽样假设]]；
- **共同协方差：** $\Sigma_1=\cdots=\Sigma_g$ 不只是各变量方差相同，还包括组内响应协方差相同；严重异质且组样本量不平衡时，经典校准尤其不能无条件沿用；
- **秩与可估性：** $N-g\ge p$ 只是连续 Gaussian 模型下 $E$ 可能可逆的维数门槛；实际响应共线或不可估设计对比仍会失败，见 [[误差SSP秩边界]]。

残差直方图、散点图和异常值检查可以发现与模型冲突的证据，却不能由“没有明显问题”证明总体假设。Box's $M$ 一类协方差齐性检验本身也依赖分布并受样本量影响；未拒绝不等于多个总体协方差已经被证明相同。若条件不合适，应按真实设计考虑变换、显式异质协方差模型、重抽样或稳健/高维方法，并重新说明其检验对象与校准依据。

这些是**经典有限样本推断**的条件，不是计算均值和 SSCP 的条件。即使条件不成立，描述统计仍可计算；失去的是把统计量解释成经典尾概率的授权。MANOVA 也不会创造随机分配、可忽略性或无混杂，因果边界见 [[多响应因果边界]]。

> [!question]- 自检
> 每名学生在三个时间点接受同一组响应测量，能否把三行直接当作独立观测套单因素 MANOVA？
>
> **答案：** 不能。重复测量共享同一学生，观测行相关；需要显式建模受试者内依赖或使用匹配该设计的重复测量方法。

## 来源与核验

- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08)：核对共同组内协方差、独立抽样、多元正态、残差诊断、Box 齐性检验与异常值边界。
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm)：核对多响应线性模型的跨观测独立、观测内响应相关和共同 $\Sigma$ 结构。
- [[矩阵正态误差结构]]、[[Wishart抽样假设]]与[[误差SSP秩边界]]：分别核对行/响应协方差方向、精确抽样律和逆矩阵秩条件。
<!-- bilingual-en:start -->
- Penn State and the SAS GLM documentation verify the classical covariance, independence, and normality model; the reused atoms fix its sampling-law and rank boundaries.
<!-- bilingual-en:end -->
