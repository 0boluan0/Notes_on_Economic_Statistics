---
aliases:
  - "负特殊方差是 Heywood 不当解而公共度大于 1 只在固定总方差参数化下与之等价"
  - "Heywood case"
  - "Improper factor solution"
student_os: knowledge-atom
atom_id: STAT-FA-009
atom_set: factor-analysis
atom_type: failure-diagnostic
status: source-checked
mastery_state: unassessed
requires:
  - "[[公共度与特殊方差]]"
  - "[[因子提取]]"
related:
  - "[[因子提取方法辨析]]"
  - "[[因子数选择]]"
leads_to:
  - "[[因子结构验证]]"
part_of:
  - "[[因子分析.canvas|因子分析]]"
---

# 负特殊方差是 Heywood 不当解而公共度大于 1 只在固定总方差参数化下与之等价
<!-- bilingual-en:start -->
*A negative unique variance is an improper Heywood solution; communality above one is equivalent only under a parameterisation that fixes total variance at one*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 合法因子模型要求 $\psi_i\ge0$。若估计得到 $\widehat\psi_i<0$，就出现 Heywood 不当解；由模型恒等式 $\widehat\sigma_{ii}=\widehat h_i^2+\widehat\psi_i$，这等价于公共度超过**模型隐含的总方差**。只有在总方差被固定为 1 且采用 $\psi_i=1-h_i^2$ 的参数化时，才能进一步写成 $\widehat h_i^2>1$。
> <!-- bilingual-en:start -->
> An improper Heywood solution violates the non-negative unique-variance boundary. It appears as communality greater than one only when the parameterisation fixes total variance at one.
> <!-- bilingual-en:end -->

## 数值收敛不等于统计上可接受

优化器可以停在参数空间外、边界附近或局部异常点，并照常打印载荷。负方差没有通常的概率解释，不能把它改写成“该因子解释了超过 100% 的信息”。若 $\widehat\psi_i=0$ 或极接近零，虽未必为负，却是边界或近边界解；标准误、检验近似和结构稳定性都可能变差，应单独报告。

Heywood 解是一项诊断信号，不对应唯一病因。常见来源包括：

- 因子数或零载荷约束错设；
- 样本量不足、指标相关过强或样本协方差不稳定；
- 异常值、编码错误或相关矩阵本身有问题；
- 某变量确实几乎没有特殊方差，使有限样本估计落到边界；
- 提取方法、起始值或优化过程未找到可接受解。

因此也不能机械地说“出现 Heywood 解就一定删变量”或“一定减少因子”。

## 处理顺序

先确认数据、量纲、反向题编码与相关矩阵；再查看具体哪个变量越界、残差模式和因子数敏感性；随后比较合理的模型约束、提取方法与新样本结果。把负值强行截成零只隐藏症状，除非零边界本身有明确理论依据并且采用适合边界参数的推断。

> [!question]- 自检
> 在把题项总方差固定为 1、并令 $\psi_i=1-h_i^2$ 的参数化下，估计公共度为 $1.08$。能否解释为“公共因子解释了 108% 的题项方差”？
>
> **答案：** 不能。它对应特殊方差 $-0.08$，是越出合法参数空间的 Heywood 不当解，需要诊断模型、数据与估计过程。

## 来源与核验

- [Wang et al. (2023), *Heywood Cases in Unidimensional Factor Models and Item Response Models for Binary Data*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9979198/)：核对不当解会随参数化与估计方法呈现为不同形式，公共度大于 1 不能脱离总方差约束解释。
- [Cooperman & Waller (2022), *Heywood You Go Away!*](https://pubmed.ncbi.nlm.nih.gov/34197140/)：核对 Heywood 解、弱因子与过度提取等成因边界。
- Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., Ch. 9：核对公共度、特殊方差与不当解的参数界限。
- [[公共度与特殊方差]]：复用 $\sigma_{ii}=h_i^2+\psi_i$ 的定义。
