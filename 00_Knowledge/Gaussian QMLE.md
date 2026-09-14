---
aliases:
  - "Gaussian QMLE 在条件均值方差正确时仍需稳健标准误"
  - Gaussian GARCH QMLE
  - Bollerslev-Wooldridge robust covariance
  - GARCH 准极大似然
student_os: knowledge-atom
atom_id: TS-VOL-012
atom_set: conditional-volatility
atom_type: inference-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH条件似然]]"
related:
  - "[[Student-t GARCH标准化]]"
  - "[[GARCH残差双层诊断]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# Gaussian QMLE 在条件均值方差正确时仍需稳健标准误
<!-- bilingual-en:start -->
*Gaussian QMLE can remain valid under correct conditional moments but still requires robust standard errors*
<!-- bilingual-en:end -->

> [!summary] 原子推断
> 即使 $z_t$ 不是正态，也可最大化 Gaussian 条件对数似然作为准极大似然（QMLE）。在条件均值、条件方差递推、识别、平稳遍历与所需矩等正则条件成立时，参数估计可保持一致并渐近正态；这不是“分布随便错都没关系”。

非正态下，Gaussian 信息矩阵等式一般不成立。推断应使用 Bollerslev–Wooldridge 型 sandwich/robust covariance，而不是直接把负 Hessian 的逆当作正确协方差。重尾严重到必要矩不存在时，常见 QMLE 渐近结论也可能失效。

QMLE 保护的是正确指定的前两个条件矩。若方差递推漏掉不对称、断点或外生尺度，Gaussian 目标函数不会自动修复设定偏误；稳健标准误也只能修正协方差估计，不能把错误模型变真。

> [!question]- 自检
> 标准化残差明显厚尾，但 Gaussian QMLE 的系数看似稳定。下一步应怎样报告推断？
>
> **答案：** 明确称为 QMLE，检查其正则/矩条件并使用稳健 sandwich 标准误；同时比较厚尾似然与尾部诊断，不能沿用正态 MLE 的信息矩阵标准误。

## 来源与核验

- [Bollerslev & Wooldridge (1992), *Quasi-Maximum Likelihood Estimation and Inference in Dynamic Models with Time-Varying Covariances*](https://doi.org/10.1080/07474939208800229)：核对非正态 QMLE 与稳健标准误。
- [Glosten, Jagannathan & Runkle (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)：核对 GARCH 实证中条件 Gaussian QMLE 配合 robust inference 的使用。
