---
aliases:
  - "Johansen 检验是在 VECM 中联合判断协整秩并估计协整空间的系统方法"
  - Johansen cointegration test
  - Johansen 系统协整检验
student_os: knowledge-atom
atom_id: TS-CI-024
atom_set: cointegration-error-correction
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR到VECM重参数化]]"
  - "[[VECM的Π秩]]"
related:
  - "[[EG多变量局限]]"
  - "[[Johansen规格选择]]"
leads_to:
  - "[[Johansen广义特征值]]"
  - "[[Johansen秩检验]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Johansen 检验是在 VECM 中联合判断协整秩并估计协整空间的系统方法
<!-- bilingual-en:start -->
*The Johansen test is a system method that jointly determines cointegration rank and estimates the cointegration space in a VECM*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> Johansen 检验从一个至多为 $I(1)$ 的多变量 VAR/VECM 出发，对长期矩阵 $\Pi$ 施加候选秩限制，并用约化秩高斯似然判断协整空间有多少个独立方向。它既回答“有几条关系”，也估计这些关系张成的空间；它并不直接赋予任何一条关系经济身份。

计算时，先把 $\Delta x_t$ 与 $x_{t-1}$ 分别对滞后差分和给定的确定性项残差化，再从两组残差的典型相关问题得到一组广义特征值。[[Johansen秩检验|trace 与最大特征值统计量]]用这些根比较不同的秩假设；它们不是在数 $\hat\Pi$ 的普通特征值。选定 $r$ 后，$\Pi=\alpha\beta'$ 中的 $\beta$ 张成协整空间，$\alpha$ 记录各方程怎样响应长期偏离。

这套方法的“系统性”有两个含义。第一，它不必像单方程 Engle–Granger 那样先指定一个被解释变量，因此能处理 $r>1$；第二，变量、滞后和确定性项作为一个联合系统进入似然。代价是规格更敏感、参数更多，小样本中也未必优于一个理论明确的单方程方法。

解释结果前至少要确认：变量的整合阶数与标准 $I(1)$ 框架相容；水平 VAR 滞后足以吸收短期相关；常数和趋势的放置与问题一致；创新与系统根满足所用参考分布的条件；结构断点和异常值没有把固定协整空间扭曲掉。秩只是入口，后面仍要检查归一化、经济限制、$\alpha$、短期动态和样本稳定性。

> [!question]- 自检
> 软件输出三组 Johansen 广义根，能否把“大于零的根数”直接当成协整秩？
>
> **答案：** 不能。样本根几乎不会精确等于零；必须用与确定性项规格匹配的顺序秩检验，并结合残差和稳定性诊断。

## 来源与核验

- [Johansen (1988), *Statistical Analysis of Cointegration Vectors*](https://doi.org/10.1016/0165-1889(88)90041-3)：核对 $I(1)$ 高斯 VAR、协整空间的最大似然估计及其维数检验。
- [[01_Math/06_时间序列分析/lecture.pdf#page=291|时间序列课程 Lecture 7]]：核对本课程的 VECM 重参数化、三种秩情形及 trace/max-eigen 统计量；课程中把检验根称作“$\Pi$ 的特征值”的旧表述按讲义自身勘误改为残差典型相关的广义根。
