---
aliases:
  - "简约型 VAR 可逐方程 OLS 但仍需正交性与动态正则条件"
  - Equation-by-equation OLS for VAR
  - VAR 逐方程 OLS
student_os: knowledge-atom
atom_id: TS-VAR-003
atom_set: vector-autoregression
atom_type: condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR(p)模型]]"
  - "[[普通最小二乘]]"
  - "[[OLS一致性条件]]"
related:
  - "[[零条件均值无偏性]]"
  - "[[满列秩与OLS唯一性]]"
  - "[[样本正交与总体外生性]]"
  - "[[简约型VAR创新]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# 简约型 VAR 可逐方程 OLS 但仍需正交性与动态正则条件
<!-- bilingual-en:start -->
*A reduced-form VAR can be estimated equation by equation by OLS, subject to orthogonality and dynamic regularity*
<!-- bilingual-en:end -->

> [!summary] 适用边界
> 简约型 VAR 的各方程拥有同一组滞后回归量，因此可逐方程做 OLS；这一计算便利依赖创新对过去回归量的正交性、设计矩阵满秩及适当的大样本稳定条件，并不等于“内生性已经被消除”。

把 VAR 写成
$$
y_t=B'z_t+u_t,
$$
其中 $z_t$ 堆叠截距、确定项和 $y_{t-1},\ldots,y_{t-p}$。每个分量方程都使用同一个 $z_t$。因此逐方程 OLS 与在共同回归量下的多元最小二乘给出相同的系数点估计；$\operatorname{Cov}(u_t)=\Sigma_u$ 的非对角元素，即不同方程创新同期相关，本身不会改变这一点估计。

但可计算不等于可相信。常用的关键条件包括：

1. $u_t$ 对用于预测的过去信息正交，例如 $E(u_t\mid\mathcal F_{t-1})=0$；
2. 回归量没有完全线性相关，使样本或总体矩阵具有足够秩；
3. 存在所需矩，且过程满足让样本矩收敛的稳定性、遍历性或相应非平稳渐近条件；
4. 推断所用标准误与检验必须匹配创新的条件异方差、序列结构和模型设定。

简约型方程右侧没有当期系统内生变量，这避免了同期联立方程的直接 OLS 问题；它却没有证明遗漏变量不存在，也没有把滞后变量变成严格外生变量，更没有从 $\Sigma_u$ 中恢复结构冲击。

> [!question]- 自检
> 如果两条 VAR 方程的残差在同一期高度相关，逐方程 OLS 的系数是否因此自动有偏？
>
> **答案：** 不会仅因同期跨方程相关而自动有偏；关键仍是创新是否与各方程的滞后回归量正交。同期相关主要进入联合协方差与结构识别问题。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 3 章：核对共同回归量下逐方程最小二乘及其渐近条件。
- [[OLS一致性条件]]、[[满列秩与OLS唯一性]]：复用 OLS 的总体正交、收敛和唯一性边界。
- [[01_Math/06_时间序列分析/lecture.pdf]]：核对课程“逐方程 OLS”结论，并修正“消除内生性”的过度表述。
