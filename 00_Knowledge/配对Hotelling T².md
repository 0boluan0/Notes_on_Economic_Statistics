---
aliases:
  - "配对多元比较先构造每对差向量再做单样本 Hotelling T²"
  - Paired multivariate comparison reduces to one-sample Hotelling on difference vectors
  - 配对 Hotelling T²
  - 配对均值向量比较
student_os: knowledge-atom
atom_id: STAT-HOT-006
atom_set: hotelling-mean-inference
atom_type: design-reduction
status: source-checked
mastery_state: unassessed
requires:
  - "[[单样本Hotelling T²]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
related:
  - "[[线性约束Hotelling T²]]"
  - "[[pooled Hotelling T²]]"
---

# 配对多元比较先构造每对差向量再做单样本 Hotelling T²
<!-- bilingual-en:start -->
*A paired multivariate comparison first forms within-pair difference vectors and then applies one-sample Hotelling T-squared*
<!-- bilingual-en:end -->

> [!summary] 原子化约
> 对第 $i$ 个匹配单位观察两个 $p$ 维向量 $(X_i,Y_i)$，先定义
> $$D_i=X_i-Y_i.$$
> 若不同 pair 之间的 $D_i$ 独立同分布，且 $D_i\sim N_p(\delta,\Sigma_D)$、$\Sigma_D\succ0$、$n>p$，则检验 $H_0:\delta=\delta_0$ 使用
> $$T_D^2=n(\bar D-\delta_0)^TS_D^{-1}(\bar D-\delta_0),$$
> $$\frac{n-p}{p(n-1)}T_D^2\sim F_{p,n-p}.$$
> <!-- bilingual-en:start -->
> Pairing is absorbed by differencing within the same unit. The resulting difference vectors, not the two margins separately, are the sampling units for the exact one-sample theorem.
> <!-- bilingual-en:end -->

同一 pair 内的 $X_i$ 与 $Y_i$ **可以相关**，这正是配对设计要利用的信息：
$$
\operatorname{Var}(D_i)
=\Sigma_X+\Sigma_Y-\Sigma_{XY}-\Sigma_{YX}.
$$
把两次测量误当成独立组，会丢掉交叉协方差项并使用错误的标准误。反过来，不同 pair 之间仍需独立；“同一对象内允许相关”不等于允许整个样本任意依赖。

若把差定义为 $Y_i-X_i$，零差异检验的 $T^2$ 与 p 值不变，但估计方向和区间符号全部反转，所以报告时必须说明减法顺序。

> [!warning] 边界
> - 配对必须有真实的设计对应关系；把两组独立样本按排序或行号强行配成 pair，不会创造有效配对信息。
> - 精确正态条件只要求差向量 $D_i$ 多元正态；边际如何相关由 $\Sigma_D$ 汇总。若 $S_D$ 奇异，不能直接使用经典 F 校准。
> - 有缺失配对时，直接化约只适用于保留下来的完整 pairs；如何处理非完整资料是额外的缺失数据问题。

> [!question]- 自检
> 为什么配对设计不要求同一对象的两次测量相互独立？
>
> **答案：** 推断单位是差向量 $D_i$。同一 pair 内的相关性进入 $\operatorname{Var}(D_i)$；需要的是不同 pairs 的差向量相互独立。

## 来源与核验

- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.2. 配对样本均值向量比较|多元统计课程 §1.2]]：核对差向量、$S_D$、$T_D^2$、F 转换与同时区间。
- [Penn State STAT 505, Lesson 7, §§7.1.6–7.1.10](https://online.stat.psu.edu/stat505/Lesson07)：核对配对 sampling unit、差向量假设与配对 Hotelling 检验。
- [[单样本Hotelling T²]]：核对化约后使用的精确单样本定理。
