---
aliases:
  - "反演单样本 Hotelling T² 检验得到均值向量置信椭球与全方向同时区间"
  - Inverting Hotelling T-squared gives a confidence ellipsoid and all-direction simultaneous intervals
  - 均值向量的 Hotelling 置信椭球
  - 所有线性组合的 Hotelling 同时区间
student_os: knowledge-atom
atom_id: STAT-HOT-004
atom_set: hotelling-mean-inference
atom_type: equivalence-theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[单样本Hotelling T²]]"
  - "[[Hotelling投影极值]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
contrasts_with:
  - "[[Bonferroni对比区间]]"
related:
  - "[[正定二次型的椭球]]"
---

# 反演单样本 Hotelling T² 检验得到均值向量置信椭球与全方向同时区间
<!-- bilingual-en:start -->
*Inverting the one-sample Hotelling test gives a confidence ellipsoid and simultaneous intervals for every linear direction*
<!-- bilingual-en:end -->

> [!summary] 原子等价
> 在经典单样本条件下，令
> $$c_\alpha=\frac{p(n-1)}{n-p}F_{p,n-p}(1-\alpha).$$
> 反演所有水平 $\alpha$ 的 $H_0:\mu=\mu_0$ 检验，得到精确覆盖率 $1-\alpha$ 的置信区域
> $$\mathcal C_{1-\alpha}=\left\{\mu:
> n(\bar X-\mu)^TS^{-1}(\bar X-\mu)\le c_\alpha\right\}.$$
> 同一个椭球事件等价于：**所有** $a\in\mathbb R^p$ 同时满足
> $$a^T\mu\in a^T\bar X\ \pm
> \sqrt{\frac{c_\alpha}{n}\,a^TSa}.$$
> <!-- bilingual-en:start -->
> One random ellipsoid has exact joint coverage. Projecting that same ellipsoid gives intervals that hold simultaneously for every linear combination, not merely for the coordinates.
> <!-- bilingual-en:end -->

取 $a=e_j$ 就得到第 $j$ 个均值分量的同时区间。取一般 $a$ 则得到任何方向的区间；因此这里的 family 是无限多个线性组合，却由同一个椭球事件统一覆盖。椭球的主轴由 $S$ 的特征向量决定，半轴长度随相应样本方差的平方根变化。

$\mu_0$ 落在 $\mathcal C_{1-\alpha}$ 外，当且仅当对应的双侧 Hotelling 检验在水平 $\alpha$ 拒绝它。这是“置信域是检验反演”的精确含义。

> [!warning] 边界
> - 这是总体**均值参数**的置信区域，不是未来个体观测的预测椭球，也不是覆盖一定比例总体观测的 tolerance region；后两者需要不同的方差与校准。
> - 若 $S$ 奇异，集合可能沿某些方向无界或二次型无普通逆，不能继续称为这里的经典有界置信椭球。
> - 全局显著只表示目标向量在椭球外，不表示每个坐标区间都排除其目标值。
> - 当 $p=1$ 时，$F_{1,n-1}(1-\alpha)=t_{n-1}^2(1-\alpha/2)$，公式退化为普通双侧 t 置信区间。

> [!question]- 自检
> 为什么从这个椭球投影出的坐标区间可以同时成立，而不是每个只有单独的 $1-\alpha$ 覆盖率？
>
> **答案：** 所有投影区间都由同一个事件 $\mu\in\mathcal C_{1-\alpha}$ 推出；只要整个向量落在椭球中，所有方向的投影就同时落在各自区间中。

## 来源与核验

- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.5. 置信区域与同时置信区间|多元统计课程 §1.5]]：核对椭球、分量区间与任意 $a^T\mu$ 区间公式。
- [Penn State STAT 505, Lesson 7, §§7.1.3–7.1.4 and §7.2.8](https://online.stat.psu.edu/stat505/Lesson07)：核对检验反演、Hotelling 同时区间及其解释。
- [[正定二次型的椭球]]：核对二次型等值面的主轴与半轴几何。
