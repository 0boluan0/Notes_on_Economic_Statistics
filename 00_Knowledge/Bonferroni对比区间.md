---
aliases:
  - "Bonferroni 对比区间通过分配边际错误率同时覆盖预先指定的有限个线性对比"
  - Bonferroni contrast intervals allocate marginal error rates to cover a prespecified finite family simultaneously
  - 多元均值的 Bonferroni 同时区间
  - Hotelling 区间与 Bonferroni 区间的边界
student_os: knowledge-atom
atom_id: STAT-HOT-005
atom_set: hotelling-mean-inference
atom_type: procedure-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Gaussian仿射闭包]]"
  - "[[正态均值协方差独立]]"
  - "[[样本协方差Wishart律]]"
  - "[[Wishart抽样假设]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
contrasts_with:
  - "[[Hotelling置信椭球]]"
related:
  - "[[单样本Hotelling T²]]"
  - "[[Hotelling投影极值]]"
---

# Bonferroni 对比区间通过分配边际错误率同时覆盖预先指定的有限个线性对比
<!-- bilingual-en:start -->
*Bonferroni contrast intervals allocate marginal error rates to cover a prespecified finite family simultaneously*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 设 $X_1,\ldots,X_n\overset{iid}{\sim}N_p(\mu,\Sigma)$，其中 $\Sigma\succ0$、$n\ge2$。预先固定 $m$ 个非零方向 $a_1,\ldots,a_m$。对每个 $j$ 使用
> $$a_j^T\bar X\ \pm
> t_{n-1}\!\left(1-\frac{\alpha}{2m}\right)
> \sqrt{\frac{a_j^TSa_j}{n}}$$
> 可使这 $m$ 个区间的联合覆盖率至少为 $1-\alpha$。保证来自 union bound，不要求这些投影彼此独立。
> <!-- bilingual-en:start -->
> Each projected sample gives an exact marginal t interval under normal sampling; allocating error probability $\alpha/m$ and applying the union bound controls the finite family-wise noncoverage probability.
> <!-- bilingual-en:end -->

Bonferroni 与 Hotelling 椭球投影同时保护的对比范围不同：

- Bonferroni 只覆盖事先列出的 $m$ 个方向；
- [[Hotelling置信椭球|Hotelling 椭球]] 同时覆盖所有 $a\in\mathbb R^p$；
- 因而不能只比较某一个数值区间的宽度，就说两种方法回答同一个问题。

当 $m$ 很小，Bonferroni 的临界乘子可能小于 Hotelling 的全方向乘子；当 $m$ 很大时它往往更保守。哪一个更短取决于 $m,p,n,\alpha$，不存在“Bonferroni 总是更宽”的无条件结论。

> [!warning] 边界
> - $a_1,\ldots,a_m$ 与这组方向的数量必须在查看用于推断的数据前确定。先搜索很多方向，再只把显著方向当作原先的 $m$ 个比较，会失去声称的覆盖率。
> - 若只给每个方向普通 $1-\alpha$ 区间而不调整，整组区间的联合覆盖率一般低于 $1-\alpha$。
> - 公式只用各投影方差 $a_j^TSa_j$，不求 $S^{-1}$；所以即使 $p\ge n$、全维 Hotelling $T^2$ 不可用，固定有限方向的区间仍可在上述正态模型下精确成立。
> - 正态性保证每个投影的有限样本 t 枢轴。非正态样本下，union bound 本身仍成立，但必须另行证明或校准各边际区间；不能继续把上式无条件称为精确区间。
> - 这些区间用于定位预设的实质对比；它们既不证明未列出的方向没有差异，也不等价于一次整体的向量检验。

> [!question]- 自检
> 两个对比高度相关时，Bonferroni 是否失效，因为它假设比较相互独立？
>
> **答案：** 不失效。union bound 对任意依赖结构都成立；相关性可能使界保守，但不是有效性的前提。

## 来源与核验

- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.6. Bonferroni 多重比较|多元统计课程 §1.6]]：核对线性对比公式、$m$ 个比较的校准与宽度解释。
- [Penn State STAT 505, Lesson 7, §7.2.4](https://online.stat.psu.edu/stat505/Lesson07)：核对 Bonferroni 校正区间及其与 Hotelling 同时区间的比较。
- [Penn State STAT 505, Lesson 7, §7.2.8](https://online.stat.psu.edu/stat505/Lesson07)：核对有限预设对比组与全方向同时区间的对象差异。
