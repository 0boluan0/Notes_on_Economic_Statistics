---
aliases:
  - "Bayes 分类先用先验与类条件分布得到后验，再按给定损失选择后验风险最小的行动"
  - "Bayes classification rule"
  - "Bayes decision rule"
student_os: knowledge-atom
atom_id: STAT-DA-002
atom_set: discriminant-analysis
atom_type: decision-rule
status: source-checked
mastery_state: unassessed
requires:
  - "[[Bayes法则]]"
leads_to:
  - "[[先验与分类风险]]"
  - "[[Gaussian判别得分]]"
part_of:
  - "[[判别分析.canvas|判别分析]]"
---

# Bayes 分类先用先验与类条件分布得到后验，再按给定损失选择后验风险最小的行动
<!-- bilingual-en:start -->
*Bayes classification first combines priors and class-conditional distributions into posteriors, then chooses the action with minimum posterior risk under the stated loss*
<!-- bilingual-en:end -->

> [!summary] 怎样决策
> 对类别 $Y\in\{1,\ldots,g\}$，先验为 $\pi_j=P(Y=j)$，类条件密度为 $f_j(x)$，则
> $$
> P(Y=j\mid x)=\frac{\pi_j f_j(x)}{\sum_{\ell=1}^g\pi_\ell f_\ell(x)}.
> $$
> 后验概率只是决策输入；真正的 Bayes 行动还取决于损失函数。
> <!-- bilingual-en:start -->
> Bayes' rule yields posterior class probabilities. A Bayes classifier then minimises posterior expected loss, of which maximum-posterior classification is only a special case.
> <!-- bilingual-en:end -->

## 一般规则是最小化后验风险

令 $a$ 是可采取的行动，$L(a,j)$ 是真实类别为 $j$ 时采取 $a$ 的损失。观察到 $x$ 后，行动 $a$ 的后验风险为

$$
\rho(a\mid x)=\sum_{j=1}^g L(a,j)P(Y=j\mid x).
$$

Bayes 规则选择

$$
a^*(x)\in\arg\min_a\rho(a\mid x).
$$

行动不一定只限于输出某个类别；在高风险场景中，还可以包含“转人工复核”或“暂缓决定”。一旦加入拒绝行动及其成本，分类区域就会随之改变。

## 最大后验只是 0–1 损失特例

若行动就是类别标签，预测正确损失为 0、任何误判损失都为 1，则

$$
a^*(x)\in\arg\max_j P(Y=j\mid x)
=\arg\max_j \pi_j f_j(x).
$$

这就是最大后验（MAP）分类。错误后果不对称时，后验最大的类别未必是风险最小的行动；不能把“概率超过 0.5”当作普遍阈值。[[先验与分类风险]]给出两类损失比与似然比的具体关系。

在实现中常比较 $\log\pi_j+\log f_j(x)$，既避免很小密度相乘造成数值下溢，也保留相同的 0–1 损失排序。

> [!question]- 自检
> 某患者患病后验概率为 0.40，因此一定应判为“不患病”吗？
>
> **答案：** 不一定。若漏诊损失远高于误报损失，判为患病或转人工复核可能具有更低的后验风险。

## 来源与核验

- [[Bayes法则]]：复用先验、似然与后验的概率关系。
- [Penn State STAT 505, Lesson 10](https://online.stat.psu.edu/stat505/Lesson10)：核对后验概率与最大后验分类规则。
- [Stanford MS&E 226, Lecture 16: Bayesian Decision Theory](https://web.stanford.edu/class/msande226/2025/lectures/lecture16_bayesian.pdf)：核对一般后验期望损失与 Bayes 行动。
