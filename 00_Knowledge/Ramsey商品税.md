---
aliases:
  - "Ramsey商品税在给定财政收入目标且忽略分配与外部性时以补偿需求反应分配税负，逆弹性规则只在补偿交叉价格效应为零等简化条件下成立"
  - "Ramsey commodity taxation and the conditional inverse-elasticity rule"
  - "Ramsey rule"
  - "拉姆齐商品税"
student_os: knowledge-atom
atom_id: PF-TAX-009
atom_set: optimal-taxation
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[税收超额负担]]"
  - "[[最优税收约束]]"
  - "[[普通与补偿需求]]"
  - "[[Slutsky方程]]"
related:
  - "[[超额负担平方律]]"
  - "[[税收公平]]"
  - "[[庇古税]]"
  - "[[一般均衡归宿]]"
part_of:
  - "[[最优税收.canvas]]"
---

# Ramsey商品税在给定财政收入目标且忽略分配与外部性时以补偿需求反应分配税负，逆弹性规则只在补偿交叉价格效应为零等简化条件下成立
<!-- bilingual-en:start -->
*Ramsey commodity taxation allocates taxes using compensated demand responses for a fixed fiscal-revenue target when distribution and externalities are ignored; the inverse-elasticity rule additionally requires conditions such as zero compensated cross-price effects*
<!-- bilingual-en:end -->

> [!summary] 定理边界
> Ramsey 问题的原始目标是：用比例商品税筹集给定财政收入目标，使效用损失尽量小。在局部、无分配差异、无外部性且价格反应可正确测量的模型中，最优税向量使各税基的补偿需求按相同比例收缩。“对低弹性商品征高税”只是没有补偿交叉价格效应、税很小等条件下的特例，不是现实商品税的普遍处方。
> <!-- bilingual-en:start -->
> Ramsey's problem raises a fixed amount of revenue while minimizing utility loss. Its local condition equalizes proportional compensated demand reductions across taxed goods. The familiar prescription of higher rates on less elastic goods is only a special case with no cross-price effects and other strong assumptions.
> <!-- bilingual-en:end -->

## 一般局部条件是一个向量问题

设生产者价格 $p_i$ 在局部比较中固定，单位税为 $t_i$，消费者价格 $q_i=p_i+t_i$，$x_j^c(q,u)$ 是固定效用 $u$ 的 Hicks 补偿需求。对一组足够小的税，Ramsey 局部一阶条件可写成

$$
\sum_i t_i\frac{\partial x_j^c}{\partial q_i}
=-\theta x_j,
\qquad j=1,\ldots,m,
$$

其中 $\theta>0$ 由收入约束确定。左边是**整个税向量**通过自价与交叉价格反应引起的商品 $j$ 补偿需求局部变化；右边要求这个变化占原需求 $x_j$ 的比例相同。交叉导数不为零时，对 A 加税会改变 B 的需求，不能按每件商品的单个自价弹性各算各的。

这里用补偿需求，是因为税收超额负担来自改变相对价格的替代效应。普通需求同时含收入效应，把两者混在一起会把转移性税负与资源损失混淆。

## 逆弹性规则是特例

若再加上三个强条件：

1. 各商品补偿需求间没有交叉价格效应；
2. 生产者价格局部固定，归宿与供给调整不另行改变税楔；
3. 税足够小，可用当前点弹性作一阶近似；

则对商品 $i$，上式简化为

$$
\tau_i\lvert\varepsilon_{ii}^c\rvert=\theta,
$$

其中 $\tau_i=t_i/q_i$ 是小额从价税率，$\varepsilon_{ii}^c<0$ 是自价补偿需求弹性。所以 $\tau_i\propto1/\lvert\varepsilon_{ii}^c\rvert$：其他条件不变时，弹性较小的税基需要较高税率，才会产生相同的比例收缩。若供给并非完全有弹性，Ramsey 原始的独立商品特例会同时出现供给与需求弹性的倒数，不能只看消费需求。

## 为什么不能把它直接变成税率表

- **分配：** Ramsey 原始问题明确忽略个人间货币边际效用差异。若低弹性商品是低收入家庭的必需品，较高税率可能带来较大分配损失；是否可用现金转移补偿又取决于信息与工具。
- **外部性：** 烟草、碳排放等税还有纠正外部损害的任务。纠正部分与纯筹资部分应分开识别，不能因为需求弹性大就否定纠正税。
- **一般均衡与动态：** 替代品、生产要素、跨境购买和长期技术调整会改变弹性和归宿。一个短期局部弹性不足以代表长期社会成本。
- **征管：** 税率差过大会创造分类和边界操作。理论上的小额差别化收益可能低于现实分类、遵从与执法成本。

> [!example] 面包与奢侈品
> 若面包补偿需求弹性为 $-0.2$，奢侈品为 $-1.0$，且需求独立、生产者价格固定、只有筹资目标，逆弹性近似给出的税率比为 $5:1$。这个结果只说明所设效率问题的局部解；一旦面包集中于低收入家庭、存在交叉需求或精准转移不可行，$5:1$ 就不再是现实处方。

> [!question]- 自检
> 某必需品的补偿需求弹性绝对值最小，是否已证明它应征最高税率？
>
> **答案：** 没有。只有在给定收入、忽略分配与外部性、无交叉价格效应、供给和价格条件已声明、小税近似可用时，才能进入需求逆弹性特例。现实还要加入分配权重、外部性、归宿、征管与动态反应。

## 来源与核验

- [Ramsey (1927), “A Contribution to the Theory of Taxation,” pp. 47–48, 51–52 and 55–57](https://upload.wikimedia.org/wikipedia/commons/c/cb/A_contribution_to_the_theory_of_taxation%2C_Frank_Plumpton_Ramsey_1927.pdf)：核对给定收入、最小化效用损失、忽略分配和外部性的原始问题，各税基同比例收缩的局部结论，以及独立商品下同时含供给与需求弹性的特例。
- [MIT 14.471, Recitation 13, p. 1](https://ocw.mit.edu/courses/14-471-public-economics-i-fall-2012/8424e343baacf31b0138421ef9529da9_MIT14_471F12_recnotes13.pdf)：核对用补偿需求矩阵写出的 single- and multiple-consumer Ramsey 条件，以及分配权重会改变结论。
- [[02_Economy/02_public finance财政学/3_财政收入/04_有效且公平地课税#原则|本地课程：拉姆齐法则]]：核对课程记号、同比例需求缩减和面包—奢侈品公平反例。
