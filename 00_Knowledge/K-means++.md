---
aliases:
  - "K-means++ 是按样本到已选中心集的最近平方距离抽取新中心的 K-means 播种算法"
  - "K-means++ seeds centres by squared distance from the nearest selected centre"
student_os: knowledge-atom
atom_id: STAT-CL-022
atom_set: cluster-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[K-means初始化]]"
  - "[[K-means目标]]"
related:
  - "[[K-means空簇]]"
  - "[[Lloyd局部收敛]]"
  - "[[K-means多启动]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# K-means++ 是按样本到已选中心集的最近平方距离抽取新中心的 K-means 播种算法
<!-- bilingual-en:start -->
*K-means++ is a K-means seeding algorithm that samples each new centre in proportion to the squared distance from the nearest centre already selected*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> K-means++ 先选一个样本作为中心，随后让距离已有中心较远的样本获得更大入选概率，直到选出 $K$ 个起始中心。它是一次 Lloyd 运行的初始化方法；不同于“把整个运行重复多次”的多启动。
> <!-- bilingual-en:start -->
> K-means++ first selects one observation as a centre, then gives observations farther from the selected set a larger chance of becoming the next centre, continuing until $K$ starting centres are chosen. It initialises one Lloyd run; it is distinct from restarting the whole run several times.
> <!-- bilingual-en:end -->

## $D^2$ 抽样规则

已选中心集合为 $M$时，定义

$$
D(x)=\min_{\mu\in M}\lVert x-\mu\rVert_2.
$$

若 $\sum_j D(x_j)^2>0$，下一个中心按

$$
\Pr(x_i\text{ 被选中})
=\frac{D(x_i)^2}{\sum_j D(x_j)^2}
$$

抽取。已选中心附近的点权重小，尚未被任何中心覆盖的区域权重大，所以种子通常比完全随机播种更分散。

若所有 $D(x_j)=0$，分母为 0，教科书抽样规则不再定义。这通常表示所有样本的不同几何位置都已被覆盖，而 $K$ 还要求更多中心；软件如何放置重复中心属于额外口径。

这里定义的是 Arthur–Vassilvitskii 的原始 $D^2$ 播种。scikit-learn 当前名为 `k-means++` 的选项是 greedy 变体：每步试抽多个候选再选一个。若要复现起点，需记录库、版本和随机种子。

## 播种之后仍是另一个求解问题

K-means++ 只定义一次运行的起始中心。随后的 [[Lloyd算法]] 仍只有 [[Lloyd局部收敛|局部收敛边界]]；要独立重复整个求解过程，还需另行设定 [[K-means多启动]]。

> [!question]- 自检
> 设置 `init="k-means++"` 后，能否直接报告“算法已做多启动”？
>
> **答案：** 不能。K-means++ 只规定一次运行怎样选种子；独立运行多少次是另一个设置。

## 来源与核验

- Arthur & Vassilvitskii (2007), [*k-means++: The Advantages of Careful Seeding*](https://research.google/pubs/k-means-the-advantages-of-careful-seeding/)：核对 $D^2$ seeding 规则。
- [scikit-learn, `KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)：核对该库使用 greedy k-means++，并核对初始化方法与 `n_init` 是两个独立参数对象。

> [!success] 审核状态
> 本卡已通过内容核验，状态为 `source-checked`；掌握状态仍为 `unassessed`。
