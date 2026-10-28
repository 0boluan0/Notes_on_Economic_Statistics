---
aliases:
- Pascal's identity
- Pascal's rule
- 帕斯卡恒等式
student_os: knowledge-atom
atom_id: MCS-COUNT-018
atom_set: mcs-counting
atom_type: identity
status: source-checked
requires:
- '[[组合数]]'
- '[[加法法则]]'
related:
- '[[组合数对称性]]'
- '[[二项式定理]]'
part_of:
- '[[组合计数原理.canvas]]'
---

<!-- bilingual-en:start -->
# Pascal's identity
<!-- bilingual-en:end -->
*Pascal 恒等式*

<!-- bilingual-en:start -->
Pascal's identity counts a selection by asking one yes-or-no question: **does it contain a particular object?** This turns one counting problem into two smaller, non-overlapping problems.

For example, choose two people from Alice, Ben, and Chen, with no roles or order. The three possible pairs are $\{A,B\}$, $\{A,C\}$, and $\{B,C\}$. If we focus on Alice, two pairs contain her and one does not. Every pair belongs to exactly one of these cases.
<!-- bilingual-en:end -->
Pascal 恒等式通过一个“是或否”的问题来数选择：**选出的对象中，有没有某个指定对象？** 这样，一个计数问题就分成两个更小、互不重叠的问题。

例如，从 Alice、Ben、Chen 中选两个人，不分职位也不计顺序。全部可能是 $\{A,B\}$、$\{A,C\}$、$\{B,C\}$。以 Alice 为关注对象，包含她的有两组，不包含她的有一组。每一组恰好属于其中一类。

<!-- bilingual-en:start -->
## From the two cases to the formula
<!-- bilingual-en:end -->
*从两个情况得到公式*

<!-- bilingual-en:start -->
Let $n$ be the number of distinct objects and $k$ the number to select. The [[组合数|binomial coefficient]] $\binom nk$ counts the $k$-element subsets: an object cannot be selected twice, and changing the order does not create a new subset. Fix one object $z$ among the $n$ objects.

If $z$ is **absent**, choose all $k$ objects from the other $n-1$, giving $\binom{n-1}{k}$ choices. If $z$ is **present**, it already fills one place; choose the remaining $k-1$ objects from the other $n-1$, giving $\binom{n-1}{k-1}$ choices. These cases cover every subset and never overlap, so the [[加法法则|sum rule for disjoint cases]] adds their counts.
<!-- bilingual-en:end -->
设 $n$ 是不同对象的总数，$k$ 是要选的个数。[[组合数]] $\binom nk$ 数的是含 $k$ 个元素的子集：同一对象不能重复选择，交换顺序也不产生新子集。从这 $n$ 个对象中固定一个 $z$。

若**不选 $z$**，就从其余 $n-1$ 个中选出全部 $k$ 个，有 $\binom{n-1}{k}$ 种。若**选 $z$**，它已经占了一个名额，只需从其余 $n-1$ 个中再选 $k-1$ 个，有 $\binom{n-1}{k-1}$ 种。这两类覆盖全部子集，而且没有重叠，因此可用[[加法法则]]相加。

> [!important] Pascal's identity / Pascal 恒等式
> <!-- bilingual-en:start -->
> For integers $n\ge1$ and $0\le k\le n$,
> <!-- bilingual-en:end -->
> 对整数 $n\ge1$、$0\le k\le n$，
>
> $$
> \boxed{\binom nk=\binom{n-1}{k}+\binom{n-1}{k-1}}.
> $$
>
> <!-- bilingual-en:start -->
> Here an impossible selection has count zero: for a nonnegative integer $r$, set $\binom rj=0$ when the integer $j<0$ or $j>r$.
> <!-- bilingual-en:end -->
> 这里把不可能的选择记为零：当 $r$ 是非负整数、整数 $j<0$ 或 $j>r$ 时，约定 $\binom rj=0$。

<!-- bilingual-en:start -->
The endpoints have concrete meanings. When $k=0$, the empty selection omits $z$, so the two counts are $1+0$. When $k=n$, every object must be selected, so the two counts are $0+1$. If $n=0$, there is one empty subset, $\binom00=1$; there is no distinguished object to split on, so the recurrence above starts at $n=1$.
<!-- bilingual-en:end -->
端点也有具体含义。$k=0$ 时，唯一的空选择不包含 $z$，两类数量是 $1+0$。$k=n$ 时，所有对象都必须入选，两类数量是 $0+1$。当 $n=0$ 时，只有一个空子集，即 $\binom00=1$；此时没有对象可以拿来分类，因此上面的递推从 $n=1$ 开始。

<!-- bilingual-en:start -->
## Reading Pascal's triangle
<!-- bilingual-en:end -->
*读懂 Pascal 三角形*

<!-- bilingual-en:start -->
Arrange row $n$ as $\binom n0,\binom n1,\ldots,\binom nn$. The first rows are
<!-- bilingual-en:end -->
把第 $n$ 行依次写成 $\binom n0,\binom n1,\ldots,\binom nn$，前几行是

$$
\begin{array}{c|ccccc}
n=0&1\\
n=1&1&1\\
n=2&1&2&1\\
n=3&1&3&3&1\\
n=4&1&4&6&4&1
\end{array}
$$

<!-- bilingual-en:start -->
Each interior entry is the sum of the two entries in the preceding row specified by the identity. Thus the triangle can be built by addition, and the selection argument explains why that construction gives binomial coefficients. Those same coefficients appear in the [[二项式定理|binomial theorem]] when a power of a two-term sum is expanded.

For a larger example, choosing three people from eight gives $\binom83=\binom73+\binom72=35+21=56$. The $35$ teams omit a fixed person; the $21$ teams contain that person and choose two others.
<!-- bilingual-en:end -->
每个内部数字都等于恒等式指定的上一行两个数字之和。因此可以只靠加法构造这个三角形，而按是否入选分类的证明解释了：为什么这样得到的恰好是组合数。展开两项之和的幂时，[[二项式定理]]使用的也是这些系数。

例如，从八个人中选三个人，$\binom83=\binom73+\binom72=35+21=56$。其中 $35$ 组不含某个固定的人；另外 $21$ 组含有这个人，并从其余人中再选两人。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=698|MIT Mathematics for Computer Science, §15.10.1, Lemma 15.10.1, printed pp. 690–691 (PDF pp. 698–699)]]: the distinguished-element proof of Pascal's identity. The endpoint convention and the $n=0$ base case above are checked directly as subset counts.
- [[01_Math/07-Mathematics for Computer Science/03_Counting#Pascal 恒等式|MCS course notes, Pascal's identity]]: course notation and the two disjoint selection cases.
<!-- bilingual-en:end -->
- MIT《Mathematics for Computer Science》§15.10.1、引理 15.10.1，印刷页 690–691（PDF 页 698–699）：核验固定元素入选与否的证明；端点约定与 $n=0$ 基础情形按子集含义直接核算。
- MCS 课程笔记“Pascal 恒等式”：核对课程记号以及两类互斥选择的分解。
