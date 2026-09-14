---
aliases:
  - 单向大O而非同阶并不足以推出小o
  - One-way big O does not imply little o
student_os: knowledge-atom
atom_id: CS-ASYM-013
atom_type: distinction
status: source-checked
part_of:
  - "[[渐近记号与算法复杂度.canvas]]"
requires:
  - "[[大O记号]]"
  - "[[大Θ记号]]"
  - "[[小o记号]]"
related:
  - "[[同阶不要求比值收敛]]"
---

# 单向大O而非同阶并不足以推出小o
<!-- bilingual-en:start -->
*One-way big O without equal order does not imply little o*
<!-- bilingual-en:end -->

这里把最终正函数间的 $f=O(g)$ 且 $g\notin O(f)$ 简称为“严格大 O”，等价于单向大 O 且非 $\Theta$。**这是本卡说明的关系，不是小 o 的另一种标准名称。** 小 o 必然满足这种单向关系，但反向不成立。
<!-- bilingual-en:start -->
Here “strict big O” abbreviates $f=O(g)$ and $g\notin O(f)$ for eventually positive functions, equivalently one-way big O without Theta. **This locally specified relation is not another standard name for little o.** Little o implies this one-way relation, but the converse fails.
<!-- bilingual-en:end -->

若 $f=o(g)$，则 $f/g\to0$，先得到 $f=O(g)$。若又有 $g=O(f)$，则某个 $C>0$ 使 $g\le Cf$ 最终成立，从而 $f/g\ge1/C>0$，与趋零矛盾。
<!-- bilingual-en:start -->
If $f=o(g)$, then $f/g\to0$, first giving $f=O(g)$. If also $g=O(f)$, some $C>0$ would eventually give $g\le Cf$, hence $f/g\ge1/C>0$, contradicting convergence to zero.
<!-- bilingual-en:end -->

反例令 $n\ge1$ 为整数，并取
<!-- bilingual-en:start -->
For a counterexample, take integers $n\ge1$ and define
<!-- bilingual-en:end -->

$$
g(n)=n,\qquad
f(n)=\begin{cases}n,&n\text{ even},\\1,&n\text{ odd}.\end{cases}
$$

始终有 $0<f\le g$，故 $f=O(g)$；但奇数处 $g/f=n$ 无界，反向大 O 不成立。另一方面，偶数处 $f/g=1$，所以比值不趋零，$f\ne o(g)$。在这个例子中 $f,g$ 已能用大 O 单向比较，却既无 $f=o(g)$，也无 $g=o(f)$：小 o 关系不能把所有最终正函数排成一条增长链。
<!-- bilingual-en:start -->
Always $0<f\le g$, so $f=O(g)$. However, $g/f=n$ is unbounded on odd inputs, disproving reverse big O. On even inputs $f/g=1$, so the ratio does not tend to zero and $f\ne o(g)$. These functions are comparable in one direction by big O, but neither $f=o(g)$ nor $g=o(f)$ holds: little o cannot arrange all eventually positive functions into one growth chain.
<!-- bilingual-en:end -->

应用时分清两种问题：“比值最终有上界但没有正下界”只给出单向大 O；“比值最终小于每个正数”才给出小 o。可用 [[渐近比较的比值判别]] 的上下极限检查这一区别。
<!-- bilingual-en:start -->
In applications, distinguish “eventually bounded above but with no positive lower bound” from “eventually smaller than every positive number”. The first gives one-way big O; the second gives little o. Upper and lower limits in [[渐近比较的比值判别|ratio tests]] help separate them.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=2|MIT Session 24，印刷页 529，PDF 页 2]]，Lemmas 13.7.7–13.7.8：支持小 o 蕴含大 O 并排除反向大 O；本文在最终正域用矛盾法写出必要推导。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp24.pdf#page=1|MIT CP24，PDF 页 1，Problem 2(a) 第五项]]：独立列出“大 O 且非反向大 O”的关系供分类。本文原创奇偶反例按三个比值结论直接核验，证明其不等同于小 o。
<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=2|MIT Session 24, printed p. 529, PDF p. 2]], Lemmas 13.7.7–13.7.8, supports little o implying big O and excluding reverse big O. The contradiction proof makes the eventually-positive domain explicit.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp24.pdf#page=1|MIT CP24, PDF p. 1, the fifth relation in Problem 2(a)]], separately lists big O without reverse big O for classification. The original parity example here is checked directly against all three ratio conclusions, proving that the relation differs from little o.
<!-- bilingual-en:end -->
