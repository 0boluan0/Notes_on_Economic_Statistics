---
aliases:
  - "Fisher 判别通过最大化投影后的类间分离相对类内变异来选择监督投影方向"
  - "Fisher linear discriminant"
  - "Fisher discriminant criterion"
student_os: knowledge-atom
atom_id: STAT-DA-007
atom_set: discriminant-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[均值向量]]"
  - "[[协方差矩阵]]"
leads_to:
  - "[[Fisher与LDA]]"
part_of:
  - "[[判别分析.canvas|判别分析]]"
---

# Fisher 判别通过最大化投影后的类间分离相对类内变异来选择监督投影方向
<!-- bilingual-en:start -->
*Fisher discrimination chooses supervised projection directions by maximising projected between-class separation relative to within-class variation*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> Fisher 判别先找投影 $z=a^Tx$，使类别均值在这条线上相隔得远，同时类内散布尽量小。它利用标签定义“类间”和“类内”，所以是监督降维准则；它本身不是概率模型。
> <!-- bilingual-en:start -->
> Fisher's criterion finds a supervised projection with large between-class separation relative to within-class spread. The class labels define both parts of the criterion; no Gaussian density is required to define it.
> <!-- bilingual-en:end -->

## 两类 Fisher 准则

令两类均值为 $\mu_1,\mu_2$，类内散布或共同组内协方差为 $S_W$。对非零方向 $a$，Fisher 准则可写成

$$
J(a)=
\frac{\left[a^T(\mu_1-\mu_2)\right]^2}
{a^TS_Wa}.
$$

若 $S_W\succ0$，使 $J(a)$ 最大的方向满足

$$
a\propto S_W^{-1}(\mu_1-\mu_2).
$$

整体缩放与反号不改变投影轴或排序，所以 Fisher 方向本身只确定到非零比例常数。若 $S_W$ 奇异或病态，直接求逆不成立或不稳定，应先检查 [[判别协方差秩边界]]，再选择是否采用 [[正则化判别]]。

## 多类情形仍使用同一分离准则

对 $g>2$ 个类别，Fisher 方法以类间散布矩阵 $S_B$ 和类内散布矩阵 $S_W$ 构造广义特征问题。可获得的非零判别方向最多为 $\min(p,g-1)$，因为 $g$ 个类别均值围绕总均值最多张成 $g-1$ 维空间。这个上限来自类间散布的秩，不是人为选择的经验规则。

两类共享协方差时，Fisher 方向为何与 LDA 的边界法向量相联、又为何不能据此把两种方法视作同一个完整分类规则，见 [[Fisher与LDA]]。

> [!question]- 自检
> 两个候选方向投影后的类别均值距离相同，为什么 Fisher 准则仍可能偏好其中一个？
>
> **答案：** Fisher 准则还用投影后的类内变异作分母；在均值间距相同的情况下，类内散布更小的方向具有更大的分离比。

## 来源与核验

- [Fisher (1936), *The use of multiple measurements in taxonomic problems*](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)：核对原始线性判别思想。
- [Penn State STAT 505, classification and discrimination notes](https://online.stat.psu.edu/stat505/Lesson10)：核对 Fisher 投影、共享组内协方差与课程分类步骤。
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed., §4.3：核对 Fisher 子空间与多类 $g-1$ 维上限。
