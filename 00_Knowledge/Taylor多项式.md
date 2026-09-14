---
aliases:
  - "$n$ 次 Taylor 多项式是在展开中心匹配函数前 $n$ 阶导数的唯一 $n$ 次以下多项式"
  - "The degree-n Taylor polynomial is the unique polynomial of degree at most n matching the first n derivatives at the expansion centre"
  - Taylor polynomial
student_os: knowledge-atom
atom_id: CALC-APP-003
atom_set: derivative-applications
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[导数的应用.canvas]]"
  - "[[无穷级数与幂级数.canvas]]"
requires:
  - "[[高阶导数]]"
related:
  - "[[线性近似]]"
  - "[[二次近似]]"
leads_to:
  - "[[Taylor余项]]"
  - "[[Taylor级数]]"
---

# $n$ 次 Taylor 多项式是在展开中心匹配函数前 $n$ 阶导数的唯一 $n$ 次以下多项式
<!-- bilingual-en:start -->
*The degree-$n$ Taylor polynomial is the unique polynomial of degree at most $n$ matching the first $n$ derivatives at the expansion centre*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 若 $f^{(0)}(a),\ldots,f^{(n)}(a)$ 存在，则
> $$
> P_{n,a}(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k
> $$
> 是在 $a$ 满足 $P_{n,a}^{(k)}(a)=f^{(k)}(a)$、$0\le k\le n$ 的唯一次数不超过 $n$ 的多项式。
> <!-- bilingual-en:start -->
> If the derivatives through order $n$ exist at $a$, then $P_{n,a}(x)=\sum_{k=0}^n f^{(k)}(a)(x-a)^k/k!$ is the unique polynomial of degree at most $n$ whose derivatives through order $n$ match those of $f$ at $a$.
> <!-- bilingual-en:end -->

唯一性来自逐阶读取系数：常数项由函数值固定，一次项由一阶导数固定，而第 $k$ 项求 $k$ 次导后在 $a$ 留下 $k!$ 倍系数。$P_{1,a}$ 就是 [[线性近似]]，$P_{2,a}$ 就是 [[二次近似]]；一般式不是另一个孤立公式，而是同一匹配思想的延伸。
<!-- bilingual-en:start -->
Uniqueness follows by reading coefficients one derivative at a time: the value fixes the constant term, the first derivative fixes the linear term, and the $k$th derivative leaves $k!$ times the $k$th coefficient at $a$. Linear and quadratic approximations are the first two cases of this same matching construction.
<!-- bilingual-en:end -->

Taylor 多项式是一个有限对象，永远可以在相应导数存在时写出；这不等于函数在附近等于它的无限 [[Taylor级数|Taylor 级数]]。甚至无限可微也不够，见 [[光滑不代表解析]]。把 $P_{n,a}$ 当数值近似时，必须另用 [[Taylor余项]] 或其他误差估计说明遗漏部分有多大。
<!-- bilingual-en:start -->
A Taylor polynomial is finite and can be formed whenever the required derivatives exist. That does not imply that the function equals its infinite Taylor series nearby; even infinite differentiability is insufficient. Numerical use therefore needs a separate bound on the omitted remainder.
<!-- bilingual-en:end -->

> [!question]- 自检
> 写出 $e^x$ 在 $0$ 的三次 Taylor 多项式，并说清它还没有证明什么。
>
> **答案：** $1+x+x^2/2+x^3/6$；它没有单独证明 $e^x$ 等于相应无穷级数，也没有给指定 $x$ 上的误差大小。
> <!-- bilingual-en:start -->
> **Check:** Write the third-order Taylor polynomial for $e^x$ at zero and explain what it has not yet established.
>
> **Answer:** It is $1+x+x^2/2+x^3/6$. This construction alone neither proves equality with the corresponding infinite series nor bounds the error at a specified $x$.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/01_calculus/02_Applications_of_Differentiation#26d：一般 $n$ 次匹配的来源|课程记录 Session 26d]]：核对由逐阶导数匹配得到 $1/k!$ 系数的骨架。
- [[Ses98b_Lecture_Notes.pdf#page=1|MIT 18.01SC Session 98b]]：核对 Taylor 系数由中心处导数决定；本卡明确保留了讲义简写中未展开的余项边界。
