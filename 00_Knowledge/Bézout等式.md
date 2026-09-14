---
aliases:
  - "两个不全为零的整数的最大公因数可写成它们的整数线性组合"
  - 裴蜀等式
  - Bézout identity
student_os: knowledge-atom
atom_id: MCS-NT-005
atom_type: theorem
status: source-checked
part_of:
  - "[[数论与RSA.canvas]]"
requires:
  - "[[最大公因数]]"
  - "[[欧几里得算法]]"
related:
  - "[[扩展欧几里得算法]]"
leads_to:
  - "[[模逆元存在条件]]"
  - "[[素数整除乘积]]"
---

# 两个不全为零的整数的最大公因数可写成它们的整数线性组合
<!-- bilingual-en:start -->
*The gcd of two integers not both zero is an integer linear combination of them*
<!-- bilingual-en:end -->

若整数 $a,b$ 不全为零，则存在整数 $s,t$ 使
<!-- bilingual-en:start -->
For integers $a,b$ not both zero, there are integers $s,t$ such that
<!-- bilingual-en:end -->

$$sa+tb=\gcd(a,b).$$

“整数线性组合”就是只允许用整数系数加减这两个数。这里最重要的是：不仅它们的共同因子能整除所有这样的组合，**gcd 本身也确实能组合出来**。例如 $\gcd(30,22)=2$，且 $2=3\cdot30-4\cdot22$。
<!-- bilingual-en:start -->
An integer linear combination uses integer coefficients to add and subtract the two inputs. The important point is not merely that common divisors divide every such combination: **the gcd itself is attainable as a combination**. For example, $\gcd(30,22)=2$ and $2=3\cdot30-4\cdot22$.
<!-- bilingual-en:end -->

## 为什么成立
<!-- bilingual-en:start -->
*Why it holds*
<!-- bilingual-en:end -->

在[[欧几里得算法]]中，初始两个数已经分别是 $1a+0b$、$0a+1b$。以后每个余数都是前两个数的“一个减去另一个的整数倍”，所以仍是 $a,b$ 的整数线性组合。算法结束时最后一个非零数就是 gcd，因此它也有这种表示。负输入先取绝对值，最后把符号吸收到系数中；一个输入为零时结论直接成立。
<!-- bilingual-en:start -->
In the [[欧几里得算法|Euclidean algorithm]], the initial values are $1a+0b$ and $0a+1b$. Each later remainder subtracts an integer multiple of one previous value from the other, preserving representation as an integer linear combination of $a,b$. The last nonzero value is the gcd, so it has that representation. For negative inputs, take absolute values and absorb signs into the coefficients; a single zero input is immediate.
<!-- bilingual-en:end -->

等式保证系数存在，不保证唯一；具体计算交给[[扩展欧几里得算法]]。当 $\gcd(a,n)=1$ 时，$sa+tn=1$ 在模 $n$ 下留下 $sa\equiv1$，这就是[[模逆元存在条件]]中“互素足以求逆”的理由。
<!-- bilingual-en:start -->
The identity guarantees existence, not uniqueness, of the coefficients; the [[扩展欧几里得算法|extended Euclidean algorithm]] computes them. When $\gcd(a,n)=1$, reducing $sa+tn=1$ modulo $n$ leaves $sa\equiv1$, which is why coprimality suffices in the [[模逆元存在条件|criterion for an inverse]].
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/99_Books/MIT6_042JS15_textbook.pdf#page=259|MIT Mathematics for Computer Science §8.2.2，PDF pp.259–261]]：核验定理 8.2.2 及余数系数保持的构造证明；正文把存在性与求系数的操作分开，例式直接复算。
  <!-- bilingual-en:start -->
  Section 8.2.2 verifies Theorem 8.2.2 and its constructive proof through remainder coefficients. The text separates existence from the coefficient-computation procedure and checks the example identity directly.
  <!-- bilingual-en:end -->
