---
aliases: [Euler Totient Theorem Proof, 欧拉定理证明]
tags: [proof, discrete-mathematics, number-theory]
type: proof
---
# Euler Totient Theorem Proof

## 假设

$\gcd(a,n)=1$；$r_1,\ldots,r_{\varphi(n)}$ 是模 $n$ 的全部可逆剩余类。

## 推导

乘以 $a$ 会把这些可逆类置换：若 $ar_i\equiv ar_j\pmod n$，可约去可逆的 $a$ 得 $r_i\equiv r_j$。因此
$$
\prod_i ar_i\equiv\prod_i r_i\pmod n.
$$
左边是 $a^{\varphi(n)}\prod_i r_i$。因乘积 $\prod_i r_i$ 也可逆，可约去它，得到
$$
a^{\varphi(n)}\equiv1\pmod n.
$$

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
