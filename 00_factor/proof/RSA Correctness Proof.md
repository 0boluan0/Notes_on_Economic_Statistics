---
aliases: [RSA Correctness Proof, RSA 正确性证明]
tags: [proof, discrete-mathematics, cryptography]
type: proof
---
# RSA Correctness Proof

设 $p,q$ 是不同素数，$N=pq$，且正整数 $e,d$ 满足 $ed=1+k\varphi(N)$。要证对每个整数消息 $m$ 都有 $m^{ed}\equiv m\pmod N$。

分别模 $p$ 与模 $q$：若 $p\mid m$，两边都为 $0$；若 $p\nmid m$，由 Fermat 小定理和 $p-1\mid\varphi(N)$ 得 $m^{k\varphi(N)}\equiv1\pmod p$，所以 $m^{ed}\equiv m\pmod p$。模 $q$ 同理。

因为 $p,q$ 互素，同时模 $p,q$ 相等推出模 $pq=N$ 相等。因此解密恢复原消息的剩余类。

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
