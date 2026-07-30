---
aliases: [RSA Cryptosystem, RSA, RSA Public-Key Encryption, RSA 公钥密码]
tags: [concept, discrete-mathematics, cryptography]
---
# RSA Cryptosystem

RSA 取不同素数 $p,q$，令 $N=pq$、$\varphi(N)=(p-1)(q-1)$。选择与 $\varphi(N)$ 互素的公开指数 $e$，并求 $d$ 使
$$
ed\equiv1\pmod{\varphi(N)}.
$$
公钥为 $(N,e)$，私钥为 $d$；消息 $m$ 加密为 $c\equiv m^e\pmod N$，解密为 $m\equiv c^d\pmod N$。

本卡只描述数学结构，不讨论 padding、侧信道和现代实现安全性。

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
