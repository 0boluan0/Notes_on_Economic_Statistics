---
aliases:
  - Radix-2 FFT
  - Cooley-Tukey FFT
  - radix-2 fast Fourier transform
  - 二进制快速傅里叶变换
  - 基 2 FFT
tags:
  - 线性代数
  - procedure
type: procedure
---

# Radix-2 FFT

## 何时使用

输入长度 $n=2^m$ 的序列 $x_0,\ldots,x_{n-1}$，输出离散 Fourier 变换

$$
X_k=\sum_{j=0}^{n-1}x_j\omega_n^{jk},
\qquad \omega_n=e^{-2\pi i/n},
$$

并把直接计算的 $O(n^2)$ 代价降为 $O(n\log n)$。

## Step 1. 按偶、奇下标拆分

令 $e_j=x_{2j}$、$o_j=x_{2j+1}$，分别计算长度 $n/2$ 的 DFT：$E_k,O_k$。

## Step 2. 使用 butterfly 合并

对 $k=0,\ldots,n/2-1$，

$$
X_k=E_k+\omega_n^kO_k,
$$

$$
X_{k+n/2}=E_k-\omega_n^kO_k.
$$

## Step 3. 递归到底层

对偶序列、奇序列重复拆分，直到长度为 $1$；长度为 $1$ 时 DFT 就是原值。

## Step 4. 检查约定

- 明确指数采用 $-2\pi i/n$ 还是 $+2\pi i/n$；
- 明确逆变换的 $1/n$ 归一化放在哪里；
- 用 $n=2$ 或 $n=4$ 的直接 DFT 验证 butterfly 顺序和 twiddle factor。

## 关联卡片

- [[Fourier Series]]
- [[Unitary Matrix]]
- [[Change of Basis]]

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
