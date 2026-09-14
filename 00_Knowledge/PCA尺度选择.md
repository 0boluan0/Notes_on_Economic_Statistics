---
aliases:
  - "协方差 PCA 与相关矩阵 PCA 采用不同尺度度量因而回答不同问题"
  - covariance versus correlation PCA
  - PCA 标准化选择
student_os: knowledge-atom
atom_id: STAT-PCA-007
atom_set: principal-component-analysis
atom_type: modeling-choice
status: source-checked
mastery_state: unassessed
part_of:
  - "[[主成分分析.canvas|主成分分析]]"
requires:
  - "[[主成分分析]]"
  - "[[相关矩阵]]"
related:
  - "[[解释方差比]]"
  - "[[PCA验证与泄漏]]"
  - "[[相关矩阵尺度不变性]]"
  - "[[响应尺度与线性变换]]"
---

# 协方差 PCA 与相关矩阵 PCA 采用不同尺度度量因而回答不同问题
<!-- bilingual-en:start -->
*Covariance PCA and correlation PCA use different scale metrics and therefore answer different questions*
<!-- bilingual-en:end -->

> [!summary] 原子选择
> 原变量先中心化后对协方差矩阵做 PCA，会保留原始单位与方差权重；先按各变量标准差缩放，再做 PCA，等价于对相关矩阵做 PCA，使每个变量的边际方差都为 1。二者不是同一结果的显示选项。
> <!-- bilingual-en:start -->
> Covariance PCA retains original variance scales. Standardising each variable first is equivalent to correlation PCA and gives every variable unit marginal variance.
> <!-- bilingual-en:end -->

若单位只是任意换算，例如米与厘米不应改变科学结论，相关矩阵 PCA 往往更可辩护。若原始波动规模本身就是问题的一部分，例如同一单位下各期限风险因子的实际变动幅度，标准化会主动改变权重，未必合适。

标准化并不自动“更公平”：它会放大小方差变量，也可能放大测量噪声。选择前应说明哪些尺度差异有实质含义、哪些只是量纲，并在新数据上沿用训练期的均值和标准差。

相关矩阵 PCA 要求每个纳入变量的标准差为正；常量变量无法标准化，应先识别并移除或另行处理。若总体方差或样本方差因缺失处理、权重或估计口径而改变，也必须把这些选择算作 PCA 定义的一部分。

> [!question]- 自检
> 把某变量从米改成厘米后，协方差 PCA 是否保证方向不变？
>
> **答案：** 不保证；该变量的方差会放大 $10^4$ 倍并可能主导结果。相关矩阵 PCA 对这种正尺度换算保持不变。

## 来源与核验

- [Penn State STAT 505, Lesson 11](https://online.stat.psu.edu/stat505/Lesson11)：核对标准化数据的协方差矩阵等于原数据相关矩阵，以及两种 PCA 的操作区别。
- [scikit-learn, PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)：核对常规 PCA 默认中心化但不自动缩放。
- [[响应尺度与线性变换]]：核对尺度变换会改变具体降维方向的边界。
