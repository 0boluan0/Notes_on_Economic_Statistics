---
aliases:
  - "满行秩线性约束把均值向量检验降为较低维 Hotelling T²"
  - Full-row-rank linear hypotheses reduce to lower-dimensional Hotelling T-squared
  - 线性变换均值向量的 Hotelling T²
  - D mu 的多元均值检验
student_os: knowledge-atom
atom_id: STAT-HOT-003
atom_set: hotelling-mean-inference
atom_type: reduction-theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[单样本Hotelling T²]]"
  - "[[Gaussian仿射闭包]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
related:
  - "[[Hotelling投影极值]]"
  - "[[配对Hotelling T²]]"
  - "[[pooled Hotelling T²]]"
---

# 满行秩线性约束把均值向量检验降为较低维 Hotelling T²
<!-- bilingual-en:start -->
*A full-row-rank linear hypothesis reduces mean-vector inference to a lower-dimensional Hotelling test*
<!-- bilingual-en:end -->

> [!summary] 原子结论
> 设 $X_1,\ldots,X_n\overset{iid}{\sim}N_p(\mu,\Sigma)$，其中 $\Sigma\succ0$。令 $D$ 为固定的 $m\times p$ 满行秩矩阵，并检验
> $$H_0:D\mu=\psi_0.$$
> 因为 $Y_i=DX_i\sim N_m(D\mu,D\Sigma D^T)$，所以
> $$T_D^2=n(D\bar X-\psi_0)^T(DSD^T)^{-1}(D\bar X-\psi_0),$$
> 且当 $n>m$ 时
> $$\frac{n-m}{m(n-1)}T_D^2\sim F_{m,n-m}$$
> 在 $H_0$ 下精确成立。
> <!-- bilingual-en:start -->
> Transform each observation first and apply the ordinary $m$-dimensional one-sample Hotelling theorem. Only $n>m$ is needed for this transformed test: its calibration uses the rank $m$ of the restrictions, not the ambient dimension $p$.
> <!-- bilingual-en:end -->

例如检验四个分量均值相等，可取
$$
D=\begin{bmatrix}
1&0&0&-1\\
0&1&0&-1\\
0&0&1&-1
\end{bmatrix},\qquad \psi_0=0,
$$
于是只在三个独立差异方向上检验，而不是检验整个四维均值是否等于某个完全指定向量。

若用任意可逆 $m\times m$ 矩阵 $A$ 把同一组约束改写为 $AD\mu=A\psi_0$，$T_D^2$ 不变；统计量依赖约束的行空间，而不依赖具体写法。

> [!warning] 边界
> - 这里只需 $n>m$；即使 $p\ge n$ 导致原始 $S$ 奇异，只要固定的 $D$ 满行秩，$DSD^T$ 在上述正态正定模型下仍几乎必然可逆。这个降维检验不需要先求 $S^{-1}$。
> - 若 $D$ 的实际秩为 $r<m$，则 $DSD^T$ 奇异。先检查 $\psi_0$ 是否与冗余约束一致；若原假设非空，就换成行空间的一组基及相应右端，要求 $n>r$，并用 $r$ 与 $F_{r,n-r}$ 校准，不能继续把行数 $m$ 当自由度。
> - $D$ 与约束 family 必须在看见用于检验的数据前固定。先搜索许多 $D$ 再只报告最显著者，会改变第一类错误率。
> - 这张卡检验的是 $D\mu$；它不自动给出未包含在 $D$ 行空间中的均值方向结论。

> [!question]- 自检
> 一个 $5\times8$ 的约束矩阵只有秩 3。精确 F 校准的分子自由度应取 5 还是 3？
>
> **答案：** 取 3。只有三个线性独立的被检验方向；先用行空间的基去掉冗余约束。

## 来源与核验

- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.3. 线性变换下的均值向量检验|多元统计课程 §1.3]]：核对 $D\mu$、$DSD^T$、维数 $m$ 与精确 F 转换。
- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.3. 参数线性变换检验|多元统计课程 §1.3]]：交叉核对均值差问题中的相同化约。
- [NIST/SEMATECH e-Handbook, Hotelling's T-squared](https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm)：独立核对应用于 $Y_i=DX_i$ 后的 $m$ 维单样本精确 F 校准。
- [Penn State STAT 505, Lesson 7](https://online.stat.psu.edu/stat505/Lesson07)：核对 profile/linear-combination 推断必须明确预先指定的比较方向。
