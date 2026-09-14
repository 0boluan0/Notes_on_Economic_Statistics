---
aliases:
  - "VaR 的单点分位权重不是普通可积光谱权重，因此不能由光谱表示自动继承一致性"
  - VaR point-mass spectral boundary
  - VaR 不是普通光谱风险度量
student_os: knowledge-atom
atom_id: RM-VAR-041
atom_set: var-es-backtesting
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
  - "[[光谱风险度量]]"
  - "[[一致风险度量]]"
related:
  - "[[VaR非次可加]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# VaR 的单点分位权重不是普通可积光谱权重，因此不能由光谱表示自动继承一致性
<!-- bilingual-en:start -->
*VaR's single-quantile weight is not an ordinary integrable spectral weight, so VaR does not inherit coherence from the spectral representation*
<!-- bilingual-en:end -->

> [!summary] “把全部权重放在一点”只是类比
> VaR 只读取 $q_\alpha(L)$。可以形式化地把它想成在 $u=\alpha$ 放置一个 Dirac 点质量，但 Dirac 点质量不是 [[光谱风险度量]] 定义所要求的普通 $L^1$ 权重函数，因此这个类比不能把 VaR 变成光谱风险度量。
>
> <!-- bilingual-en:start -->
> VaR reads only $q_\alpha(L)$. One can formally imagine a Dirac point mass at $u=\alpha$, but a Dirac mass is not the ordinary $L^1$ weight function required by [[光谱风险度量|the spectral-risk definition]]. The analogy therefore does not make VaR a spectral risk measure.
> <!-- bilingual-en:end -->

若用普通函数 $\phi$ 表示光谱，必须满足

$$
\phi\ge0,
\qquad
\int_0^1\phi(u)\,du=1,
$$

并随损失分位水平不减。任何只在单点非零的普通函数，其 Lebesgue 积分都为零，不能同时承担“全部权重集中在该点”和“总权重为一”。Dirac 测度能形式化点质量，却已经离开这里的普通可积权重函数类。

<!-- bilingual-en:start -->
An ordinary function that is non-zero only at one point has Lebesgue integral zero, so it cannot both concentrate all weight at that point and have total weight one. A Dirac measure formalizes the point mass only by leaving the class of ordinary integrable weight functions used in the spectral definition.
<!-- bilingual-en:end -->

因此，VaR 是否一致必须直接检查自身公理，而不能从“像一个极端光谱”推出。一般损失分布下，它确实可能违反次可加，见 [[VaR非次可加]]。

<!-- bilingual-en:start -->
Consequently, coherence of VaR must be checked from VaR itself, not inferred from the phrase “an extreme spectrum.” For general loss distributions, VaR can fail subadditivity; see [[VaR非次可加|VaR can fail subadditivity]].
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“在 99% 分位点放 100% 权重”不能直接证明 VaR 是光谱风险度量？
>
> **答案：** 该点质量不是定义所要求的普通可积权重函数；形式上的 Dirac 类比不携带光谱类的一致性结论。
>
> <!-- bilingual-en:start -->
> **Self-check:** Why does “put 100% weight at the 99th percentile” not prove that VaR is a spectral risk measure?
>
> **Answer:** The point mass is not an ordinary integrable weight function of the required class, so the Dirac analogy does not carry the coherence result for spectral measures.
> <!-- bilingual-en:end -->

## 来源与核验

- [Acerbi, *Spectral Measures of Risk: A Coherent Representation of Subjective Risk Aversion*](https://doi.org/10.1016/S0378-4266(02)00281-9)：核对光谱风险度量的可积权重条件与一致性结论。
- [Artzner et al., *Coherent Measures of Risk*](https://doi.org/10.1111/1467-9965.00068)：核对一致性公理及 VaR 的次可加边界。
