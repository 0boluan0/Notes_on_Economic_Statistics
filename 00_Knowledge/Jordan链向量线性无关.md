---
aliases:
  - 同一条 Jordan 链中的向量必然线性无关
  - Linear independence of vectors in a Jordan chain
  - Jordan 链向量线性无关
student_os: knowledge-atom
atom_id: LA-EIG-045
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Jordan链]]"
  - "[[线性无关]]"
related:
  - "[[Jordan块]]"
  - "[[Jordan标准形]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 同一条 Jordan 链中的向量必然线性无关
<!-- bilingual-en:start -->
*The vectors in a single Jordan chain are necessarily linearly independent*
<!-- bilingual-en:end -->

> [!summary] 核心定理
> 若非零向量 $v_1,\ldots,v_r$ 满足
> $$(A-\lambda I)v_1=0,
> \qquad
> (A-\lambda I)v_{j+1}=v_j\quad(1\le j<r),$$
> 则 $v_1,\ldots,v_r$ 线性无关。
>
> <!-- bilingual-en:start -->
> Every sequence satisfying the Jordan-chain recurrence is linearly independent.
> <!-- bilingual-en:end -->

令 $N=A-\lambda I$，并假设
$$
c_1v_1+\cdots+c_rv_r=0.
$$
对等式施加 $N^{r-1}$。由于 $N^{r-1}v_j=0$ 对 $j<r$ 成立，而 $N^{r-1}v_r=v_1\ne0$，可得 $c_rv_1=0$，所以 $c_r=0$。删去这一项后，对剩余关系施加 $N^{r-2}$，得到 $c_{r-1}=0$。如此从链尾逐级向前，最终有
$$
c_1=\cdots=c_r=0.
$$

这个结论说明一条长度为 $r$ 的链确实提供 $r$ 个独立方向，因此可作为对应[[Jordan块]]的基。它并不自动说明若干条任意选取的链合在一起仍线性无关；要组成全空间的 Jordan 基，还需要按广义特征空间的结构选择整组链。
<!-- bilingual-en:start -->
The proof isolates coefficients from the tail of the chain by applying descending powers of $A-\lambda I$. A chain of length $r$ therefore supplies $r$ independent directions, but arbitrary chains cannot be pooled without also checking the generalized-eigenspace decomposition.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 对关系 $c_1v_1+c_2v_2+c_3v_3=0$ 施加 $(A-\lambda I)^2$，首先能确定哪个系数？
>
> **答案：** 首先得到 $c_3v_1=0$，所以 $c_3=0$。
>
> <!-- bilingual-en:start -->
> Apply $(A-\lambda I)^2$ to $c_1v_1+c_2v_2+c_3v_3=0$. Which coefficient is determined first?
>
> **Answer:** The result is $c_3v_1=0$, so $c_3=0$.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|MIT 18.06SC Session 3.4 summary]]：核对 Jordan 链递推结构与一条链对应一个 Jordan 块；线性无关性由正文对递推式逐次施加 $A-\lambda I$ 直接验证。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.4.4 Jordan链与Jordan标准形|课程 3.4.4]]：核对链方程与链向量在 Jordan 块基中的排列。
<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|MIT 18.06SC Session 3.4 summary]] was checked for the Jordan-chain recurrence and the correspondence between one chain and one Jordan block; linear independence is then verified directly by applying successive powers of $A-\lambda I$.
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.4.4 Jordan链与Jordan标准形|Course Section 3.4.4]] was checked for the chain equations and the ordering of chain vectors in a Jordan-block basis.
<!-- bilingual-en:end -->
