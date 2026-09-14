---
aliases:
  - "规模报酬不变只允许在指定投入口径下把 Solow 生产函数人均化"
  - "Constant returns permit Solow intensification only for the specified input measure"
  - "Solow intensive form boundary"
student_os: knowledge-atom
atom_id: MACRO-SOLOW-001
atom_set: solow-growth
atom_type: representation-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Solow 增长模型、稳态与收敛.canvas]]"
implies:
  - "[[Solow 资本积累方程]]"
related:
  - "[[Solow 的外生技术边界]]"
  - "[[Harrod 离轨不稳定性]]"
---

# 规模报酬不变只允许在指定投入口径下把 Solow 生产函数人均化
<!-- bilingual-en:start -->
*Constant returns permit Solow intensification only for the specified input measure*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 若无技术项且 $Y=F(K,L)$ 对 $K,L$ 规模报酬不变，令 $k=K/L$、$y=Y/L$，才可写成 $y=F(k,1)\equiv f(k)$。若生产函数明确为劳动增进型 $Y=F(K,AL)$，相同齐次性给出的是每单位**有效劳动**变量 $\tilde k=K/(AL)$、$\tilde y=Y/(AL)=f(\tilde k)$；此时每名劳动者产出是 $Y/L=A f(\tilde k)$。两个分母不能混用。
> <!-- bilingual-en:start -->
> Constant returns turn $Y=F(K,L)$ into a per-worker form, but turn $Y=F(K,AL)$ into a per-effective-worker form. Output per worker is then $A f(\tilde k)$, not $f(\tilde k)$. The normalization is valid only for the production inputs actually specified.
> <!-- bilingual-en:end -->

规模报酬不变的含义是，对任意 $\lambda>0$，
$$
F(\lambda K,\lambda L)=\lambda F(K,L).
$$
在无技术版本中取 $\lambda=1/L$，得到
$$
\frac{Y}{L}=F\!\left(\frac{K}{L},1\right)=f(k).
$$
在劳动增进技术版本中，齐次的两个投入是 $K$ 与 $AL$，所以应取 $\lambda=1/(AL)$。仅有规模报酬不变并不能把任意形式的技术变化都改写成 $AL$；在 Solow 的平衡增长分析中，劳动增进技术是明确采用的表示与口径。

这一步也不自动推出资本边际产出递减、正稳态存在或稳态唯一。那些结论还需要生产函数的单调性、凹性与端点条件，见 [[Solow 稳态条件]]。

> [!question]- 自检
> 在 $Y=F(K,AL)$ 中，若 $\tilde k$ 在稳态不变，为什么 $Y/L$ 仍可增长？
>
> **答案：** 因为 $Y/L=A f(\tilde k)$；$f(\tilde k)$ 不变时，外生技术水平 $A$ 仍可按 $g$ 增长。

## 来源与核验

- [[02_Economy/10_发展经济学/发展经济学拍屏ppt.pdf#page=69|发展经济学课程 PDF pp. 69–70]]：核对课程的规模报酬不变假设及从总量生产函数到人均形式的推导。
- MIT 14.452，[The Solow Growth Model, Lectures 2–3](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2b68057aa4e74410d00ae89a0c49752f_MIT14_452F16_Lec2and3.pdf)，pp. 4–5、75–79：核对 CRS 条件、劳动增进表示、每有效劳动变量与每名劳动者变量的区别。
