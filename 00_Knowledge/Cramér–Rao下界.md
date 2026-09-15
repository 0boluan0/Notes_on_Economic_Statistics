---
student_os: knowledge-atom
atom_id: 6b0b2784-4c54-4860-816e-01b7dda7f4bb
status: source-checked
aliases:
  - "Cramér–Rao 下界在正则参数模型中限制无偏估计量可以达到的方差"
  - "The Cramér–Rao bound limits the variance of unbiased estimators in regular parametric models"
  - "Cramer–Rao bound"
  - "Cramér–Rao bound"
---

# Cramér–Rao 下界在正则参数模型中限制无偏估计量可以达到的方差

<!-- bilingual-en:start -->
*The Cramér–Rao bound limits the variance of unbiased estimators in regular parametric models*
<!-- bilingual-en:end -->

> [!summary] 核心
> Cramér–Rao 下界是一类信息不等式：在满足相应正则条件的参数模型中，无偏估计量的方差不能低于模型给出的下界。使用它需要检查模型、目标与正则条件，不能只凭“无偏”两字套用。
>
> <!-- bilingual-en:start -->
> The Cramér–Rao information inequality gives a lower variance bound for unbiased estimators under suitable regularity conditions. Applying it requires checking the model, target, and conditions.
> <!-- bilingual-en:end -->
^core

Lecture 1 只用它说明[[估计量有效性|有效性]]需要明确比较条件：与 BLUE 的线性无偏类别不同，这里的边界来自正则参数模型。下界存在不代表必有估计量达到，也不直接比较有偏估计量的 MSE。

<!-- bilingual-en:start -->
Lecture 1 uses it to illustrate why [[估计量有效性|efficiency]] needs explicit conditions. Its parametric information bound differs from the linear unbiased comparison defining BLUE. A bound need not be attained and does not by itself compare biased estimators’ MSE.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*PSI Lecture 1 — Foundations*（slides 58，PDF p.58）：支持本条在课程中的定义、条件与用途。

<!-- bilingual-en:start -->
*The EC400 lecture supplies the course context and notation. Additional cited references support the stated definitions, assumptions, or boundaries; worked arithmetic and direct implications are checked explicitly.*
<!-- bilingual-en:end -->
