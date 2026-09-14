---
aliases:
  - "AR 与 MA 多项式有公共因子时表示非最小且参数不可识别"
  - Common ARMA factors
  - Minimal ARMA representation
  - Coprime ARMA polynomials
  - ARMA 公共因子
student_os: knowledge-atom
atom_id: TS-ARMA-008
atom_set: arma-modeling
atom_type: identification-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARMA(p,q)模型]]"
related:
  - "[[MA可逆性]]"
  - "[[ARMA信息准则]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# AR 与 MA 多项式有公共因子时表示非最小且参数不可识别
<!-- bilingual-en:start -->
*Common AR and MA factors make an ARMA representation non-minimal and its parameters unidentified*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 在通常的因果、可逆 ARMA 参数域内，若 $\phi(B)$ 与 $\theta(B)$ 有非平凡公共因子 $c(B)$，则
> $$c(B)a(B)(y_t-\mu)=c(B)b(B)\varepsilon_t$$
> 可约成 $a(B)(y_t-\mu)=b(B)\varepsilon_t$。约分前的较高阶参数并不代表额外动态，而是同一过程的冗余表示。
> <!-- bilingual-en:start -->
> Within the usual causal-invertible ARMA parameterisation, a nonconstant common AR/MA factor can be cancelled. The higher-order representation is therefore non-minimal, and its extra parameters do not identify additional dynamics.
> <!-- bilingual-en:end -->

最直观的例子是
$$(1-aB)y_t=(1-aB)\varepsilon_t.$$
在约分后的表示仍属于所声明的解类、且没有用约分抹掉初值或边界解信息时，这个所谓 ARMA(1,1) 其实只是 $y_t=\varepsilon_t$。用样本估计未约分形式会出现近抵消、平坦似然、巨大标准误或非常不稳定的预测解释。

所以一个标准 ARMA 身份通常要求 $\phi$ 与 $\theta$ 互素。因果性检查 AR 根，可逆性检查 MA 根，而**最小性**检查二者有没有共同根；三项回答不同问题，不能互相替代。接近但不完全相同的根虽可识别，却仍可能造成弱识别与数值不稳定。
<!-- bilingual-en:start -->
For example, $(1-aB)y_t=(1-aB)\varepsilon_t$ reduces to white noise when cancellation is admissible in the stated solution class and does not discard boundary or initial-condition information. Causality, invertibility, and minimality are separate checks: the first concerns AR roots, the second MA roots, and the third common roots. Near cancellation can remain formally identified while producing weak identification and unstable estimates.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个 AR 根与 MA 根都在单位圆外，是否已经保证 ARMA 参数可识别？
>
> **答案：** 没有。两组根还必须没有公共因子；否则因果、可逆但非最小。

## 来源与核验

- [MIT OCW 14.384, Recitation 1](https://ocw.mit.edu/courses/14-384-time-series-analysis-fall-2013/ca390a7534c2594b397af2164697352b_MIT14_384F13_rec1.pdf)：明确要求 AR 与 MA 多项式 relatively prime，并说明 ARMA 表示一般不唯一。
- [MIT OCW 18.S096, Lecture 8](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/1926c83ecd7ea700f7cb63914c6d7c0f_MIT18_S096F13_lecnote8.pdf)：核对多项式约分与最小表示。
