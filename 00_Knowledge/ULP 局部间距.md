---
aliases:
  - "ULP 描述某一尺度附近相邻浮点数的局部间距"
  - ULP describes the local spacing between floating-point values at a given scale
  - Unit in the last place
  - 浮点数的局部间距
student_os: knowledge-atom
atom_id: CS-FP-005
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[浮点数与数值误差.canvas]]"
requires:
  - "[[浮点舍入]]"
leads_to:
  - "[[绝对误差与相对误差]]"
  - "[[浮点容差比较]]"
---

# ULP 描述某一尺度附近相邻浮点数的局部间距

<!-- bilingual-en:start -->
*ULP describes the local spacing between floating-point values at a given scale*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> ULP（unit in the last place，末位单位）描述给定浮点值所在尺度上，最低有效位对应的数值间距。对不是最大有限值的正有限 Python `float`，`math.ulp(x)` 给出从 `x` 向上到下一个可表示值的距离；因此 ULP 是局部量，不是整个格式通用的固定小数。零、最大有限值、无穷与 NaN 由接口另行规定。
>
> <!-- bilingual-en:start -->
> An ULP (unit in the last place) is the numerical spacing associated with the least significant stored digit at a value's scale. For a positive finite Python `float` other than the largest finite value, `math.ulp(x)` gives the distance from `x` to the next representable value above it. Zero, the largest finite value, infinity, and NaN have separately documented interface rules. ULP is therefore local, not one fixed decimal increment for the whole format.
> <!-- bilingual-en:end -->

对规格化二进制浮点数，有效数位数固定而指数随尺度变化，所以相邻数间距大体随 $|x|$ 增大。`1e16` 附近的间距可能大于 `1.0`；把 `1.0` 加到 `1e16` 上，精确和可能仍舍入回 `1e16`。这不是加法忽略了操作数，而是目标格式在该尺度上没有足够密的点。
<!-- bilingual-en:start -->
For normal binary floating-point values, significand precision is fixed while the exponent changes with scale, so adjacent spacing grows roughly with $|x|$. Near `1e16`, the spacing can exceed `1.0`; adding `1.0` may round back to `1e16`. The addition did not ignore an operand—the destination format simply has no sufficiently close point at that scale.
<!-- bilingual-en:end -->

ULP 与 machine epsilon 不能互换。machine epsilon 描述 1 附近的相对分辨率约定；`ulp(x)` 随 $x$ 改变。在零和次正规区间，间距规则还有专门边界，不能从普通规格化区间直接外推。
<!-- bilingual-en:start -->
ULP and machine epsilon are not interchangeable. Machine epsilon is a convention for relative resolution near 1, whereas `ulp(x)` changes with $x$. Zero and the subnormal region have special spacing rules and cannot be extrapolated directly from the normal range.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么一个固定的 `1e-12` 不能代表所有浮点数的“一格”？
>
> **答案：** 浮点间距随指数和尺度变化；不同量级的值有不同 ULP，零和次正规区间还采用特殊间距。

## 来源与核验

- Python Software Foundation, [`math.ulp`](https://docs.python.org/3/library/math.html#math.ulp) 与 [`math.nextafter`](https://docs.python.org/3/library/math.html#math.nextafter)：核验 Python 对普通值、零、最大值、无穷与 NaN 的 ULP 接口语义。
- MIT 18.335J, [*Lecture 2: Floating-Point Arithmetic, the IEEE Standard*](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/2f313023ae3404bc217a81a31b227170_MIT18_335JS19_lec2.pdf)：核验相邻数间距随尺度变化。
- David Goldberg, [*What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://doi.org/10.1145/103162.103163)，§2.2：核验 ULP 与相对误差的关系和边界。
