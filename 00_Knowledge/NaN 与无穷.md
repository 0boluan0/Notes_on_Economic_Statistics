---
aliases:
  - "NaN 与无穷是带专门传播和比较规则的特殊值"
  - NaN and infinity are special values with dedicated propagation and comparison rules
  - NaN 与 Infinity
student_os: knowledge-atom
atom_id: CS-FP-012
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[浮点数与数值误差.canvas]]"
requires:
  - "[[浮点溢出与下溢]]"
leads_to:
  - "[[浮点容差比较]]"
---

# NaN 与无穷是带专门传播和比较规则的特殊值

<!-- bilingual-en:start -->
*NaN and infinity are special values with dedicated propagation and comparison rules*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> `+inf` 与 `-inf` 表示带方向的无穷特殊值，不是“非常大的有限数”；NaN（not a number）表示没有可用实数或无穷结果的特殊状态。它们占用浮点编码，但遵守专门算术与比较规则，不能作为普通有限观测继续统计。
>
> <!-- bilingual-en:start -->
> `+inf` and `-inf` are directional infinity values, not merely very large finite numbers. NaN (not a number) is a special state for a result with no usable real or infinite value. They occupy floating-point encodings but obey dedicated arithmetic and comparison rules and should not silently enter ordinary finite-data analysis.
> <!-- bilingual-en:end -->

IEEE 754 中，溢出在默认舍入情形下可产生无穷；`0/0`、`inf - inf` 等无定义形式产生 NaN。NaN 通常沿后续运算传播，而且所有有序比较都失败；在 Python 中甚至 `nan == nan` 也是 `False`。无穷可参与一些定义良好的运算，但 `inf - inf` 仍为 NaN。
<!-- bilingual-en:start -->
Under IEEE 754, overflow under default rounding can produce infinity, while indeterminate forms such as `0/0` or `inf - inf` produce NaN. NaN generally propagates and fails ordered comparisons; in Python, even `nan == nan` is `False`. Infinity can participate in some defined operations, but `inf - inf` remains NaN.
<!-- bilingual-en:end -->

在 Python 中应用 `math.isfinite(x)`、`math.isinf(x)` 和 `math.isnan(x)` 检查状态，不用 `x == math.nan`。同时要区分标准层与 API 层：部分 `math` 函数会对无效输入或溢出抛异常，而不是总返回 NaN/inf。
<!-- bilingual-en:start -->
In Python, inspect state with `math.isfinite(x)`, `math.isinf(x)`, and `math.isnan(x)`, not `x == math.nan`. Also distinguish the standard from the API layer: some `math` functions raise exceptions for invalid input or overflow instead of always returning NaN or infinity.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么 `x != x` 可以提示 NaN，却不如 `math.isnan(x)` 作为接口清晰？
>
> **答案：** NaN 的确不等于自身，但 `math.isnan` 直接表达要检查的特殊状态，也避免把异常比较规则误当普通不等关系。

## 来源与核验

- MIT 18.335J, [*Lecture 2: Floating-Point Arithmetic, the IEEE Standard*](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/2f313023ae3404bc217a81a31b227170_MIT18_335JS19_lec2.pdf)：核验 IEEE 特殊量、无穷运算与 NaN 产生条件。
- David Goldberg, [*What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://doi.org/10.1145/103162.103163)，§3.2.1–3.2.2：核验 NaN 传播、无穷和无定义形式。
- Python Software Foundation, [`math.isnan`, `math.isinf`, `math.isfinite`](https://docs.python.org/3/library/math.html)：核验 Python 的检测接口、NaN 比较与异常行为边界。
