---
aliases:
  - "两个整数相除会向零截断且事后转换不能恢复小数部分"
  - C integer division
  - 整数除法向零截断
student_os: knowledge-atom
atom_id: CS-C-006
atom_set: c-foundations
atom_type: rule
status: source-checked
mastery_state: unassessed
requires:
  - "[[C 整数范围与宽度]]"
part_of:
  - "[[C 语言基础：编译、类型、控制流与函数.canvas]]"
---

# 两个整数相除会向零截断且事后转换不能恢复小数部分
<!-- bilingual-en:start -->
*Division of two integers truncates toward zero, and conversion afterward cannot recover the fractional part*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> `/` 的结果由**运算发生时的操作数类型**决定。两个操作数都是整数时，商的小数部分向零截断：`7 / 3 == 2`，`-7 / 3 == -2`。若需要实数商，至少一个操作数必须在除法前转换为浮点类型。
> <!-- bilingual-en:start -->
>
> &nbsp;
> The result of `/` is determined by the operand types at the time of the operation. With two integer operands, the fractional part of the quotient is discarded toward zero: `7 / 3 == 2` and `-7 / 3 == -2`. To obtain a floating-point quotient, convert at least one operand before division.
> <!-- bilingual-en:end -->

```c
int total = 5;
int n = 2;

double wrong = total / n;          // 2 先产生，再变成 2.0
double right = (double) total / n; // 2.5
```

目标变量是 `double` 并不会反向改变右侧表达式的求值方式。`(double)(total / n)` 也只是把已经截断的 `2` 转成 `2.0`。转换必须作用在除法的某个操作数上，才能让通常算术转换在运算前把另一方也带到浮点类型。
<!-- bilingual-en:start -->
A `double` destination does not reach backward and change how the right-hand expression is evaluated. `(double)(total / n)` likewise converts an already truncated `2` into `2.0`. The conversion must apply to an operand so that the usual arithmetic conversions bring the other operand to a floating type before division.
<!-- bilingual-en:end -->

## 余数与负数

对可表示的整数除法，`a / b` 与 `a % b` 满足
$$a=(a/b)b+a\%b.$$
商向零截断，因此余数为非零时与被除数 `a` 同号。不能把 C 的负数 `%` 直接当作数学中总为非负的模运算。
<!-- bilingual-en:start -->
For representable integer division, `a / b` and `a % b` satisfy $a=(a/b)b+a\%b$. Because the quotient truncates toward zero, a nonzero remainder has the sign of the dividend `a`. C's `%` on negative operands is therefore not automatically the nonnegative mathematical modulo operation.
<!-- bilingual-en:end -->

## 边界

除数为零时行为未定义。只要代数商不能由通常算术转换后的结果类型表示，`a / b` 与 `a % b` **两者**的行为都未定义。C23 已要求标准有符号整数采用二进制补码，因此 `INT_MIN / -1` 与 `INT_MIN % -1` 是典型实例；C17 及更早版本还允许正负范围对称的表示，所以跨版本表述应以“商是否可表示”为根本条件，而不是只背这一对操作数。必须在运算前验证除数和可表示范围。
<!-- bilingual-en:start -->
Division by zero has undefined behaviour. Whenever the algebraic quotient is not representable in the result type selected by the usual arithmetic conversions, **both** `a / b` and `a % b` have undefined behaviour. C23 requires the standard signed integer types to use two's-complement representation, making `INT_MIN / -1` and `INT_MIN % -1` canonical examples. C17 and earlier revisions also permitted representations with symmetric positive and negative ranges, so the version-independent rule is representability of the quotient, not memorising one operand pair. Validate both the divisor and the representable range before the operation.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 `double ratio = (double)(wins / games);` 在 `wins=3, games=4` 时仍得到 `0.0`？
>
> **答案：** 括号内先执行整数除法并得到 `0`，外层转换只能得到 `0.0`。应写 `(double) wins / games`。

## 来源与核验

- [ISO C11 committee draft N1570, 6.3.1.8 and 6.5.5](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对通常算术转换、向零截断、余数恒等式与除零边界。
- [WG14 N3096, C23 draft, 6.2.6.2 and 6.5.5](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3096.pdf)：核对 C23 的二进制补码要求，以及商不可表示时 `/` 与 `%` 都未定义。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.3.1.8 and 6.5.5](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for the usual arithmetic conversions, truncation toward zero, the quotient–remainder identity, and the division-by-zero boundary.
- [WG14 N3096, C23 draft, 6.2.6.2 and 6.5.5](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3096.pdf) was checked for C23's two's-complement requirement and for undefined behaviour of both `/` and `%` when the quotient is not representable.
<!-- bilingual-en:end -->
