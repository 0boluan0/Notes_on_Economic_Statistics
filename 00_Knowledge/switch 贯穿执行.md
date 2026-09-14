---
aliases:
  - "switch 从匹配的 case 标签进入并继续执行直到离开 switch 语句"
  - C switch fallthrough
  - switch case 贯穿
student_os: knowledge-atom
atom_id: CS-C-009
atom_set: c-foundations
atom_type: control-flow
status: source-checked
mastery_state: unassessed
part_of:
  - "[[C 语言基础：编译、类型、控制流与函数.canvas]]"
---

# switch 从匹配的 case 标签进入并继续执行直到离开 switch 语句
<!-- bilingual-en:start -->
*A switch enters at the matching case label and continues until control leaves the switch statement*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> `switch` 先计算一次整型控制表达式并对它做整数提升，再把各 `case` 常量转换到这个提升后的类型后匹配；没有匹配时跳到 `default`，若也没有 `default` 就跳过整个语句。`case` 是入口标签，不是自动封闭的分支，因此执行会自然落入后续标签，直到控制流离开 `switch` 或到达其末尾。
> <!-- bilingual-en:start -->
>
> &nbsp;
> A `switch` evaluates its integer controlling expression once and applies the integer promotions, then converts each `case` constant to that promoted type for matching. If none matches, control transfers to `default`, or skips the whole statement if no default exists. A `case` is an entry label, not an automatically enclosed branch, so execution falls through later labels until control leaves the `switch` or reaches its end.
> <!-- bilingual-en:end -->

```c
switch (grade) {
case 'A':
case 'B':
    puts("pass with distinction");
    break;
case 'C':
    puts("pass");
    break;
default:
    puts("review input");
}
```

这里 `'A'` 有意贯穿到 `'B'`，让两个输入共享同一动作。若忘记 `break`，后一个动作也会执行；所以每次贯穿都应从业务含义解释，而不应只因为语法允许。编译器的隐式贯穿警告可以捕捉许多遗漏，但不能替代人对意图的判断。
<!-- bilingual-en:start -->
Here `'A'` intentionally falls through to `'B'` so both inputs share one action. A forgotten `break` would also execute the next action, so every fallthrough needs a semantic reason rather than mere syntactic permission. Compiler fallthrough warnings catch many omissions but cannot decide the intended behaviour for the programmer.
<!-- bilingual-en:end -->

常见离开方式包括 `break`、`return` 和跳到外部标签的 `goto`；若 `switch` 位于循环体内，属于该循环的 `continue` 也会转到循环的继续点，从而离开当前 `switch`。这些只是控制转移的例子，不是一张穷尽所有库调用或异常执行环境的清单。
<!-- bilingual-en:start -->
Common exits include `break`, `return`, and a `goto` to an outside label. If the `switch` is inside a loop, a `continue` belonging to that loop also transfers to the loop-continuation point and therefore leaves the current `switch`. These are examples of control transfer, not an exhaustive catalogue of library calls or exceptional execution environments.
<!-- bilingual-en:end -->

## 什么时候不用 switch

`case` 值必须是整型常量表达式，并且在同一个 `switch` 中转换后不能重复。连续范围、复合谓词或依赖多个变量的选择通常由 `if/else` 表达得更直接；不要为了“看起来整齐”把本来是谓词的问题硬塞进离散标签。
<!-- bilingual-en:start -->
Each `case` value must be an integer constant expression and, after conversion, must not duplicate another case in the same `switch`. Ranges, compound predicates, or choices depending on several variables are often clearer as `if/else`; do not force a predicate problem into discrete labels merely for visual symmetry.
<!-- bilingual-en:end -->

标签本身既不会阻断顺序执行，也不会创建一个新作用域。若从 `switch` 入口跳过了某个自动对象的初始化而后续代码仍读取它，问题不会因为代码视觉上位于同一对花括号内而消失。
<!-- bilingual-en:start -->
Labels neither block sequential execution nor create a new scope. If entry through a `switch` label skips the initialisation of an automatic object and later code reads it, enclosing both in the same braces does not make the read valid.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么两个相邻 `case` 共用一段代码时，第一个 `case` 后面不需要 `break`？
>
> **答案：** 两个标签只是同一语句序列的两个入口；有意贯穿让它们执行同一代码，真正离开点放在共享代码之后。

## 来源与核验

- [ISO C11 committee draft N1570, 6.8.1, 6.8.4.2, 6.8.6.2 and 6.8.6.3](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对标签不阻断控制流、`switch` 的整数提升与匹配、case/default 约束，以及 `continue`/`break` 的语义。
- [GCC, Warning Options: `-Wimplicit-fallthrough`](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html)：核对隐式贯穿诊断的工具边界。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.8.1, 6.8.4.2, 6.8.6.2, and 6.8.6.3](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for labels not impeding control flow, integer promotion and matching in `switch`, case/default constraints, and `continue`/`break` semantics.
- [GCC, Warning Options: `-Wimplicit-fallthrough`](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html) was checked for the tooling boundary around implicit fallthrough diagnostics.
<!-- bilingual-en:end -->
