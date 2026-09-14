---
aliases:
  - "while 先测试而 do while 至少执行一次且 for 并列循环三部件"
  - C loop forms
  - for while do while 区别
student_os: knowledge-atom
atom_id: CS-C-011
atom_set: c-foundations
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[C 条件与赋值表达式]]"
part_of:
  - "[[C 语言基础：编译、类型、控制流与函数.canvas]]"
leads_to:
  - "[[循环不变量与终止]]"
---

# while 先测试而 do while 至少执行一次且 for 并列循环三部件
<!-- bilingual-en:start -->
*`while` tests first, `do while` executes at least once, and `for` places three loop components together*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> `while` 在每轮之前检查条件，初始条件为假时一次也不执行；`do ... while` 在循环体之后检查，因此循环体至少执行一次；`for` 把初始化、继续条件和每轮后的表达式放在同一结构中。三者都能表达重复，但测试位置决定了边界语义。
> <!-- bilingual-en:start -->
>
> &nbsp;
> `while` tests before each iteration and may execute zero times. `do ... while` tests after the body and therefore executes the body at least once. `for` places initialisation, the continuation test, and the post-iteration expression in one structure. All three express repetition, but the test location determines boundary behaviour.
> <!-- bilingual-en:end -->

```c
for (size_t i = 0; i < n; i++) {
    use(values[i]);
}
```

这段 `for` 把索引的起点、合法区间和推进方式放在一起，适合边界清楚的遍历。若推进取决于循环体中的多条路径，`while` 往往更诚实；若必须先显示一次菜单或先读取一次状态再决定是否重复，`do ... while` 才自然。
<!-- bilingual-en:start -->
This `for` loop keeps the index start, valid range, and update together, which suits a traversal with a clear boundary. If progress depends on several paths inside the body, `while` is often more honest. `do ... while` is natural when a menu must be shown once, or a state must be read once, before deciding whether to repeat.
<!-- bilingual-en:end -->

## `for` 的实际控制顺序

`for (init; test; step) body` 先执行一次 `init`，再检查 `test`；条件为真才执行 `body`，之后执行 `step`，再回到条件。`continue` 在 `for` 中转去执行 `step`，在 `while` 中转去下一次前置条件测试，在 `do ... while` 中则转去末尾的后置条件测试。把一种循环机械改写成另一种形式时，这个差异可能改变状态是否推进以及条件何时求值。
<!-- bilingual-en:start -->
`for (init; test; step) body` executes `init` once, then tests `test`; only a true test enters `body`, after which `step` runs before the next test. A `continue` in a `for` loop proceeds to `step`, in a `while` loop to the next pre-test, and in a `do ... while` loop to the post-test at the end. A mechanical rewrite can therefore change both whether state advances and when the condition is evaluated.
<!-- bilingual-en:end -->

## 边界

选择哪种语法不能替代正确性论证。若循环按设计应结束，就要明确合法状态、不变量和终止推进。`for (;;)` 的缺省条件按非零常量处理，`while (1)` 也使用恒真条件；它们本身不会因条件变假而退出，只能在设计需要退出时依靠循环体中的 `break`、`return` 等控制转移。有意永久运行的事件循环则不承担“最终退出”的目标。
<!-- bilingual-en:start -->
Syntax choice does not replace a correctness argument. If a loop is designed to finish, state its valid states, invariant, and termination progress. The omitted condition in `for (;;)` is treated as a nonzero constant, while `while (1)` also has an always-true condition; neither exits because its condition becomes false, so a design that must exit relies on control transfer such as `break` or `return` inside the body. A deliberately permanent event loop has no “eventual exit” goal to prove.
<!-- bilingual-en:end -->

> [!question]- 自检
> 把含有 `continue` 的 `for` 循环改写成 `while` 时，最容易漏掉什么？
>
> **答案：** `for` 的 `continue` 仍会执行迭代表达式；改成 `while` 后若没有在 `continue` 前推进状态，可能卡在同一状态。

## 来源与核验

- [ISO C11 committee draft N1570, 6.8.5–6.8.5.3 and 6.8.6.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对三种迭代语句、缺省条件与 `continue` 的控制转移。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.8.5–6.8.5.3 and 6.8.6.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for the three iteration forms, omitted conditions, and `continue` control transfer.
<!-- bilingual-en:end -->
