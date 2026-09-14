---
aliases:
  - "C 条件把标量零解释为假而赋值表达式仍会产生一个值"
  - C truth and assignment in conditions
  - C 条件真值与赋值
student_os: knowledge-atom
atom_id: CS-C-008
atom_set: c-foundations
atom_type: rule
status: source-checked
mastery_state: unassessed
part_of:
  - "[[C 语言基础：编译、类型、控制流与函数.canvas]]"
---

# C 条件把标量零解释为假而赋值表达式仍会产生一个值
<!-- bilingual-en:start -->
*C conditions interpret scalar zero as false, while an assignment expression still produces a value*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> `if`、`while` 等控制表达式接受标量：与零比较不等则为真，与零比较相等则为假。赋值 `x = value` 本身也是一个表达式，其值是赋值后的左操作数，因此它在条件中合法，却经常不是作者真正想写的比较。
> <!-- bilingual-en:start -->
>
> &nbsp;
> Control expressions such as those of `if` and `while` accept scalar values: a value unequal to zero is true and a value equal to zero is false. An assignment `x = value` is itself an expression whose value is the value stored in its left operand. It is therefore legal in a condition, even though it is often not the comparison the author intended.
> <!-- bilingual-en:end -->

```c
if (count = 0) {   // 把 count 改为 0；整个条件随后为假
    /* 不会执行 */
}

if (count == 0) {  // 比较，不修改 count
    /* count 为 0 时执行 */
}
```

指针也属于标量：空指针在条件中为假，非空指针为真。因此 `if (p)` 是 `if (p != NULL)` 的惯用缩写。比较和逻辑运算的结果是 `int` 值 `0` 或 `1`，但 C 并不要求传入条件的真值原本就只能是 `0` 或 `1`；任何非零标量都被视为真。
<!-- bilingual-en:start -->
Pointers are scalar too: a null pointer is false in a condition and a non-null pointer is true, making `if (p)` an idiom for `if (p != NULL)`. Comparison and logical operators produce `int` values `0` or `1`, but an incoming condition need not already be one of those two values; any nonzero scalar is treated as true.
<!-- bilingual-en:end -->

赋值表达式虽然有值，但在 C 中它本身不是左值；`(x = 3) = 4` 不是另一重合法赋值。这个边界也说明“表达式产生一个值”和“表达式还能指认一个可修改对象”是两件不同的事。
<!-- bilingual-en:start -->
Although an assignment expression has a value, it is not itself an lvalue in C, so `(x = 3) = 4` is not another valid assignment. Producing a value and designating a modifiable object are distinct properties.
<!-- bilingual-en:end -->

## 有意在条件中赋值时

`while ((ch = getchar()) != EOF)` 是有意赋值的典型写法：先保存读取结果，再明确比较终止标记。额外括号和显式比较把意图写给读者，也能让常见编译器警告区分“有意赋值”与笔误。
<!-- bilingual-en:start -->
`while ((ch = getchar()) != EOF)` is a standard intentional assignment: it stores the input result and then explicitly compares the termination marker. The extra parentheses and explicit comparison communicate intent and help common compiler warnings distinguish deliberate assignment from a typo.
<!-- bilingual-en:end -->

> [!question]- 自检
> `if (flags & READY)` 为什么可以工作，即使结果不一定等于 `1`？
>
> **答案：** 按位与结果只要非零，控制表达式就视为真；若只想测试是否含该位，这正是所需语义。

## 来源与核验

- [ISO C11 committee draft N1570, 6.5.3.3, 6.5.8–9, 6.5.13–14, 6.5.16, 6.8.4.1 and 6.8.5](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对比较与逻辑运算、赋值表达式的值和非左值边界，以及选择与循环条件的标量真值规则。
- [GCC, Warning Options: `-Wparentheses`](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html)：核对条件中可疑赋值的诊断方式。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.5.3.3, 6.5.8–9, 6.5.13–14, 6.5.16, 6.8.4.1, and 6.8.5](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for comparison and logical results, assignment-expression values and their non-lvalue boundary, and scalar truth in selection and loop conditions.
- [GCC, Warning Options: `-Wparentheses`](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html) was checked for diagnostics of suspicious assignment in a condition.
<!-- bilingual-en:end -->
