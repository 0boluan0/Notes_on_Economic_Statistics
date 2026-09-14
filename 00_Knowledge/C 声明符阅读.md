---
aliases:
  - "C 声明符从标识符向外组合且星号只修饰同一声明符"
  - C pointer declarators
  - C 指针声明解析
student_os: knowledge-atom
atom_id: CS-C-017
atom_set: c-memory
atom_type: procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[对象与指针对象]]"
related:
  - "[[取地址与解引用]]"
  - "[[字符串字面量修改边界]]"
part_of:
  - "[[C 指针、数组、字符串与动态内存.canvas]]"
leads_to:
  - "[[数组到指针转换]]"
  - "[[二级指针与指针回写]]"
---

# C 声明符从标识符向外组合且星号只修饰同一声明符
<!-- bilingual-en:start -->
*A C declarator composes outward from its identifier, and each asterisk belongs only to its own declarator*
<!-- bilingual-en:end -->

> [!summary] 解析方法
> C 声明由共同的**声明说明符**和一个或多个以逗号分开的**声明符**组成。声明符说明名字如何由基础类型构造出来；每个名字都要从标识符向外独立解析。于是 `int *p, q;` 声明的是“`p` 是指向 `int` 的指针，`q` 是 `int`”，不是两个指针。
> <!-- bilingual-en:start -->
> A C declaration combines declaration specifiers with one or more comma-separated declarators. Each declarator must be read independently from its identifier outward. Thus `int *p, q;` declares `p` as a pointer to `int` and `q` as an `int`.
> <!-- bilingual-en:end -->

```c
int *p, q;                 // p: pointer to int; q: int
const int *reader;         // modifiable pointer to const int
int *const fixed = &q;     // const pointer to modifiable int
int values[4];             // array of 4 int
int (*whole)[4] = &values; // pointer to array of 4 int
int *items[4];             // array of 4 pointers to int
```

`[]` 和 `()` 比前缀 `*` 结合得更紧，所以括号会改变类型结构。`int (*whole)[4]` 先看到名字 `whole`，向外遇到括号内的 `*`，再遇到 `[4]`：它是“指向含四个 `int` 的数组的指针”。去掉括号的 `int *items[4]` 则先遇到 `[4]`，是“含四个 `int *` 元素的数组”。
<!-- bilingual-en:start -->
Postfix `[]` and `()` bind more tightly than prefix `*`, so parentheses change the type structure. `int (*whole)[4]` is a pointer to an array of four `int`; `int *items[4]` is an array of four pointers to `int`.
<!-- bilingual-en:end -->

`const` 也要看它限定哪一层。`const int *reader` 禁止通过 `reader` 改写所指 `int`，但 `reader` 可以改指别处；`int *const fixed` 固定的是指针对象 `fixed` 自己，仍可通过它修改所指 `int`。把一个复杂声明改成 `typedef` 可以减轻阅读负担，却不能改变这些类型关系。
<!-- bilingual-en:start -->
Qualifiers bind to a particular layer. `const int *reader` may be redirected but cannot modify the pointed-to `int` through that access path; `int *const fixed` cannot be redirected but may modify its pointed-to `int`.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> `char *a, b[8];` 中，哪个名字是指针，哪个名字是数组？
>
> **答案：** `a` 是 `char *`；`b` 是含 8 个 `char` 的数组。逗号不会把 `a` 的 `*` 分享给 `b`。

## 来源与核验

- [ISO C11 committee draft N1570, 6.7 and 6.7.6–6.7.6.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对声明说明符、声明符的递归组合、指针限定符和数组声明符。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.7 and 6.7.6–6.7.6.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for declaration specifiers, recursive declarator composition, pointer qualifiers, and array declarators.
<!-- bilingual-en:end -->
