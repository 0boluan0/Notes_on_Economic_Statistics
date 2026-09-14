---
aliases:
  - "声明说明名字和类型而定义还要提供对象或函数本身"
  - C declaration versus definition
  - C 声明与定义
student_os: knowledge-atom
atom_id: CS-C-003
atom_set: c-foundations
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[C 语言基础：编译、类型、控制流与函数.canvas]]"
---

# 声明说明名字和类型而定义还要提供对象或函数本身
<!-- bilingual-en:start -->
*A declaration introduces a name and type, while a definition also provides the object or function itself*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> 声明让当前翻译单元知道“这个名字代表什么类型的实体”；定义还给出函数体，或使一个对象获得定义所要求的存储。每个定义都是声明，但声明不一定是定义。
> <!-- bilingual-en:start -->
>
> &nbsp;
> A declaration tells the current translation unit what kind of entity a name denotes. A definition additionally provides a function body or causes an object to receive the storage required by its definition. Every definition is a declaration, but not every declaration is a definition.
> <!-- bilingual-en:end -->

```c
extern int count;          // 声明，不在这里定义 count
double mean(const int *, size_t);  // 函数声明

int count = 0;            // 对象定义
double mean(const int *a, size_t n) { /* ... */ } // 函数定义
```

声明可以在多个需要使用该接口的文件中重复出现，只要彼此兼容。若一个具有外部链接的函数或对象在表达式中被使用，整个程序必须恰有一个外部定义；若没有这样被使用，则至多一个即可。因而，不能把“见过声明”当成“定义已经参与链接”，也不能依赖多个翻译单元各给一份定义。
<!-- bilingual-en:start -->
Compatible declarations may appear in every file that needs the interface. If a function or object with external linkage is used in an expression, the entire program must contain exactly one external definition; if it is not so used, there may be at most one. Seeing a declaration therefore does not prove that a definition participates in the link, and separate translation units must not each supply their own definition.
<!-- bilingual-en:end -->

## 为什么头文件通常放声明

把共享声明放进头文件，让调用方和实现方看到同一份接口；把函数定义放进一个源文件，让链接时只有一个实现需要被选中。`#include` 的价值在于共享声明，不在于替代实现文件参与链接。
<!-- bilingual-en:start -->
Shared declarations normally belong in a header so callers and implementers see the same interface. A function definition normally belongs in one source file so the program has one implementation to link. The value of `#include` is sharing declarations, not replacing the implementation file in the link.
<!-- bilingual-en:end -->

## 暂定定义不是普通的 `extern` 声明

文件作用域的对象声明若没有初始化器，并且没有存储类说明符或只写了 `static`，就是**暂定定义**：例如 `int count;` 与 `static int cache;`。同一翻译单元中若只有一个或多个同名暂定定义而没有带初始化器的外部定义，翻译单元结束时的效果等同于用 `0` 初始化的一份定义。相反，单独的 `extern int count;` 不是暂定定义；`extern int count = 0;` 因为带初始化器，反而是定义。
<!-- bilingual-en:start -->
A file-scope object declaration with no initializer and either no storage-class specifier or only `static` is a **tentative definition**, as in `int count;` or `static int cache;`. If one translation unit contains one or more tentative definitions for the same identifier and no initialized external definition, the end-of-translation-unit effect is one definition initialized to zero. By contrast, `extern int count;` alone is not tentative, while `extern int count = 0;` is a definition because it has an initializer.
<!-- bilingual-en:end -->

这些合并规则只发生在**同一个翻译单元**内，不能把多个 `.c` 文件中的同名外部暂定定义当成可移植的“一份变量”。`inline` 还另有更细的定义与链接规则。基础判断仍然成立：看到 `extern`、函数原型或头文件内容时，不要据此断定程序已经拥有可链接的实现。
<!-- bilingual-en:start -->
This coalescing rule is confined to **one translation unit**; same-named external tentative definitions in several `.c` files are not a portable way to create one object. `inline` also has more detailed definition and linkage rules. The foundational judgement remains: seeing `extern`, a function prototype, or a header entry does not prove that the program already has a linkable implementation.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么把 `double mean(...);` 写进 `stats.h` 能消除调用处的隐式接口问题，却仍可能得到链接错误？
>
> **答案：** 头文件只提供声明；若没有任何参与链接的目标文件提供 `mean` 的定义，链接器仍找不到实现。

## 来源与核验

- [ISO C11 committee draft N1570, 6.7, 6.9, 6.9.1 and 6.9.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对声明、外部定义、函数定义与对象定义。
- [GCC, Header Files](https://gcc.gnu.org/onlinedocs/cpp/Header-Files.html)：核对头文件经预处理共享声明的机制。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.7, 6.9, 6.9.1, and 6.9.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for declarations, external definitions, function definitions, and object definitions.
- [GCC, Header Files](https://gcc.gnu.org/onlinedocs/cpp/Header-Files.html) was checked for how headers share declarations through preprocessing.
<!-- bilingual-en:end -->
