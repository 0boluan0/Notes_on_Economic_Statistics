---
aliases:
  - "C 的所有函数实参都按值传递而指针值可用于修改所指对象"
  - C pass by value
  - C 指针参数不是引用传递
student_os: knowledge-atom
atom_id: CS-C-013
atom_set: c-foundations
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[函数原型与语义契约]]"
part_of:
  - "[[C 语言基础：编译、类型、控制流与函数.canvas]]"
  - "[[C 指针、数组、字符串与动态内存.canvas]]"
leads_to:
  - "[[二级指针与指针回写]]"
---

# C 的所有函数实参都按值传递而指针值可用于修改所指对象
<!-- bilingual-en:start -->
*All C function arguments are passed by value, while a copied pointer value can be used to modify its pointee*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> 调用 C 函数时，实参的值经过规定转换后赋给形参；函数得到的是自己的形参对象。传入指针也没有改成“引用传递”：复制的是指针值。函数可通过这份指针副本修改它所指的调用者对象，但给形参指针重新赋值不会改动调用者的指针变量。
> <!-- bilingual-en:start -->
>
> &nbsp;
> In a C function call, converted argument values initialise the parameter objects, which belong to the called function. Passing a pointer does not switch the language to pass-by-reference: the pointer value is copied. The function can modify a caller-owned object through that pointer copy, but assigning a new value to the pointer parameter does not alter the caller's pointer variable.
> <!-- bilingual-en:end -->

```c
#include <stddef.h>

void set_zero(int *p) {
    *p = 0;       // 修改 p 所指的 int
}

void forget(int *p) {
    p = NULL;     // 只修改本地形参 p
    (void) p;
}

void example(void) {
    int x = 7;
    int *q = &x;
    set_zero(q);  // x 变为 0
    forget(q);    // q 仍指向 x
}
```

这一区别能直接回答“为什么函数没改到调用者变量”：若传入普通 `int`，函数只改到副本；若要改该 `int`，传入它的地址并解引用。若要改调用者自己的指针变量，则进入[[二级指针与指针回写|返回指针或二级指针输出参数]]；若结果还涉及动态分配，接口还要另写所有权语义。
<!-- bilingual-en:start -->
This distinction directly answers why a function failed to change a caller variable. Passing an ordinary `int` lets the function alter only its copy; to alter the caller's `int`, pass its address and dereference it. Changing the caller's pointer object is the separate return-value or pointer-to-pointer application, with an additional ownership contract when allocation is involved.
<!-- bilingual-en:end -->

## 边界

能通过指针写入不等于允许写入。函数在解引用或写入之前，必须确保指针非空、指向仍在生命周期内且具有适当类型与可修改性的对象；单纯传入、复制或比较一个空指针本身并不违规。`const int *` 禁止通过该访问路径修改 `int`，却不自动断言底层对象在所有访问路径上都不可变。这些条件超出“值传递”本身，需要由对象与接口规则继续约束。
<!-- bilingual-en:start -->
The ability to write through a pointer is not permission to do so. Before dereferencing or writing, the function must ensure that the pointer is non-null and designates a live, suitably typed, modifiable object; merely passing, copying, or comparing a null pointer is not itself invalid. `const int *` forbids modifying the `int` through that access path but does not by itself make the underlying object immutable through every path. These conditions lie beyond pass-by-value itself and must be maintained by object and interface rules.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 `swap(int a, int b)` 交换不了调用者的两个整数，而 `swap(int *a, int *b)` 可以？
>
> **答案：** 前者只交换两个形参副本；后者复制两个地址，再通过 `*a`、`*b` 修改地址所指的调用者对象。

## 来源与核验

- [ISO C11 committee draft N1570, 6.5.2.2 paragraph 4](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对实参值赋给形参的调用语义。
- [CS50x 2026, Lecture 4 notes](https://cs50.harvard.edu/x/notes/4/)：核对课程语境中地址、指针与通过地址修改对象的解释。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.5.2.2 paragraph 4](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for the semantics of assigning argument values to parameters.
- [CS50x 2026, Lecture 4 notes](https://cs50.harvard.edu/x/notes/4/) were checked for the course treatment of addresses, pointers, and modifying objects through addresses.
<!-- bilingual-en:end -->
