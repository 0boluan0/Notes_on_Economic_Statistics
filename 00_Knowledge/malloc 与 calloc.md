---
aliases:
  - "malloc 与 calloc 建立分配对象但初始化保证不同且都可能失败"
  - C malloc and calloc
  - C 分配对象初始化
student_os: knowledge-atom
atom_id: CS-C-027
atom_set: c-memory
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[对象与指针对象]]"
  - "[[空、未确定与悬空指针]]"
related:
  - "[[动态分配大小检查]]"
  - "[[free 与单一所有权]]"
part_of:
  - "[[C 指针、数组、字符串与动态内存.canvas]]"
leads_to:
  - "[[realloc 失败安全]]"
---

# malloc 与 calloc 建立分配对象但初始化保证不同且都可能失败
<!-- bilingual-en:start -->
*`malloc` and `calloc` create allocated objects with different initialisation guarantees, and either may fail*
<!-- bilingual-en:end -->

> [!summary] 分配与初始化分开
> `malloc(size)` 请求一个 `size` 字节的分配对象，其初始值未确定；`calloc(count, size)` 请求由 `count` 个、每个 `size` 字节的成员组成的空间，并把所有位初始化为零。两者成功时返回指向新对象起始位置的 `void *`，其对齐足以用于具有 **fundamental alignment requirement（基本对齐要求）** 的对象类型；失败时返回空指针。任何读取或写入都必须先确认返回非空，并在 `malloc` 后先初始化相应部分再读取。
> <!-- bilingual-en:start -->
> `malloc(size)` requests an allocated object of `size` bytes with indeterminate initial value. `calloc(count, size)` requests space for `count` members of `size` bytes and initialises all bits to zero. On success either returns a `void *` to a new object, suitably aligned for object types with a fundamental alignment requirement; on failure it returns a null pointer. Check success and initialise `malloc` storage before reading it.
> <!-- bilingual-en:end -->

```c
#include <stdlib.h>

void allocation_examples(size_t count) {
    int *one = malloc(sizeof *one);
    if (one == NULL) {
        /* 处理失败 */
    } else {
        *one = 42;            // 先写后读
    }
    free(one);                // 空指针也可交给 free

    int *zeros = calloc(count, sizeof *zeros);
    if (zeros == NULL && count != 0) {
        /* 处理失败 */
    }
    free(zeros);
}
```

在 C 中，`void *` 可隐式转换为对象指针，所以不需要给 `malloc` 返回值加类型转换。用 `sizeof *one` 或 `sizeof *zeros` 从目标指针推导元素大小，类型变化时不容易让分配公式悄悄过期。若要把 `count * sizeof *p` 先算成一个字节数再交给 `malloc` 或 `realloc`，必须在乘法前检查可表示性；`calloc` 把两个操作数分开接收，不需要调用方先做这次乘法，但返回值仍必须检查。
<!-- bilingual-en:start -->
In C, `void *` converts implicitly to an object-pointer type, so no cast is required. `sizeof *one` or `sizeof *zeros` derives the element size from the target pointer, reducing type-drift errors. A caller-side multiplication for `malloc` or `realloc` must be checked before it is evaluated. `calloc` receives the two factors separately, avoiding that caller-side multiplication, but its result still requires a null check.
<!-- bilingual-en:end -->

“所有位为零”是对象表示保证，不应一概翻译成“每种类型的值都初始化为语言意义的零”。N1570 特别指出，它不必等于浮点零或空指针的表示。若要建立某种类型的业务初值，仍应按该类型显式赋值，而不是把 `calloc` 当通用构造器。
<!-- bilingual-en:start -->
All-bits-zero is an object-representation guarantee, not a universal promise of every type's semantic zero. N1570 explicitly notes that it need not represent floating zero or a null pointer. Establish type-specific initial values explicitly rather than treating `calloc` as a general constructor.
<!-- bilingual-en:end -->

“适当对齐”也有范围。N1570 的保证覆盖不超过 `_Alignof(max_align_t)` 的基本对齐；实现支持的更严格扩展对齐不由普通 `malloc`/`calloc` 这句话自动保证。若对象类型需要扩展对齐，应使用该语言版本与实现明确提供的对齐分配接口，并核对其释放规则。
<!-- bilingual-en:start -->
“Suitably aligned” also has a boundary. N1570 guarantees fundamental alignment, no stricter than `_Alignof(max_align_t)`; an implementation's extended alignments are not automatically covered by the ordinary `malloc`/`calloc` guarantee. An over-aligned type requires an allocation facility explicitly suitable for that language version and implementation, together with its matching release contract.
<!-- bilingual-en:end -->

ISO C 称这类区域为 **allocated storage（分配存储）**；“heap（堆）”是常见实现和课程模型，不是标准要求的具体数据结构。分配对象的生命周期从成功分配延续到释放。零字节请求的结果由实现决定：可能返回空指针，也可能返回一个不得用来访问对象的非空值；需要真实元素时不要把零大小成功当作可访问容量。
<!-- bilingual-en:start -->
ISO C calls this allocated storage; “heap” is a common implementation and teaching model, not a required data structure. The allocated object's lifetime lasts from successful allocation to deallocation. A zero-size request is implementation-defined and never provides an element that may be accessed.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> `int *p = malloc(sizeof *p);` 成功返回后，为什么仍不能在赋值前立刻读取 `*p`？
>
> **答案：** 成功只建立了足够且适当对齐的分配对象；`malloc` 没有初始化其值，读取前必须先写入有效的 `int`。

## 来源与核验

- [ISO C11 committee draft N1570, 6.2.8 paragraphs 2–3, 7.19, 7.22.3, 7.22.3.2, and 7.22.3.4](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对基本与扩展对齐、`max_align_t`、对象互不相交、生命周期、零大小请求、失败、`calloc` 全零位与 `malloc` 未确定初值。
- [CS50x 2026, Lecture 4 notes](https://cs50.harvard.edu/x/notes/4/)：核对课程对 `malloc`、空返回、`free` 与未初始化“garbage values”的引入。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.2.8 paragraphs 2–3, 7.19, 7.22.3, 7.22.3.2, and 7.22.3.4](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for fundamental and extended alignment, `max_align_t`, disjoint objects, lifetime, zero-size requests, failure, all-bits-zero `calloc`, and indeterminate `malloc` values.
- [CS50x 2026 Lecture 4 notes](https://cs50.harvard.edu/x/notes/4/) were checked for the course introductions to `malloc`, null results, `free`, and uninitialised “garbage values.”
<!-- bilingual-en:end -->
