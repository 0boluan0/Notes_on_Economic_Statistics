---
aliases:
  - "realloc 成功会替换旧对象而失败必须保留原分配"
  - C realloc transaction
  - C realloc 失效规则
student_os: knowledge-atom
atom_id: CS-C-029
atom_set: c-memory
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[malloc 与 calloc]]"
  - "[[动态分配大小检查]]"
related:
  - "[[二级指针与指针回写]]"
  - "[[free 与单一所有权]]"
part_of:
  - "[[C 指针、数组、字符串与动态内存.canvas]]"
leads_to:
  - "[[动态数组三元不变量]]"
---

# realloc 成功会替换旧对象而失败必须保留原分配
<!-- bilingual-en:start -->
*A successful `realloc` replaces the old object, whereas failure must preserve the original allocation*
<!-- bilingual-en:end -->

> [!summary] 两条互斥路径
> 对正的目标大小，若 `old` 非空，`realloc(old, new_size)` 成功时会结束旧分配对象的生命周期，并返回一个指定大小的新对象；返回地址的数值可能与旧值相同，也可能不同。若 `old` 为空，这次调用等同相同大小的 `malloc`，并没有旧对象可替换。失败时返回空指针，非空旧对象不被释放且内容不变。因此必须先把结果接到临时指针，确认成功后再替换唯一所有者。
> <!-- bilingual-en:start -->
> For a positive new size and non-null `old`, successful `realloc(old, new_size)` ends the old allocated object's lifetime and returns a new object of the requested size; the numerical pointer value may or may not change. A null `old` instead makes the call equivalent to `malloc(new_size)`, with no prior object to replace. On failure it returns null and leaves any non-null old object unchanged. Receive the result in a temporary pointer and replace the owner only after success.
> <!-- bilingual-en:end -->

下面的调整函数还有一个调用方前置：`*items` 必须是空指针，或者与一个仍存活、可交给 `realloc` 的分配值匹配；`*capacity` 也必须如实描述当前元素容量。C 不能仅从裸指针反查这两件事。
<!-- bilingual-en:start -->
The caller must additionally provide `*items` as null or as a still-live allocation value acceptable to `realloc`, with `*capacity` accurately describing its current element capacity. C cannot recover either fact from the bare pointer alone.
<!-- bilingual-en:end -->

```c
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

bool resize_ints(int **items, size_t *capacity, size_t new_capacity) {
    if (items == NULL || capacity == NULL || new_capacity == 0 ||
        new_capacity > SIZE_MAX / sizeof **items) {
        return false;
    }

    size_t bytes = new_capacity * sizeof **items;
    void *tmp = realloc(*items, bytes);
    if (tmp == NULL) {
        return false;              // *items 与 *capacity 都保持原状
    }

    *items = tmp;
    *capacity = new_capacity;
    return true;
}
```

不能直接写 `*items = realloc(*items, bytes)`：一旦失败，空指针会覆盖通往旧对象的唯一所有者，旧对象仍活着却可能再也无法释放，形成泄漏。临时指针写法是由失败语义推出的工程协议，不是因为标准要求 `realloc` 一定搬家。
<!-- bilingual-en:start -->
Direct assignment can overwrite the only owner with null on failure, leaking an unchanged live allocation. The temporary-pointer pattern follows from failure semantics; it is not based on an assumption that `realloc` must move the object.
<!-- bilingual-en:end -->

当旧指针非空且调用成功时，新对象开头 `min(old_size, new_size)` 个字节保留旧内容；扩大的尾部具有未确定值，必须初始化后再读。更严格地说，旧对象已经被释放，所以所有指向旧对象或其内部的旧别名都失效，即使返回指针的数值与原指针相同也不能继续使用旧别名；必须从新返回值重新取得需要的元素地址。
<!-- bilingual-en:start -->
When a non-null old pointer is successfully reallocated, the first `min(old_size, new_size)` bytes are preserved and any grown tail is indeterminate. The old object has nevertheless been deallocated, so every old alias to it or its interior is invalid, even when the returned pointer has the same numerical value. Derive later element pointers again from the new result.
<!-- bilingual-en:end -->

本原子故意要求 `new_capacity > 0`，因为上面的“返回空就保留旧对象”不能直接套给 `realloc(p, 0)`。在 N1570/C11 中，零大小请求的结果由实现决定；WG14 DR 400 进一步记录，不同实现对 `realloc(p, 0)` 返回空时是否已释放旧对象并不一致，调用方不能只看空返回就可移植地恢复所有权。C17 后来把相关选择明确留给实现并将该用法列为过时方向；后续版本又改变了边界。若接口要清空，本原子采用的工程协议是显式 `free`，再把所有者、长度和容量一起重置，而不是把零大小混进正大小事务。
<!-- bilingual-en:start -->
This atom deliberately requires a positive new capacity because the positive-size rule “null preserves the old object” cannot simply be applied to `realloc(p, 0)`. Under N1570/C11, a zero-size request is implementation-defined; WG14 DR 400 records incompatible implementation choices over whether a null result has already released the old object. C17 later made the choice explicit and marked the usage obsolescent, while later revisions changed the boundary again. This atom therefore adopts an explicit `free` followed by resetting owner, length, and capacity as an engineering protocol rather than mixing zero size into the positive-size transaction.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 若 `realloc` 返回的地址与旧地址数值相同，为什么旧的元素指针仍不能继续使用？
>
> **答案：** 成功语义先结束旧对象、再给出新对象；对象身份和生命周期已经改变，数值地址相同不能让旧别名复活。

## 来源与核验

- [ISO C11 committee draft N1570, 6.2.4 paragraph 2 and 7.22.3.5](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对生命周期结束、内容保留范围、扩展尾部、成功替换、失败保留与可能相同的返回值。
- [WG14 Defect Report 400, `realloc` with size zero](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2243.htm#dr_400)：核对 C11 实现分歧、C17 勘误后的实现定义分支与过时标记；正大小主规则仍以 N1570 为准。
- [SEI CERT C, ERR33-C](https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152272/ERR33-C.%2BDetect%2Band%2Bhandle%2Bstandard%2Blibrary%2Berrors)：作为工程建议，核对用临时指针保留原分配的错误处理模式；`realloc` 语义仍来自 N1570。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.2.4 paragraph 2 and 7.22.3.5](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for lifetime expiry, the preserved prefix, indeterminate growth, successful replacement, failure preservation, and a possibly identical returned value.
- [WG14 Defect Report 400 on zero-size `realloc`](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2243.htm#dr_400) was checked for divergent C11 implementation choices and the C17 corrigendum's implementation-defined branch and obsolescent marker; the positive-size rule remains grounded in N1570.
- [SEI CERT C ERR33-C](https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152272/ERR33-C.%2BDetect%2Band%2Bhandle%2Bstandard%2Blibrary%2Berrors) was used only as engineering guidance for the temporary-pointer error pattern; `realloc` semantics come from N1570.
<!-- bilingual-en:end -->
