---
aliases:
  - "free 结束分配对象生命周期而所有权必须保证只释放一次"
  - C free and ownership
  - C 分配对象所有权
student_os: knowledge-atom
atom_id: CS-C-030
atom_set: c-memory
atom_type: invariant
status: source-checked
mastery_state: unassessed
requires:
  - "[[空、未确定与悬空指针]]"
  - "[[malloc 与 calloc]]"
related:
  - "[[C 按值传参与指针]]"
  - "[[二级指针与指针回写]]"
  - "[[指针别名与字符串深拷贝]]"
part_of:
  - "[[C 指针、数组、字符串与动态内存.canvas]]"
leads_to:
  - "[[内存契约错误分类]]"
  - "[[动态数组三元不变量]]"
---

# free 结束分配对象生命周期而所有权必须保证只释放一次
<!-- bilingual-en:start -->
*`free` ends an allocated object's lifetime, so ownership must ensure exactly one release*
<!-- bilingual-en:end -->

> [!summary] 生命周期与责任
> `free(ptr)` 接收空指针时不做任何事；否则 `ptr` 必须与某次尚未释放的内存管理函数返回值匹配。成功释放会立刻结束该分配对象的生命周期。C 的 `T *` 类型不记录谁有权释放，因此接口必须另外指定唯一的**所有者（owner）**，并保证对象在最后一次使用后恰好释放一次。
> <!-- bilingual-en:start -->
> `free(ptr)` does nothing for a null pointer; otherwise `ptr` must match a still-live value returned by a memory-management function. Deallocation immediately ends the allocated object's lifetime. A C `T *` type does not encode who may release it, so the interface must name an owner and guarantee exactly one release after the last use.
> <!-- bilingual-en:end -->

```c
#include <stdio.h>
#include <stdlib.h>

void ownership_example(void) {
    int *owner = malloc(sizeof *owner);
    if (owner != NULL) {
        *owner = 42;
        int *borrowed = owner;  // 只借用，不取得释放权

        printf("%d\n", *borrowed);
        free(owner);
        owner = NULL;
        // borrowed 现在也不能读取、比较或释放
    }
}
```

`borrowed` 是**借用别名（borrowed alias）**：它能在所有者允许且对象仍存活时访问，却不负责 `free`。释放不是修改一个指针变量，而是结束对象本身；所以所有指向它及其内部的别名一起失效。把 `owner` 置空只能降低这个变量被再次误用的机会，不能更新 `borrowed`。
<!-- bilingual-en:start -->
`borrowed` is a borrowed alias: it may access the object only while the owner permits and the object remains alive, and it must not call `free`. Deallocation ends the object rather than editing one pointer variable, so all aliases to it or its interior become invalid. Nulling one owner variable does not update them.
<!-- bilingual-en:end -->

合法释放值只能是空指针，或先前由 `malloc`、`calloc`、`realloc` 等返回且尚未释放的匹配值。不能 `free` 栈上数组、字符串字面量、全局对象地址，也不能释放 `p + 1` 这样的内部指针；即使它们指向真实存储，也不是分配函数返回的对象起始值。
<!-- bilingual-en:start -->
A valid release argument is null or a matching, not-yet-released value returned by an allocation function. Stack arrays, string literals, global objects, and interior pointers such as `p + 1` must not be freed, even though they may designate real storage.
<!-- bilingual-en:end -->

所有权转移时，旧所有者必须停止释放和使用，新所有者承担最终释放；只复制指针并不会自动转移责任。若 API 返回新分配、接收并消费旧分配或只暂借地址，应在接口文字中分别写明。没有这份契约，单看 `char *` 或 `void *` 无法决定调用方该不该释放。
<!-- bilingual-en:start -->
On ownership transfer, the old owner stops using and releasing the object and the new owner assumes final release. Copying a pointer does not transfer responsibility automatically. An API must distinguish returning a new allocation, consuming an existing one, and merely borrowing an address.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 为什么 `free(p); p = NULL;` 不能证明对象没有其他释放后使用？
>
> **答案：** `free` 让对象寿命结束，但其他别名不会随 `p` 置空；它们仍可能被误用。必须控制全部借用不超过所有者的生命周期。

## 来源与核验

- [ISO C11 committee draft N1570, 6.2.4 paragraph 2, 7.22.3, and 7.22.3.3](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对分配对象生命周期、合法 `free` 参数、空指针与重复释放的规范边界。
- [CS50x 2026, Lecture 4 notes](https://cs50.harvard.edu/x/notes/4/)：核对课程中每次成功分配最终 `free` 的基本责任；owner/borrow 是在 C 不编码该责任时使用的工程约定，不是新增语言语义。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 6.2.4 paragraph 2, 7.22.3, and 7.22.3.3](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for allocated-object lifetime, valid `free` arguments, null, and repeated release.
- [CS50x 2026 Lecture 4 notes](https://cs50.harvard.edu/x/notes/4/) were checked for the basic responsibility to `free` each successful allocation; owner and borrow are engineering conventions used because C does not encode that responsibility.
<!-- bilingual-en:end -->
