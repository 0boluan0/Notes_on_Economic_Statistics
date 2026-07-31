---
aliases:
  - "C Pointers, Arrays, Strings, and Dynamic Memory"
  - "C Memory Management"
  - "C指针与内存"
status: source-checked
---

# C 指针、数组、字符串与动态内存
<!-- bilingual-en:start -->
*C Pointers, Arrays, Strings, and Dynamic Memory*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在 C 中明确一段内存的地址、边界、生命周期与释放责任。
> **具体锚点：** 一个指针只保存地址；把它传给函数不会同时传递数组长度，也不会证明所指内存仍然有效。
> **核心难点：** 数组在表达式中常衰变为指针，但数组对象、指针值和已分配内存的所有权不是同一件事。
> **为什么重要：** 越界、悬空指针、重复释放和泄漏都来自边界或生命周期契约不清。
> **继续：** 对每个指针写出来源、可访问元素数、所有者和失效时刻，再读代码。
> <!-- bilingual-en:start -->
> **Problem addressed:** Make the address, bounds, lifetime, and release responsibility of a region of C memory explicit.
> **Concrete anchor:** A pointer stores only an address; passing it to a function does not carry an array length or prove that the referenced storage is still alive.
> **Central difficulty:** An array often decays to a pointer in an expression, but the array object, pointer value, and ownership of allocated storage are different concepts.
> **Why it matters:** Out-of-bounds access, dangling pointers, double frees, and leaks all arise from an unclear bounds or lifetime contract.
> **Continue with:** For every pointer, state its origin, accessible element count, owner, and invalidation point before following the code.
> <!-- bilingual-en:end -->

## 数组、字符串与指针
<!-- bilingual-en:start -->
*Arrays, Strings, and Pointers*
<!-- bilingual-en:end -->

数组在多数表达式中衰变为首元素指针，但数组大小信息不会随指针传递。`a[i]` 等价于 `*(a+i)`；指针算术按元素大小移动。字符串必须有 `\0` 且缓冲区足够，输入函数需限制长度。
<!-- bilingual-en:start -->
In most expressions an array decays to a pointer to its first element, but its size does not travel with that pointer. `a[i]` is equivalent to `*(a+i)`, and pointer arithmetic advances in units of the pointed-to type. A string needs a terminating `\0` and a sufficiently large buffer, so input length must be bounded.
<!-- bilingual-en:end -->

数组和指针为何“相关但不同”：数组对象拥有固定数量的连续元素，`sizeof array` 在其声明作用域可得到总字节数；指针只是一个地址值，`sizeof pointer` 只得到地址本身大小。
<!-- bilingual-en:start -->
Arrays and pointers are related but distinct. An array object owns a fixed number of contiguous elements, and `sizeof array` in its declaration scope yields total bytes. A pointer is an address value, so `sizeof pointer` yields only the size of that address.
<!-- bilingual-en:end -->

## 动态内存
<!-- bilingual-en:start -->
*Dynamic Memory*
<!-- bilingual-en:end -->

`malloc/calloc` 在 heap 分配，检查空指针并用 `free` 精确释放一次。内存泄漏、use-after-free、double free 和越界是不同错误；所有权约定应明确谁释放。
<!-- bilingual-en:start -->
`malloc` and `calloc` allocate on the heap. Check for a null result and release each successful allocation exactly once with `free`. A leak, use-after-free, double free, and out-of-bounds access are distinct failures; the ownership contract must identify who releases the allocation.
<!-- bilingual-en:end -->

动态数组通常需要同时保存指针、逻辑长度和容量。扩容时先用临时指针接收 `realloc`，因为失败会返回 `NULL` 而原分配仍有效；直接覆盖唯一指针会在失败时泄漏。
<!-- bilingual-en:start -->
A dynamic array normally stores a pointer, logical length, and capacity together. Assign `realloc` to a temporary pointer first: failure returns `NULL` while the original allocation remains valid, so overwriting the only pointer would leak it.
<!-- bilingual-en:end -->

## Worked example：复制字符串并明确所有权
<!-- bilingual-en:start -->
*Worked Example: Copy a String with Explicit Ownership*
<!-- bilingual-en:end -->

返回的新字符串由调用者拥有。分配大小必须包含终止符，复制只能在分配成功后发生。
<!-- bilingual-en:start -->
The caller owns the returned string. The allocation size must include the terminator, and copying can occur only after allocation succeeds.
<!-- bilingual-en:end -->

```c
char *copy_string(const char *source) {
    size_t n = strlen(source) + 1;
    char *copy = malloc(n);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, source, n);
    return copy;
}
```

调用者取得非空返回值后必须最终 `free` 一次，且释放后不再解引用。若 API 没写这条所有权规则，单看类型 `char *` 无法推断责任。
<!-- bilingual-en:start -->
After receiving a non-null result, the caller must eventually call `free` exactly once and must not dereference the pointer afterward. Without this ownership rule in the API, the type `char *` alone cannot reveal responsibility.
<!-- bilingual-en:end -->

## 正确性与工具
<!-- bilingual-en:start -->
*Correctness and Tools*
<!-- bilingual-en:end -->

编译器警告开全、使用 debugger 和 sanitizer，边界用小输入/空输入/最大值测试。手工内存管理不是练勇气，而是练可证明的生命周期。
<!-- bilingual-en:start -->
Enable compiler warnings, use a debugger and sanitizers, and test boundaries with tiny, empty, and maximum-sized inputs. Manual memory management is not an exercise in bravery; it is an exercise in provable lifetimes.
<!-- bilingual-en:end -->

AddressSanitizer 擅长发现越界和 use-after-free，LeakSanitizer/Valgrind 可帮助发现泄漏，但工具报告仍需回到“谁拥有、边界多大、何时失效”解释根因。
<!-- bilingual-en:start -->
AddressSanitizer is effective at detecting out-of-bounds access and use-after-free, while LeakSanitizer or Valgrind can expose leaks. A tool report still has to be explained through ownership, bounds, and invalidation to locate the root cause.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- segmentation fault：先找无效地址来源，再核对索引边界和对象是否已离开生命周期。
  <!-- bilingual-en:start -->
  Segmentation fault: identify the origin of the invalid address, then check index bounds and whether the object has outlived its storage.
  <!-- bilingual-en:end -->
- 字符串偶尔多出乱码：检查是否缺 `\0`、缓冲区是否够大，以及读取函数是否知道最大长度。
  <!-- bilingual-en:start -->
  A string occasionally has garbage suffixes: inspect the `\0` terminator, buffer capacity, and whether the input routine knows the maximum length.
  <!-- bilingual-en:end -->
- `free` 后仍工作一阵：这是未定义行为，不是安全证据；立即清除后续访问并重新建立所有权路径。
  <!-- bilingual-en:start -->
  Code appears to work after `free`: this is undefined behavior, not evidence of safety; remove later accesses and reconstruct the ownership path.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 数组和指针为什么相关但不相同？
<!-- bilingual-en:start -->
*Why are arrays and pointers related but not identical?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 数组是固定数量元素的对象；多数表达式中会转换为首元素指针，转换后长度和数组对象语义丢失。
> <!-- bilingual-en:start -->
> An array is an object containing a fixed number of elements. In most expressions it converts to a pointer to the first element, after which length and array-object semantics are lost.
> <!-- bilingual-en:end -->

### `malloc` 成功后最少需要管理什么？
<!-- bilingual-en:start -->
*What is the minimum information that must be managed after `malloc` succeeds?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 分配大小和类型、空返回、初始化、唯一所有权、所有访问边界以及恰好一次释放。
> <!-- bilingual-en:start -->
> Allocation size and type, null failure, initialization, unique ownership, every access bound, and exactly one release.
> <!-- bilingual-en:end -->

### 为什么只把释放后的指针设成 `NULL` 不能消除 use-after-free？
<!-- bilingual-en:start -->
*Why does setting one freed pointer to `NULL` not eliminate use-after-free?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 其他别名仍可能保存旧地址；根本办法是控制所有权与借用生命周期，而不只是修改一个名称。
> <!-- bilingual-en:start -->
> Other aliases may still retain the old address; the root solution is controlling ownership and borrowing lifetimes, not changing one name.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [CS50x 2026 official course](https://cs50.harvard.edu/x/)：核验数组、字符串、指针、动态分配和典型内存错误。
  <!-- bilingual-en:start -->
  The [official CS50x 2026 course](https://cs50.harvard.edu/x/) verifies arrays, strings, pointers, dynamic allocation, and common memory errors.
  <!-- bilingual-en:end -->
