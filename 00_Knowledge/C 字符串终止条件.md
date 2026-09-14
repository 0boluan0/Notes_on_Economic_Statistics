---
aliases:
  - "字符数组只有在容量内存在首个空字符时才表示 C 字符串"
  - C null-terminated byte string
  - C 字符串容量
student_os: knowledge-atom
atom_id: CS-C-023
atom_set: c-memory
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[数组到指针转换]]"
  - "[[数组下标与边界]]"
related:
  - "[[形参数组退化与边界]]"
part_of:
  - "[[C 指针、数组、字符串与动态内存.canvas]]"
leads_to:
  - "[[字符串字面量修改边界]]"
  - "[[指针别名与字符串深拷贝]]"
---

# 字符数组只有在容量内存在首个空字符时才表示 C 字符串
<!-- bilingual-en:start -->
*A character array represents a C string only when its first null character lies within the available capacity*
<!-- bilingual-en:end -->

> [!summary] 表示不变量
> C 的窄字符串是从首字符开始、到**第一个空字符**为止并包含该空字符的连续字符序列。空字符写作 `\0`，数值为零；它不是显示出来的字符 `'0'`。一个 `char` 数组只有在可访问边界内存在这个终止符时，才能交给要求字符串的函数。
> <!-- bilingual-en:start -->
> A C narrow string is a contiguous character sequence terminated by and including its first null character. The null character is written `\0` and has value zero; it is not the visible character `'0'`. A `char` array may be passed to a string function only when that terminator exists within the accessible object.
> <!-- bilingual-en:end -->

```c
#include <stddef.h>
#include <string.h>

void string_extent_example(void) {
    char word[4] = {'c', 'a', 't', '\0'};
    char raw[3]  = {'c', 'a', 't'};
    size_t capacity = sizeof word;   // 4：数组容量（字节）
    size_t length = strlen(word);    // 3：首个 \0 之前的字节数

    (void) raw;       // raw 是字符数组，但不是 C 字符串
    (void) capacity;
    (void) length;
}
```

`word` 是字符串，`raw` 只是三个字符的数组。对 `raw` 调用 `strlen`、用 `%s` 输出或交给 `strcpy` 会让函数继续越过数组寻找 `\0`，从而产生越界访问和未定义行为。`strlen` 也不能告诉你目标缓冲区容量：它只在“输入已经是有效字符串”的前提下数到首个终止符，并且对 UTF-8 等多字节编码数的是字节，不是用户看到的字符个数。
<!-- bilingual-en:start -->
`word` is a string; `raw` is merely an array of three characters. Passing `raw` to `strlen`, `%s`, or `strcpy` makes the operation search beyond the object for a terminator, causing undefined behaviour. `strlen` reports bytes before the first null only after validity is established; it neither reports destination capacity nor counts user-perceived characters in a multibyte encoding.
<!-- bilingual-en:end -->

容量必须同时容纳数据和终止符。四字节缓冲区最多保存三个普通单字节字符再加 `\0`。例如 `fgets(buf, sizeof buf, stdin)` 最多为数据使用 `sizeof buf - 1` 个位置，并在成功读入时写终止符；若改用带宽度的 `scanf`，`char buf[4]` 的 `%3s` 才为 `\0` 留下位置。无宽度的 `%s` 无法保护固定缓冲区。
<!-- bilingual-en:start -->
Capacity must cover both data and terminator. A four-byte buffer holds at most three ordinary single-byte characters plus `\0`. `fgets(buf, sizeof buf, stdin)` reserves room for termination on a successful read; a `%s` conversion requires an explicit width such as `%3s` for `char buf[4]`.
<!-- bilingual-en:end -->

若数组内部较早出现 `\0`，标准字符串函数在此停止。例如 `{'a','\0','b','\0'}` 的数组容量是 4，但从首元素开始表示的字符串长度是 1；后面的 `b` 仍在数组里，却不属于这一个字符串的值。
<!-- bilingual-en:start -->
If an earlier null occurs, string operations stop there. An array `{'a','\0','b','\0'}` has capacity four, but the string beginning at its first element has length one; `b` remains in the array but is not part of that string value.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 为什么 `char name[5]` 不能保存五个普通单字节字符后再作为 C 字符串使用？
>
> **答案：** 五个数据字节已占满容量，没有位置保存第一个 `\0`；它最多容纳四个这类字符加终止符。

## 来源与核验

- [ISO C11 committee draft N1570, 7.1.1 paragraph 1, 7.21.6.2, 7.21.7.2, and 7.24.6.3](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对字符串定义、`%s` 字段宽度、`fgets` 的终止规则与 `strlen` 的前置边界。
- [CS50x 2026, Lecture 4 notes](https://cs50.harvard.edu/x/notes/4/)：核对课程的 `HI!\0`、`strlen(s)+1` 与固定缓冲区输入示例。
- [SEI CERT C, STR31-C](https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152048/STR31-C.%2BGuarantee%2Bthat%2Bstorage%2Bfor%2Bstrings%2Bhas%2Bsufficient%2Bspace%2Bfor%2Bcharacter%2Bdata%2Band%2Bthe%2Bnull%2Bterminator)：作为工程建议，核对容量必须包含终止符；规范定义仍来自 N1570。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 7.1.1 paragraph 1, 7.21.6.2, 7.21.7.2, and 7.24.6.3](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for the string definition, `%s` field width, `fgets` termination, and the precondition of `strlen`.
- [CS50x 2026 Lecture 4 notes](https://cs50.harvard.edu/x/notes/4/) were checked for the course's `HI!\0`, `strlen(s)+1`, and fixed-buffer input examples.
- [SEI CERT C STR31-C](https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152048/STR31-C.%2BGuarantee%2Bthat%2Bstorage%2Bfor%2Bstrings%2Bhas%2Bsufficient%2Bspace%2Bfor%2Bcharacter%2Bdata%2Band%2Bthe%2Bnull%2Bterminator) was used only as engineering guidance that capacity includes the terminator; the normative definition comes from N1570.
<!-- bilingual-en:end -->
