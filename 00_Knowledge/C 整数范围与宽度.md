---
aliases:
  - "C 整数类型的范围有限而具体宽度需要由实现或定宽类型确认"
  - C integer ranges
  - C 整数宽度与范围
student_os: knowledge-atom
atom_id: CS-C-004
atom_set: c-foundations
atom_type: boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[C 语言基础：编译、类型、控制流与函数.canvas]]"
---

# C 整数类型的范围有限而具体宽度需要由实现或定宽类型确认
<!-- bilingual-en:start -->
*C integer types have finite ranges, while exact widths must be confirmed from the implementation or fixed-width types*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> C 保证整数类型的最小能力和相对等级，却不把 `int`、`long` 等类型在所有平台上固定为同一位数。写算法时必须同时回答：所需值域是什么、当前实现的类型范围是什么、跨平台接口是否需要定宽整数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> C guarantees minimum capabilities and a ranking among integer types, but it does not assign `int`, `long`, and related types one universal width. An algorithm must therefore answer three questions: what range the values require, what range the implementation provides, and whether a portable interface needs a fixed-width integer type.
> <!-- bilingual-en:end -->

`<limits.h>` 给出当前实现中 `CHAR_BIT`、`INT_MAX`、`LONG_MIN` 等边界：`sizeof` 以字节计数，而每字节含多少 bit 要看 `CHAR_BIT`。`<stdint.h>` 在实现存在符合标准条件的相应无填充精确宽度类型时给出 `int32_t` 这类名字；在 C11 中，有符号精确宽度类型还必须使用二进制补码表示。`int_least32_t` 保证至少 32 位，`int_fast32_t` 则是实现为通常较快运算选择的至少 32 位类型，并不保证在每个用途上绝对最快。`size_t` 是 `sizeof` 结果所用的无符号整数类型，足以表示实现支持的对象字节大小；它的具体底层类型并不固定，也不是一般意义上的有符号指针差类型。
<!-- bilingual-en:start -->
`<limits.h>` exposes implementation limits such as `CHAR_BIT`, `INT_MAX`, and `LONG_MIN`: `sizeof` counts bytes, while `CHAR_BIT` states how many bits a byte contains. When a corresponding unpadded exact-width type meeting the standard's conditions exists, `<stdint.h>` supplies a name such as `int32_t`; in C11, a signed exact-width type must also use two's-complement representation. `int_least32_t` guarantees at least 32 bits, while `int_fast32_t` is an at-least-32-bit type chosen by the implementation as usually fast, not as absolutely fastest for every purpose. `size_t` is the unsigned integer type of a `sizeof` result and can represent supported object sizes in bytes. Its underlying type is not fixed, and it is not the general signed type for pointer differences.
<!-- bilingual-en:end -->

## 选择类型时看语义而不是习惯

- 计数可能超过 `int` 时，先由上界推出所需范围，而不是事后等溢出出现。
- 磁盘格式、网络协议或二进制接口若规定精确位宽，才用相应定宽类型并同时处理字节序。
- 普通循环索引若与 `sizeof` 或容器大小比较，`size_t` 往往能避免不必要的有符号/无符号错配，但倒序循环要特别小心它不能表示负数。
<!-- bilingual-en:start -->
- If a count may exceed `int`, derive the required range from an upper bound before overflow appears.
- Use an exact-width type for a disk format, network protocol, or binary interface only when that contract fixes the width, and specify byte order separately.
- For loop indices compared with `sizeof` or an object count, `size_t` often avoids a needless signed–unsigned mismatch; reverse loops need special care because it cannot represent negative values.
<!-- bilingual-en:end -->

## 边界

定宽类型解决的是表示宽度，不自动解决溢出、序列化字节序或算术单位。类型足够宽也不等于表达式的中间结果足够宽；中间运算使用什么类型还受整数提升与通常算术转换影响。
<!-- bilingual-en:start -->
Fixed-width types solve representation width, not overflow, serialisation byte order, or units. Even if the destination type is wide enough, an intermediate expression may still be evaluated in a narrower type because of integer promotions and the usual arithmetic conversions.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个文件格式明确规定字段是无符号 32 位整数。为什么只写 `unsigned long` 不足以表达这个契约？
>
> **答案：** `unsigned long` 的精确宽度随实现而变；应确认 `uint32_t` 可用，并另外规定序列化字节序。

## 来源与核验

- [ISO C11 committee draft N1570, 5.2.4.2.1, 6.2.5, 7.19 and 7.20](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对整数最小范围、标准整数类型、`size_t` 与定宽整数类型。
<!-- bilingual-en:start -->
- [ISO C11 committee draft N1570, 5.2.4.2.1, 6.2.5, 7.19, and 7.20](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for minimum integer ranges, standard integer types, `size_t`, and fixed-width integer types.
<!-- bilingual-en:end -->
