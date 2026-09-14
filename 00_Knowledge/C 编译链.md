---
aliases:
  - "预处理、编译、汇编与链接把不同形式的程序逐步交给下一阶段"
  - C build stages
  - C 构建阶段
student_os: knowledge-atom
atom_id: CS-C-001
atom_set: c-foundations
atom_type: process
status: source-checked
mastery_state: unassessed
part_of:
  - "[[C 语言基础：编译、类型、控制流与函数.canvas]]"
---

# 预处理、编译、汇编与链接把不同形式的程序逐步交给下一阶段
<!-- bilingual-en:start -->
*Preprocessing, compilation, assembly, and linking hand progressively lower-level program forms to the next stage*
<!-- bilingual-en:end -->

> [!summary] 原子过程
> 一次常见的 C 构建可以读成：
> $$\text{源文件}\rightarrow\text{预处理后的源代码}\rightarrow\text{汇编代码}\rightarrow\text{目标文件}\rightarrow\text{可执行文件}.$$
> 预处理器处理 `#include`、宏与条件编译；编译器检查并翻译 C；汇编器生成目标代码；链接器把目标文件和所需库中的定义连接起来。每一步接收的对象和能发现的问题都不同。
> <!-- bilingual-en:start -->
>
> &nbsp;
> A common C build can be read as source file → preprocessed source → assembly → object file → executable. The preprocessor handles `#include`, macros, and conditional compilation; the compiler checks and translates C; the assembler produces object code; and the linker connects object files with required definitions from libraries. Each stage receives a different form and can expose different problems.
> <!-- bilingual-en:end -->

## 怎样观察这条流水线

以 GCC 为例，`-E` 停在预处理结果，`-S` 停在汇编代码，`-c` 生成目标文件但不链接。直接运行 `gcc main.c helper.c -o app`，驱动程序会替你依次调用所需阶段。

这也解释了一个常见误会：`#include "helper.h"` 主要是把声明等文本带进当前翻译单元，并不会自动把 `helper.c` 的函数实现加入最终程序；实现仍须编译成目标文件并参与链接。
<!-- bilingual-en:start -->
With GCC, `-E` stops after preprocessing, `-S` after producing assembly, and `-c` after producing an object file without linking. A command such as `gcc main.c helper.c -o app` lets the compiler driver invoke the required stages in sequence.

This also corrects a common misconception: `#include "helper.h"` mainly brings declarations and other text into the current translation unit. It does not automatically add the implementation in `helper.c`; that implementation must still be compiled and linked.
<!-- bilingual-en:end -->

## 边界

ISO C 规定了八个概念性的翻译阶段；“预处理—编译—汇编—链接”则是 GCC、Clang 和 CS50 语境中常用的工具链模型，两者不能逐项等同。真实工具链还可能合并阶段、保留中间表示而不落盘，或在链接时继续优化。因此，这条四步流水线适合解释工具接口和诊断位置，却不是 ISO C 对所有编译器内部命令的承诺。
<!-- bilingual-en:start -->
ISO C specifies eight conceptual translation phases. “Preprocess–compile–assemble–link” is instead the toolchain model commonly used with GCC, Clang, and CS50; the two are not a one-to-one partition. A real toolchain may also fuse stages, keep intermediate representations only in memory, or optimise again at link time. The four-step pipeline is therefore useful for tool interfaces and diagnostic locations, not an ISO C promise about every compiler's internal commands.
<!-- bilingual-en:end -->

> [!question]- 自检
> `main.c` 已包含声明 `double mean(const int *, size_t);`，但定义只写在没有参与构建的 `stats.c` 中。最可能在哪个阶段失败？
>
> **答案：** 编译器已能依据声明检查调用，因此通常到链接时才报告找不到 `mean` 的定义。

## 来源与核验

- [GCC, Overall Options](https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html)：核对预处理、编译、汇编与链接的可停止阶段及 `-E`、`-S`、`-c`。
- [ISO C11 committee draft N1570, 5.1.1.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)：核对翻译阶段与 translation unit 的标准语义。
- [CS50x 2026, Lecture 2 notes](https://cs50.harvard.edu/x/notes/2/)：核对课程语境中的 `make`、Clang 与四阶段解释。
<!-- bilingual-en:start -->
- [GCC, Overall Options](https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html) was checked for the stoppable preprocessing, compilation, assembly, and linking stages and for `-E`, `-S`, and `-c`.
- [ISO C11 committee draft N1570, 5.1.1.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) was checked for translation phases and translation units.
- [CS50x 2026, Lecture 2 notes](https://cs50.harvard.edu/x/notes/2/) were checked for the course treatment of `make`, Clang, and the four-stage model.
<!-- bilingual-en:end -->
