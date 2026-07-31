---
aliases:
  - "C Programming Foundations"
  - "C语言基础"
status: source-checked
---

# C 语言基础：编译、类型、控制流与函数
<!-- bilingual-en:start -->
*C Foundations: Compilation, Types, Control Flow, and Functions*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用接近机器内存模型的语言表达控制流和数据，理解源代码怎样被编译成程序。
> **具体锚点：** `mean(values, n)` 可能分别在编译、链接或运行时失败；错误出现在哪个阶段，决定先检查声明、定义还是输入边界。
> **核心难点：** C 的类型和控制流接近机器执行方式；必须分清值传递、作用域、整数表示和循环不变量。
> **为什么重要：** C 让函数、内存、编译和数据结构的底层关系可见，是理解高级语言抽象成本的好入口。
> **继续：** 先会编译、类型、分支、循环和函数，再进入数组、字符串、指针和动态内存。
> <!-- bilingual-en:start -->
> **Problem addressed:** Express data and control flow in a language close to the machine memory model, and understand how source code becomes an executable program.
> **Concrete anchor:** A call to `mean(values, n)` can fail during compilation, linking, or execution; the stage of failure tells you whether to inspect declarations, definitions, or input bounds first.
> **Central difficulty:** C exposes machine-level execution choices, so you must distinguish value passing, scope, integer representation, and loop invariants.
> **Why it matters:** C exposes the relationships among functions, memory, compilation, and data structures, making the cost of higher-level abstractions visible.
> **Continue with:** Learn compilation, types, branching, loops, and functions before moving to [[C 指针、数组、字符串与动态内存|pointers and dynamic memory]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [CS50x 2026 official course](https://cs50.harvard.edu/x/)：核验 C、内存、算法与课程范围。
> <!-- bilingual-en:start -->
> - The [official CS50x 2026 course](https://cs50.harvard.edu/x/) verifies C, memory, algorithms, and the course scope.
> <!-- bilingual-en:end -->

## 从源代码到程序
<!-- bilingual-en:start -->
*From Source Code to a Program*
<!-- bilingual-en:end -->

预处理展开头文件/宏，编译把 C 翻译为汇编，汇编生成目标代码，链接把目标文件和库合成可执行文件。编译错误、链接错误和运行时错误处在不同阶段；先读第一条诊断并缩小最小复现。
<!-- bilingual-en:start -->
Preprocessing expands headers and macros, compilation translates C into assembly, assembly produces object code, and linking combines objects and libraries into an executable. Compile-time, link-time, and runtime failures occur at different stages; read the first diagnostic and reduce the program to a minimal reproduction.
<!-- bilingual-en:end -->

阶段决定排查对象：缺少声明通常在编译期出现，声明存在但定义未链接会在链接期出现，越界或空指针则常到运行期才暴露。
<!-- bilingual-en:start -->
The stage determines what to inspect: a missing declaration usually appears during compilation, a declared but unlinked definition at link time, and an out-of-bounds access or null dereference often only at runtime.
<!-- bilingual-en:end -->

## 类型、表达式与整数
<!-- bilingual-en:start -->
*Types, Expressions, and Integers*
<!-- bilingual-en:end -->

`char/int/long/float/double` 有有限范围和表示。整数除法截断，signed overflow 在 C 中是未定义行为，unsigned 按模运算。隐式转换可能改变符号和精度；接口处明确类型和范围。
<!-- bilingual-en:start -->
`char`, `int`, `long`, `float`, and `double` have finite ranges and representations. Integer division truncates, signed overflow is undefined behavior in C, and unsigned arithmetic is modular. Implicit conversion can alter sign or precision, so interfaces should state types and ranges explicitly.
<!-- bilingual-en:end -->

## 分支、循环与不变量
<!-- bilingual-en:start -->
*Branches, Loops, and Invariants*
<!-- bilingual-en:end -->

`if/switch` 表达选择，`for/while/do` 表达迭代。每个循环应能说清初始化、保持的不变量和终止度量。`=` 是赋值、`==` 是比较；条件中非零为真并不意味着所有写法都清晰安全。
<!-- bilingual-en:start -->
`if` and `switch` express selection, while `for`, `while`, and `do` express iteration. Every loop should have a clear initialization, invariant, and termination measure. `=` assigns and `==` compares; the rule that nonzero is true does not make every legal condition clear or safe.
<!-- bilingual-en:end -->

## 函数、作用域与接口
<!-- bilingual-en:start -->
*Functions, Scope, and Interfaces*
<!-- bilingual-en:end -->

函数签名说明参数和返回类型；C 默认按值传递，若要修改调用者对象需传地址。局部变量有块作用域和自动存储期，`static`/全局对象生命周期不同。头文件声明接口，源文件提供定义。
<!-- bilingual-en:start -->
A function signature states parameter and return types. C passes arguments by value; modifying a caller-owned object requires passing its address. Local variables have block scope and automatic storage duration, while `static` and global objects have different lifetimes. A header declares an interface and a source file supplies its definition.
<!-- bilingual-en:end -->

## Worked example：区分编译、链接与运行时失败
<!-- bilingual-en:start -->
*Worked Example: Distinguishing Compile, Link, and Runtime Failures*
<!-- bilingual-en:end -->

若调用 `mean(values, n)` 时编译器报告参数类型不匹配，先修正声明或调用；若报告 undefined reference，检查定义是否参与链接；若程序启动后崩溃，再检查 `values` 是否有效及 `n` 是否与数组长度一致。
<!-- bilingual-en:start -->
If a call to `mean(values, n)` produces a compile-time type mismatch, fix the declaration or call. If the linker reports an undefined reference, verify that the definition is linked. If the executable crashes, inspect whether `values` is valid and whether `n` matches the array length.
<!-- bilingual-en:end -->

```c
double mean(const int values[], size_t n) {
    if (n == 0) {
        return 0.0;
    }
    long total = 0;
    for (size_t i = 0; i < n; i++) {
        total += values[i];
    }
    return (double) total / n;
}
```

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 编译器连续报很多错：先修第一条，因为后续诊断可能只是解析已偏离后的连锁反应。
  <!-- bilingual-en:start -->
  The compiler reports many errors: fix the first one because later diagnostics may be cascades from an earlier parse failure.
  <!-- bilingual-en:end -->
- 结果在大输入变负：检查 signed overflow、窄类型累加器和隐式转换，而不只检查循环次数。
  <!-- bilingual-en:start -->
  A result becomes negative for large input: inspect signed overflow, a narrow accumulator, and implicit conversions rather than only the loop count.
  <!-- bilingual-en:end -->
- 函数没有修改调用者变量：确认传入的是地址，并在函数内正确解引用；参数本身仍是地址的副本。
  <!-- bilingual-en:start -->
  A function did not change a caller variable: verify that an address was passed and dereferenced correctly; the parameter itself remains a copy of that address.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 函数如何修改调用者的变量？
<!-- bilingual-en:start -->
*How can a function modify a caller's variable?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 把变量地址传入，函数通过指针解引用修改；参数本身仍是按值传递的地址副本。
> <!-- bilingual-en:start -->
> Pass the variable's address and modify the object through pointer dereference; the parameter itself is still a by-value copy of the address.
> <!-- bilingual-en:end -->

### 为什么 signed overflow 不能按“自动绕回”推理？
<!-- bilingual-en:start -->
*Why must signed overflow not be reasoned about as automatic wraparound?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> C 将其定义为未定义行为，编译器可据“不会溢出”的假设优化；需要更宽类型、显式边界检查或合适的 unsigned 语义。
> <!-- bilingual-en:start -->
> C defines it as undefined behavior, so a compiler may optimize under the assumption that it never occurs; use a wider type, explicit bounds checks, or deliberately chosen unsigned semantics.
> <!-- bilingual-en:end -->

### 一个错误应从哪个构建阶段开始排查？
<!-- bilingual-en:start -->
*At which build stage should an error investigation begin?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 从实际报告它的最早阶段开始：语法/类型看编译，缺定义看链接，非法状态和内存访问看运行时。
> <!-- bilingual-en:start -->
> Begin at the earliest stage that reports it: syntax and types during compilation, missing definitions during linking, and invalid state or memory access at runtime.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [CS50x 2026 official course](https://cs50.harvard.edu/x/)：核验 C、内存、算法与课程范围。
  <!-- bilingual-en:start -->
  The [official CS50x 2026 course](https://cs50.harvard.edu/x/) verifies C, memory, algorithms, and the course scope.
  <!-- bilingual-en:end -->
