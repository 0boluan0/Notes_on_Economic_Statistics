---
aliases:
  - "global 把名称解释为模块绑定而 nonlocal 指向最近已有的外层函数绑定"
  - "global targets the module binding while nonlocal targets the nearest existing enclosing function binding"
  - "Python global 与 nonlocal"
student_os: knowledge-atom
atom_id: CS-PY-FN-008
atom_set: python-functions
atom_type: rebinding-directive
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Python 函数、作用域、闭包与高阶函数.canvas]]"
requires:
  - "[[局部名称判定]]"
  - "[[词法作用域]]"
related:
  - "[[闭包环境保留]]"
  - "[[名称绑定与赋值]]"
---

# global 把名称解释为模块绑定而 nonlocal 指向最近已有的外层函数绑定
<!-- bilingual-en:start -->
*`global` targets the module binding, whereas `nonlocal` targets the nearest existing binding in an enclosing function scope*
<!-- bilingual-en:end -->

> [!summary] 原子改绑边界
> 函数中的 `global name` 让该代码块对 `name` 的使用和赋值指向定义此函数的模块全局命名空间；`nonlocal name` 则指向最近一个已经绑定该名称的外层函数作用域。两者都是作用于整个当前代码块的解析指令，不会复制、同步或自动保护所引用的对象。
> <!-- bilingual-en:start -->
> Inside a function, `global name` makes uses and assignments of `name` target the defining module's global namespace, while `nonlocal name` targets the nearest enclosing function scope that already binds it. Both are whole-block name-resolution directives; neither copies, synchronizes, nor protects the referenced object.
> <!-- bilingual-en:end -->

`nonlocal` 不能凭空在外层创建名称：若任何外层函数作用域都没有既有绑定，会在编译时得到 `SyntaxError`。`global` 可以让后续赋值在模块命名空间创建绑定。声明必须出现在同一代码块对该名称的使用或赋值之前。若只是读取自由名称，两种声明通常都不需要；它们解决的是改绑目标，而不是读取权限。

在模块层，名称本来就是全局绑定，因而 `global` 不改变绑定层级；`nonlocal` 因没有外层函数作用域而非法。同一函数的形参不能同时声明为 `global` 或 `nonlocal`，同一名称也不能在同一代码块同时声明为两者；这些冲突都在执行函数体之前引发 `SyntaxError`。两种声明都是当前编译代码块的解析指令，不会穿透到另行传给 `exec()` 的字符串，也不会由那个字符串反向改变外层代码块。

隐式共享状态会扩大推理和测试范围。若状态变化不是接口本身的主题，优先通过返回值显式传递新状态；若闭包计数器等状态演化就是目标，再用 `nonlocal` 并把副作用写进契约。

> [!question]- 自检
> 为什么内层函数可以不写 `nonlocal` 就读取外层 `count`，但要执行 `count += 1` 通常必须写？
>
> **答案：** 单纯读取按词法作用域解析自由名称；赋值会让名称默认成为内层局部，`nonlocal` 才把改绑目标重定向到既有外层绑定。

## 来源与核验

- [Python Language Reference: the `global` statement](https://docs.python.org/3/reference/simple_stmts.html#the-global-statement)：核对模块全局目标、整块作用域与声明顺序。
- [Python Language Reference: the `nonlocal` statement](https://docs.python.org/3/reference/simple_stmts.html#the-nonlocal-statement)：核对最近既有外层绑定、缺失绑定的 `SyntaxError` 与声明顺序。
- [Python Language Reference: naming and binding](https://docs.python.org/3/reference/executionmodel.html#naming-and-binding)：核对两种声明对局部名称判定和解析的影响。
