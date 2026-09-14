---
aliases:
  - "super 返回沿接收者 MRO 从当前类之后继续查找的代理而不是固定父类调用"
  - "super returns a proxy that continues lookup after the current class in the receiver MRO rather than calling one fixed parent"
student_os: knowledge-atom
atom_id: CS-PY-OOP-009
atom_set: python-oop
atom_type: cooperative-dispatch
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Python 面向对象、类与继承.canvas]]"
requires:
  - "[[继承与MRO]]"
related:
  - "[[覆盖与可替换性]]"
  - "[[函数调用帧]]"
---

# super 返回沿接收者 MRO 从当前类之后继续查找的代理而不是固定父类调用
<!-- bilingual-en:start -->
*`super` returns a proxy that continues lookup after the current class in the receiver's MRO rather than calling one fixed parent*
<!-- bilingual-en:end -->

> [!summary] 原子协作规则
> 在类体中编译的普通实例方法或类方法里，零参数 `super()` 近似于 `super(__class__, first_arg)`：编译器提供词法 `__class__` 闭包单元，紧邻函数的首参数通常是 `self` 或 `cls`。代理从该接收者 MRO 中 `__class__` 之后的位置继续查找。因此 `super().method()` 可能调用父类，也可能在多重继承中调用同一 MRO 上的兄弟类实现，不能翻译成“固定调用我的直接父类”。
> <!-- bilingual-en:start -->
> In an ordinary instance method or class method compiled in a class body, zero-argument `super()` is approximately `super(__class__, first_arg)`: the compiler supplies the lexical `__class__` closure cell, while the immediately enclosing function's first argument is usually `self` or `cls`. The proxy continues lookup after `__class__` in that receiver's MRO. Therefore `super().method()` may reach a parent implementation or, under multiple inheritance, a sibling implementation later in the same MRO; it is not a fixed “call my direct parent.”
> <!-- bilingual-en:end -->

```python
class A:
    def run(self):
        return ["A"]

class B(A):
    def run(self):
        return ["B"] + super().run()

class C(A):
    def run(self):
        return ["C"] + super().run()

class D(B, C):
    pass

D().run()                 # ["B", "C", "A"]
```

要让多重继承真正协作，同一链上的实现应接受兼容参数，并在需要继续链时都调用 `super()`；某一层硬编码 `A.method(self)` 或中途不继续，会绕开或截断 MRO。嵌套函数和生成器表达式会建立新的函数作用域，零参数形式不能想当然沿用外层方法的首参数。

> [!question]- 回忆与应用
> 为什么 `B.run` 中的 `super().run()` 在 `D(B, C)` 实例上先到 `C.run`，而不是直接到 `A.run`？
>
> **答案：** `D` 的 MRO 是 `D, B, C, A, object`；`super()` 从 `B` 之后继续这个接收者的 MRO。

## 来源与核验

- [Python Built-in Functions: `super`](https://docs.python.org/3/library/functions.html#super)：核对查找起点、接收者 MRO、零参数形式与 cooperative multiple inheritance 的签名条件。
- [The Python Tutorial: multiple inheritance](https://docs.python.org/3/tutorial/classes.html#multiple-inheritance)：核对动态 MRO 与 cooperative calls 的语言语境。
- [[03_Computer_Science/02_CS61A/UCB-CS61A-Textbook-1.0.0/Composing Programs - John DeNero.epub|Composing Programs，2.5.7 Multiple Inheritance]]：核对课程中的多重继承与共享基类边界；`super` 的精确代理语义以官方文档为准。
