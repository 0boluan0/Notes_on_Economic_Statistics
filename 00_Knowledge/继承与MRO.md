---
aliases:
  - "继承属性查找沿类的 MRO 线性顺序进行而多重继承不会按每条路径重复搜索"
  - "Inherited attribute lookup follows the class MRO linearization rather than searching every multiple-inheritance path repeatedly"
student_os: knowledge-atom
atom_id: CS-PY-OOP-008
atom_set: python-oop
atom_type: inheritance-lookup
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Python 面向对象、类与继承.canvas]]"
requires:
  - "[[属性查找顺序]]"
implies:
  - "[[super协作调用]]"
  - "[[覆盖与可替换性]]"
related:
  - "[[动态类型边界]]"
---

# 继承属性查找沿类的 MRO 线性顺序进行而多重继承不会按每条路径重复搜索
<!-- bilingual-en:start -->
*Inherited attribute lookup follows the class MRO linearization rather than searching every multiple-inheritance path repeatedly*
<!-- bilingual-en:end -->

> [!summary] 原子继承查找
> 子类会把基类记入自身定义；当当前类没有所需类属性时，查找按该类的 method resolution order（MRO，方法解析顺序）继续。单继承中它近似“子类 → 父类 → 更上层”；多重继承则用一致的线性化顺序处理菱形关系，使每个类只出现一次并保持声明的局部优先级。可用 `C.__mro__` 直接查看真实顺序。
> <!-- bilingual-en:start -->
> A subclass records its base classes. When the current class lacks a requested class attribute, lookup continues along that class's method resolution order (MRO). Under single inheritance this resembles “subclass to parent to further ancestors.” Multiple inheritance instead uses a consistent linearization for diamond relationships, listing each class once while preserving local precedence. `C.__mro__` exposes the actual order.
> <!-- bilingual-en:end -->

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

D.__mro__
# (D, B, C, A, object)
```

不能把多重继承机械记成“深度优先、从左到右”：Python 的 C3 MRO 还要求单调性和一致的父类顺序。若类头给出的基类顺序无法形成一致 MRO，类定义本身会失败，而不是运行时任意挑一条路径。

> [!question]- 回忆与应用
> 在菱形结构 `D(B, C)` 且 `B`、`C` 都继承 `A` 时，为什么不应把 `A` 当作两次独立查找？
>
> **答案：** MRO 把整个层级线性化，每个类只出现一次；`D.__mro__` 给出统一顺序。

## 来源与核验

- [The Python Tutorial: inheritance and multiple inheritance](https://docs.python.org/3/tutorial/classes.html#inheritance)：核对继承查找、覆盖、`isinstance` 与多重继承；其下一节核对 MRO 的一致线性化。
- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec19.pdf#page=17|MIT 6.100L Lecture 19 slides，pp. 17–28]]：核对父类、子类、逐层查找与覆盖的课程边界。
- [[03_Computer_Science/02_CS61A/UCB-CS61A-Textbook-1.0.0/Composing Programs - John DeNero.epub|Composing Programs，2.5.5–2.5.7]]：核对继承、类查找与多重继承的课程模型。
