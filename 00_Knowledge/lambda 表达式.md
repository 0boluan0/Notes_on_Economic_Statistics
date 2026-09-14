---
aliases:
  - "lambda 表达式创建单表达式匿名函数对象而不是多语句函数体"
  - "A lambda expression creates an anonymous single-expression function object rather than a multi-statement body"
  - "Python lambda"
student_os: knowledge-atom
atom_id: CS-PY-FN-013
atom_set: python-functions
atom_type: expression-form
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Python 函数、作用域、闭包与高阶函数.canvas]]"
requires:
  - "[[一等函数对象]]"
related:
  - "[[高阶函数抽象]]"
  - "[[闭包晚绑定]]"
  - "[[函数定义与调用]]"
---

# lambda 表达式创建单表达式匿名函数对象而不是多语句函数体
<!-- bilingual-en:start -->
*A `lambda` expression creates an anonymous single-expression function object rather than a multi-statement function body*
<!-- bilingual-en:end -->

> [!summary] 原子语法边界
> `lambda parameters: expression` 是一个表达式，求值结果是函数对象；调用该对象时，唯一的 body expression 被求值并作为返回值。lambda 不能包含普通语句或函数注解，而 `def` 可容纳多条语句、docstring、显式 `return` 和注解。两者创建的函数都遵循相同的参数绑定与词法作用域规则。
> <!-- bilingual-en:start -->
> `lambda parameters: expression` is itself an expression whose result is a function object. Calling that object evaluates its sole body expression and returns the result. A lambda cannot contain ordinary statements or function annotations, whereas `def` supports statements, docstrings, explicit `return`, and annotations. Functions from both forms follow the same parameter-binding and lexical-scope rules.
> <!-- bilingual-en:end -->

```python
is_even = lambda x: x % 2 == 0
```

“匿名”只表示 lambda 表达式本身不执行一次名称绑定；生成的对象仍可以像上例一样绑定名称、放进容器并多次调用，所以不能说 lambda 必然只能用一次。短小、局部且含义明显的行为参数适合 lambda；需要多步控制流、清楚文档、稳定 traceback 名称或独立测试时，命名 `def` 通常更可读。

> [!question]- 自检
> 为什么 `lambda x: x + 1` 可以直接传入高阶函数，却不能在冒号后写普通 `if` 语句块？
>
> **答案：** lambda 是产生函数对象的表达式，body 语法只允许一个表达式；多语句函数体要用 `def`。

## 来源与核验

- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec09.pdf#page=3|MIT 6.100L Lecture 9 slides，pp. 3–12]]：核对 lambda 产生匿名函数对象并作为高阶函数实参。
- [Python Language Reference: lambdas](https://docs.python.org/3/reference/expressions.html#lambda)：核对语法、隐式返回、不能包含语句或注解。
- [[03_Computer_Science/02_CS61A/UCB-CS61A-Textbook-1.0.0/Composing Programs - John DeNero.epub|Composing Programs，1.6.7 Lambda Expressions]]：核对 lambda 与 `def` 的可读性和使用边界。
