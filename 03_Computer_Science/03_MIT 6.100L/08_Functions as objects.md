---
date: 2026-02-12
科目: MIT 6.100L
---

# Functions as objects, scope, and higher-order procedures

## 本讲主线

- 函数就是对象，可以赋值/传递/返回；Python 是一等函数语言
- 每个调用都会创建新的环境（scope），局部变量与全局隔离
- 把函数当参数/返回值写出更简洁的组合逻辑

## 1. 函数总有返回值

- 即使没有 `return`，函数也会返回 `None`
- `return` 语句在执行后立即结束函数，之后的行不执行
- `print` 只是副作用，`return` 才是结果

### 示例对比

```python
def add(x, y):
    return x + y

def mult(x, y):
    print(x * y)

print(add(2, 3))  # 5
print(mult(4, 5)) # None，函数里只打印
```

## 2. Scope：每次调用都是新环境

- 调用函数时，Python 为该函数创建一个新的 scope
- 参数在该 scope 内绑定；作用域在 `return` 后销毁
- 函数内不能直接修改外部变量（除非用 `global`，但不推荐）

> [!important] Big Idea
> 你需要知道自己正在执行哪个作用域里的表达式，才能理解变量的绑定。

- 例如：`calc(add, 2, 3)`—先在 `calc` 的 scope 中 `x=2`, `y=3`，再进入 `add` 的 scope
- 打印 `x` 在不同 scope 会得到不同值；调试时用 Python Tutor 逐步观察

## 3. 函数就是对象

- 函数有类型，可以和 `int`、`str` 一样赋值给变量
- `my_func = is_even` 之后 `my_func(4)` 和 `is_even(4)` 效果相同
- 既可做参数，也可作为返回值

## 4. 高阶函数（functions as arguments）

- `calc(op, x, y)` 接受一个操作函数 `op`，返回 `op(x, y)`

```python
def calc(op, x, y):
    return op(x, y)

def add(a, b):
    return a + b

def div(a, b):
    if b != 0:
        return a / b
    print("Denominator was 0.")

print(calc(add, 2, 3))  # 5
print(calc(div, 4, 0))  # prints error
```

- 从程序的角度，每次传入函数就会创建 `calc` scope；`op` 被绑定为函数本身，`op(x, y)` 又会创建 `add`/`div` 的 scope

### 例子：`apply(criteria, n)`

- `criteria` 是一个接受数字返回布尔的函数
- `apply` 遍历 `0..n`，统计 `criteria(i)` 为 `True` 的个数

## 5. 课堂练习（You Try It）

- [ ] 手写 `calc` 的执行过程：哪些 scope 是什么时候创建的？
- [ ] 分析 `calc(div, 2, 0)` 会打印什么、返回什么
- [ ] 实现 `apply(criteria, n)`，尝试用 `lambda` 或已有函数作为 `criteria`
- [ ] 设计一个 `count_nums_with_sqrt_close_to(n, epsilon)`，借助前几讲的 `bisection_root`
- [ ] 尝试用 `func_c(func_b, 3)` 这样的 trace 推理返回值

## 小结

- 函数是对象，可以像其它类型一样存储/传递
- 每次调用都会创建新 scope，函数体只在被调用时运行
- 高阶函数让我们写出更抽象、可组合的逻辑
- 理解作用域有助于 debug：知道值在哪个环境中绑定

## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec08.pdf]]
