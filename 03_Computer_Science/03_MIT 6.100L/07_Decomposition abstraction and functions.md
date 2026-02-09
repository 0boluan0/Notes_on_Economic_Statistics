---
date: 2026-02-11
科目: MIT 6.100L
---

# Decomposition, abstraction, and the mechanics of functions

## 本讲主线

- 把复杂系统拆成“黑盒”（抽象）与自包含模块（分解）
- 函数定义/调用/返回时机与作用域
- 用函数解决问题的“纸上推导 + print 调试”流程

## 1. Abstraction + Decomposition

- 手机是输入/输出的黑盒，用户只需了解接口
- 程序也应如此：编写函数隐藏细节，暴露参数/返回值
- 每个部件独立实现，降低交互复杂度

> [!tip] Big Idea
> 抽象（What it does）+ 分解（How it fits into the bigger picture）是写清晰代码的基础。

## 2. Function anatomy

- 组成：函数名 + 参数 + docstring（规范输入/输出）+ 域 + `return`
- 例：

```python
def is_even(i):
    """Input: i, a positive int
    Returns True if i is even, otherwise False"""
    return i % 2 == 0
```

- 定义函数仅把代码注册为对象；只有调用时才运行
- `return` 结束函数并把值交给调用者；之后的行不执行

### 调用与作用域

- 每次函数调用都会创建一个**新环境（scope）**：参数在其中绑定
- 函数内的变量不会影响全局，除非故意使用 `global`
- `print` 语句能帮助 debug，但最终的契约由 `return` 确定

## 3. 从纸上推导到代码

- 写函数前三步：
  1. 明确目标问题（输入/输出）
  2. 在纸上模拟简单例子（例如 `sum_odd(2,4)`）
  3. 分步构建：初始化 accumulator → 遍历（for/while）→ 条件→ 返回

- `sum_odd` 例：
  - 先写 `sum_of_odds = 0`
  - 用 `for i in range(a, b+1)` 或 `while i <= b` 循环
  - 只累加 `i % 2 == 1`
  - `print` 输出中间值帮助 debug（看 `i`、`sum_of_odds` 的变化）

> [!important] Big Idea
> 先在纸上处理一个简单例子，再写代码；每次改动都运行测试，碰到怪结果就加 `print` 调试看。

## 4. 函数的调试与测试习惯

- 让程序输出中间状态，可用 `print(i, sum_of_odds)` 检查流程
- `range(a, b)` 仅到 `b-1`，要加 `+1` 才包含终点
- “一步步交代”比一次写完更容易定位 bug

## 5. 课堂练习（You Try It）

- [ ] 写 `div_by(n, d)`：判断 `d` 是否整除 `n`
- [ ] 在 `sum_odd` 案例里，先实现“加所有数”，再加 `if i % 2 == 1`
- [ ] 写 `is_palindrome(s)`：判断字符串是否回文
- [ ] 尝试在 `sum_odd` 中交替使用 `for` 和 `while`
- [ ] 多运行 `sum_odd(2,4)`、`sum_odd(2,7)` 观察输出，用 print 追踪变量

## 小结

- 函数是“黑盒”：通过参数/返回值隐藏实现细节
- 每次调用产生独立作用域，变量不会交叉污染
- 解复杂问题：先模拟/测试、再写代码，再用 `print` debug，再重构
- 书写函数时多思考“问题是什么”“有哪些输入”“输出是什么”

## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec07.pdf]]
