---
aliases:
  - "try 的 else 分支只在 try 正常完成且未提前转移控制时执行"
  - "The else clause of try"
student_os: knowledge-atom
atom_id: CS-EXC-005
atom_type: execution-rule
status: source-checked
part_of:
  - "[[测试、调试、异常与断言.canvas]]"
requires:
  - "[[异常传播]]"
related:
  - "[[异常捕获边界]]"
  - "[[控制转移语句]]"
---

# try 的 else 分支只在 try 正常完成且未提前转移控制时执行

<!-- bilingual-en:start -->
*A try statement's else clause runs only when try completes normally without an early control transfer*
<!-- bilingual-en:end -->

`try/except/else` 的 `else` 在 `try` 正常执行到末尾时运行：没有异常向该 `try` 报出，也没有由 `return`、`break` 或 `continue` 提前离开它。若某个 `except` 已处理了异常，随后也不会再进入 `else`。`else` 中的新异常不由同一个 `try` 的 `except` 处理。

<!-- bilingual-en:start -->
The `else` in `try/except/else` runs when `try` reaches its end normally: no exception escapes its body and no `return`, `break`, or `continue` leaves it early. Finishing an `except` handler does not then enter `else`. A new exception raised in `else` is not handled by that same statement's `except` clauses.
<!-- bilingual-en:end -->

```python
def reciprocal_text(text):
    try:
        number = int(text)
    except ValueError:
        return None
    else:
        return 1 / number
```

`"cat"` 返回 `None`，`"2"` 返回 `0.5`，`"0"` 则向外抛 `ZeroDivisionError`。这让“解析失败”与“解析成功后的运算失败”保持区别。不要把 `else` 理解成“程序没崩溃就执行”，也不要与[[控制转移语句|循环的 else]] 混用同一判断条件。

<!-- bilingual-en:start -->
`"cat"` returns `None`, `"2"` returns `0.5`, and `"0"` propagates `ZeroDivisionError`. Parsing failure remains distinct from failure during subsequent computation. Do not interpret `else` as “run whenever the program did not crash,” or give it the same condition as a [[控制转移语句|loop's else]].
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec13.pdf#page=8|MIT 6.100L Lecture 13，PDF p. 8]]：核对 `else` 承载 `try` 正常完成后的路径；原页已渲染检查。
  <!-- bilingual-en:start -->
  This slide supports the successful-completion role of `else`; the original page was also checked visually.
  <!-- bilingual-en:end -->
- [Python Language Reference §8.4.3, else clause](https://docs.python.org/3/reference/compound_stmts.html#else-clause)：补足 `return/break/continue` 边界及 `else` 不受同组处理器保护的语义；三种输入已执行核对。
  <!-- bilingual-en:start -->
  This section supplies the early-transfer boundary and the fact that sibling handlers do not protect `else`. All three inputs were checked by execution.
  <!-- bilingual-en:end -->
