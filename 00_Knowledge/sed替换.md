---
aliases:
  - "sed替换命令把模式空间中的匹配片段改写为替换文本"
  - "Substitution with sed"
student_os: knowledge-atom
atom_id: CS-TEXT-007
atom_type: method
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[正则表达式]]"
related:
  - "[[正则方言边界]]"
  - "[[贪婪匹配跨越分隔符]]"
---

# sed替换命令把模式空间中的匹配片段改写为替换文本

<!-- bilingual-en:start -->
*sed substitution rewrites a matching part of pattern space with replacement text*
<!-- bilingual-en:end -->

`sed 's/pattern/replacement/'` 对当前模式空间执行替换。普通逐行用法中，模式空间初始是读入的一行，不含行尾换行；默认只替换该行的第一次匹配，增加 `g` 则替换所有不重叠匹配。模式与替换文本属于不同语法层。

<!-- bilingual-en:start -->
`sed 's/pattern/replacement/'` substitutes within the current pattern space. In ordinary line-oriented use, pattern space initially holds one input line without its terminating newline. By default only the first match is replaced; `g` replaces all nonoverlapping matches. The pattern and replacement text use different syntactic rules.
<!-- bilingual-en:end -->

```bash
printf '%s\n' 'cat cat' | sed 's/cat/dog/'
printf '%s\n' 'cat cat' | sed 's/cat/dog/g'
printf '%s\n' 'user=alice' | sed 's/user=\(.*\)/name:\1 [&]/'
```

三条依次输出 `dog cat`、`dog dog`、`name:alice [user=alice]`。替换中的 `&` 表示整个匹配，`\1` 表示模式中第一组捕获；第三条使用 BRE 的 `\(...\)` 捕获写法。要写字面量 `&` 或反斜杠，必须按替换语法转义，不能认为引用后的文本都按字面量插入。

<!-- bilingual-en:start -->
The commands output `dog cat`, `dog dog`, and `name:alice [user=alice]`, respectively. In replacement text, `&` denotes the entire match and `\1` the first captured group; the third command uses BRE capture syntax `\(...\)`. Literal ampersands and backslashes require replacement-syntax escaping: Shell quoting does not make every replacement character literal.
<!-- bilingual-en:end -->

默认情况下，即使没有替换，`sed` 仍会输出该行。若只想输出发生替换的行，可以组合 `-n` 与替换标志 `p`，例如 `sed -n 's/^user=/name=/p'`。这些命令把结果写到标准输出，不会自动改写输入文件；原地编辑选项及行为需要另查具体实现。

<!-- bilingual-en:start -->
By default, `sed` outputs a line even when no substitution occurs. To output only lines where substitution succeeds, combine `-n` with the substitution flag `p`, as in `sed -n 's/^user=/name=/p'`. These commands write results to standard output; they do not automatically modify the input file. In-place editing options and behavior require implementation-specific checking.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [GNU sed, The s Command](https://www.gnu.org/software/sed/manual/html_node/The-_0022s_0022-Command.html)：支持替换语法、第一次与全局替换、`&`、捕获引用及 `p` 标志。
  <!-- bilingual-en:start -->
  This section supports substitution syntax, first versus global replacement, `&`, capture references, and the `p` flag.
  <!-- bilingual-en:end -->
- [POSIX sed, Extended Description](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/sed.html)：支持模式空间、逐行周期、默认输出与 `-n` 抑制默认输出的配合。
  <!-- bilingual-en:start -->
  This specification supports pattern space, the processing cycle, default output, and suppression of that output with `-n`.
  <!-- bilingual-en:end -->
