---
aliases:
  - "awk是对输入记录按模式执行动作的文本处理语言"
  - "awk is a pattern–action text-processing language"
student_os: knowledge-atom
atom_id: CS-TEXT-008
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
leads_to:
  - "[[awk条件聚合]]"
related:
  - "[[字段切分不等于格式解析]]"
  - "[[正则表达式]]"
---

# awk是对输入记录按模式执行动作的文本处理语言

<!-- bilingual-en:start -->
*awk is a text-processing language that applies pattern–action rules to input records*
<!-- bilingual-en:end -->

`awk` 是一种按“模式 `{ 动作 }`”规则处理记录的语言，也是运行这种程序的命令。它通常逐行读取文本，把一行作为一个记录，依次检查程序中的规则，并对满足模式的记录执行相应动作。省略模式表示每条记录都执行；省略动作则默认输出整条记录。

<!-- bilingual-en:start -->
`awk` is a language for processing records through `pattern { action }` rules, as well as the command that runs such programs. It normally reads text line by line, treats each line as a record, checks the rules in program order, and executes actions whose patterns match. Omitting a pattern applies an action to every record; omitting an action defaults to printing the complete record.
<!-- bilingual-en:end -->

`$0` 表示当前完整记录，`$1`、`$2` 等表示字段，`NF` 是当前字段数，`NR` 是累计读入的记录序号。默认字段分隔 `FS=" "` 有特殊空白规则：连续空格或 TAB 作为分隔，开头结尾空白不形成空字段。它与按每个逗号或 TAB 单独切分的契约不同。

<!-- bilingual-en:start -->
`$0` is the complete current record; `$1`, `$2`, and so on are fields. `NF` is the current field count, and `NR` is the cumulative record number. The default field separator `FS=" "` has special whitespace rules: runs of spaces or tabs separate fields, and leading or trailing whitespace does not create empty fields. This differs from splitting at each comma or tab under an explicit delimiter contract.
<!-- bilingual-en:end -->

```bash
printf '%s\n' 'GET / 200' 'POST /login 403' | awk '{print NR, NF, $3}'
```

结果为 `1 3 200` 和 `2 3 403`。程序中的 `$3` 由 `awk` 解释，外层单引号阻止 Shell 先展开它。`awk '$3'` 不是“打印第三字段”：那是把第三字段当条件，满足时执行默认整行打印；提取字段需要 `{print $3}`。

<!-- bilingual-en:start -->
The result is `1 3 200` followed by `2 3 403`. Here `$3` is interpreted by awk; the surrounding single quotes prevent prior Shell expansion. `awk '$3'` does not mean “print the third field”: it uses that field as a condition and performs default whole-record printing when true. Field extraction requires `{print $3}`.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [POSIX awk, Overall Program Structure、Expressions and Special Variables](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/awk.html)：支持模式—动作执行、默认记录、`$0/$n`、`NF/NR` 与省略动作的规则。
  <!-- bilingual-en:start -->
  These sections support pattern–action execution, default records, `$0/$n`, `NF/NR`, and the default action.
  <!-- bilingual-en:end -->
- [GNU awk, How Fields Are Separated](https://www.gnu.org/software/gawk/manual/html_node/Default-Field-Splitting.html)：支持默认空白分字段与显式单字符分隔符之间的区别。
  <!-- bilingual-en:start -->
  This section supports the distinction between default whitespace splitting and an explicit single-character separator.
  <!-- bilingual-en:end -->
