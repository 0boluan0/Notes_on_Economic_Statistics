---
student_os: knowledge-atom
atom_id: CS-CLI-005
atom_type: definition
aliases:
  - dotfiles是工具读取的用户配置文件
  - User configuration dotfiles
status: source-checked
related:
  - "[[Bash启动文件选择]]"
leads_to:
  - "[[dotfiles配置管理]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# dotfiles是工具读取的用户配置文件
<!-- bilingual-en:start -->
*Dotfiles are user configuration files read by tools.*
<!-- bilingual-en:end -->

Dotfiles 通常指保存工具用户配置的文件，例如 `.bashrc`、`.vimrc`、`.gitconfig`。名称来自传统上以点开头、默认不被 `ls` 列出的文件；配置也可能位于专门配置目录。
<!-- bilingual-en:start -->
Dotfiles commonly mean files holding a tool's user configuration, such as `.bashrc`, `.vimrc`, and `.gitconfig`. The name comes from traditional dot-prefixed files omitted by plain `ls`; configuration may also live in dedicated directories.
<!-- bilingual-en:end -->

隐藏是显示约定，不是用途定义。缓存、命令历史和私钥也可能隐藏，却不因此成为用户配置文件；文件何时被读取取决于工具规则，例如 [[Bash启动文件选择]]。版本化与安装这些配置的方法另见[[dotfiles配置管理]]。
<!-- bilingual-en:start -->
Hidden status is a display convention, not a definition of purpose. Caches, command histories, and private keys may also be hidden without being user configuration files. Loading follows each tool's rules, such as [[Bash启动文件选择|Bash startup-file selection]]. Versioning and installation belong to [[dotfiles配置管理|dotfile configuration management]].
<!-- bilingual-en:end -->

## 来源与核验

[MIT Missing Semester 2020, Dotfiles](https://missing.csail.mit.edu/2020/command-line/#dotfiles)：支持用户配置文件的含义、点开头名称、默认隐藏和各工具的实例。隐藏不决定文件用途，是据此区分配置与其他用户状态的定义边界。
<!-- bilingual-en:start -->
[MIT Missing Semester 2020: Dotfiles](https://missing.csail.mit.edu/2020/command-line/#dotfiles) supports user configuration files, dot-prefixed names, default hiding, and tool examples. The distinction between configuration and other user state follows because hidden status does not determine purpose.
<!-- bilingual-en:end -->
