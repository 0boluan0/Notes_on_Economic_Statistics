#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将课堂笔记中知识点卡片链接化的脚本
功能：
1. 索引 00_factor/ 目录下的所有知识点卡片
2. 遍历 01_Math/ 和 02_Economy/ 目录下的笔记
3. 自动识别并添加 Obsidian 双链链接
"""

import os
import re
import yaml
import markdown
from pathlib import Path


class KnowledgeBase:
    """知识点卡片索引库"""

    def __init__(self, factor_dir):
        self.factor_dir = Path(factor_dir)
        self.entries = []
        self.keyword_map = {}

    def load_entries(self):
        """加载所有知识点卡片信息"""
        # 遍历 00_factor/ 目录及其子目录下的所有 .md 文件
        for md_file in self.factor_dir.rglob("*.md"):
            # 跳过隐藏文件和报告文件
            if md_file.name.startswith(".") or md_file.name.startswith("_"):
                continue
            try:
                content = md_file.read_text(encoding="utf-8")

                # 解析 frontmatter
                frontmatter = self._parse_frontmatter(content)
                filename = md_file.stem  # 不带 .md 的文件名

                entry = {
                    "filename": filename,
                    "path": str(md_file),
                    "aliases": frontmatter.get("aliases", []),
                    "subject": frontmatter.get("科目", ""),
                    "tags": frontmatter.get("tags", []),
                }

                self.entries.append(entry)

                # 建立关键词映射：文件名和所有别名都作为关键词
                keywords = [filename] + entry["aliases"]
                for keyword in keywords:
                    if keyword:
                        # 预处理关键词：去除首尾空格，转换为小写
                        processed_keyword = keyword.strip().lower()
                        if processed_keyword not in self.keyword_map:
                            self.keyword_map[processed_keyword] = []
                        # 避免重复添加同一个文件
                        if filename not in [e["filename"] for e in self.keyword_map[processed_keyword]]:
                            self.keyword_map[processed_keyword].append(entry)

            except Exception as e:
                print(f"警告：无法处理文件 {md_file}: {e}")

        print(f"成功加载 {len(self.entries)} 个知识点卡片")
        print(f"建立了 {len(self.keyword_map)} 个关键词映射")

    def _parse_frontmatter(self, content):
        """解析 Markdown 文件的 YAML frontmatter"""
        frontmatter = {}
        if content.startswith("---"):
            end_index = content.find("---", 3)
            if end_index != -1:
                try:
                    frontmatter = yaml.safe_load(content[3:end_index])
                except Exception as e:
                    print(f"警告：解析 frontmatter 失败: {e}")
        return frontmatter

    def find_matches(self, text):
        """在文本中查找匹配的知识点"""
        matches = []
        # 按关键词长度降序排序，避免短关键词匹配长关键词的问题
        sorted_keywords = sorted(self.keyword_map.keys(), key=len, reverse=True)

        for keyword in sorted_keywords:
            # 严格控制关键词长度，避免过短的关键词导致过度匹配
            if len(keyword) < 3:
                continue

            # 对于包含中文的关键词，要求更长的长度
            if re.search(r'[\u4e00-\u9fff]', keyword) and len(keyword) < 4:
                continue

            # 使用正则表达式匹配
            if re.match(r'^[a-zA-Z0-9_]+$', keyword):
                # 英文关键词使用单词边界
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            else:
                # 中文关键词使用严格的边界检测
                # 只在中文标点、空格、换行、英文单词边界等位置匹配
                pattern = re.compile(
                    r'(?<=[^\w\u4e00-\u9fff]|^)' + re.escape(keyword) + r'(?=[^\w\u4e00-\u9fff]|$)',
                    re.IGNORECASE
                )

            for match in pattern.finditer(text):
                # 检查匹配是否在代码块或 LaTeX 公式中
                if not self._is_in_special_block(text, match.start(), match.end()):
                    matches.append({
                        "keyword": keyword,
                        "start": match.start(),
                        "end": match.end(),
                        "entries": self.keyword_map[keyword]
                    })

        # 去重：如果多个匹配重叠，保留最长的匹配
        filtered_matches = []
        for match in matches:
            overlaps = False
            for existing in filtered_matches:
                if (match["start"] < existing["end"] and match["end"] > existing["start"]):
                    overlaps = True
                    # 保留长度更长的匹配
                    if len(match["keyword"]) > len(existing["keyword"]):
                        filtered_matches.remove(existing)
                        filtered_matches.append(match)
                    break
            if not overlaps:
                filtered_matches.append(match)

        return filtered_matches

    def _is_in_special_block(self, text, start, end):
        """检查位置是否在代码块或 LaTeX 公式中"""
        # 检查代码块
        code_blocks = self._find_code_blocks(text)
        for cb_start, cb_end in code_blocks:
            if start >= cb_start and end <= cb_end:
                return True

        # 检查 LaTeX 公式
        latex_blocks = self._find_latex_blocks(text)
        for lb_start, lb_end in latex_blocks:
            if start >= lb_start and end <= lb_end:
                return True

        return False

    def _find_code_blocks(self, text):
        """找到所有代码块位置"""
        blocks = []
        # 匹配三个反引号的代码块
        pattern = re.compile(r"```[\s\S]*?```")
        for match in pattern.finditer(text):
            blocks.append((match.start(), match.end()))
        return blocks

    def _find_latex_blocks(self, text):
        """找到所有 LaTeX 公式块位置"""
        blocks = []
        # 匹配 $$...$$
        pattern = re.compile(r"\$\$[\s\S]*?\$\$")
        for match in pattern.finditer(text):
            blocks.append((match.start(), match.end()))
        # 匹配 $...$
        pattern = re.compile(r"\$[^\$]*?\$")
        for match in pattern.finditer(text):
            blocks.append((match.start(), match.end()))
        return blocks


class LinkGenerator:
    """链接生成器"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def generate_links(self, content):
        """处理 Markdown 内容，生成链接"""
        matches = self.kb.find_matches(content)
        matches = sorted(matches, key=lambda x: x["start"])

        # 替换内容（从后往前替换，避免位置偏移）
        modified_content = content
        offset = 0
        for match in sorted(matches, key=lambda x: x["start"], reverse=True):
            original_start = match["start"]
            original_end = match["end"]
            keyword = content[original_start:original_end]

            # 选择最合适的链接目标（优先选择同名或同主题的）
            best_entry = self._select_best_entry(match["entries"], keyword)

            # 生成链接
            link_text = keyword
            link = f"[[{best_entry['filename']}|{link_text}]]"

            # 替换
            start = original_start + offset
            end = original_end + offset
            modified_content = modified_content[:start] + link + modified_content[end:]
            offset += len(link) - (original_end - original_start)

        return modified_content

    def _select_best_entry(self, entries, keyword):
        """选择最合适的链接目标"""
        if len(entries) == 1:
            return entries[0]

        # 简单的选择策略：
        # 1. 优先选择与关键词完全匹配的
        for entry in entries:
            if keyword.lower() == entry["filename"].lower() or keyword.lower() in [a.lower() for a in entry["aliases"]]:
                return entry

        # 2. 选择第一个匹配项
        return entries[0]


class Linker:
    """执行器"""

    def __init__(self, source_dirs, factor_dir, output_dir=None):
        self.source_dirs = [Path(d) for d in source_dirs]
        self.factor_dir = Path(factor_dir)
        self.output_dir = Path(output_dir) if output_dir else None

        # 初始化知识库
        self.kb = KnowledgeBase(factor_dir)
        self.kb.load_entries()

        # 初始化链接生成器
        self.link_generator = LinkGenerator(self.kb)

    def process_notes(self):
        """处理所有笔记"""
        total_files = 0
        processed_files = 0

        for source_dir in self.source_dirs:
            if not source_dir.exists() or not source_dir.is_dir():
                print(f"警告：目录 {source_dir} 不存在或不是目录")
                continue

            for md_file in source_dir.rglob("*.md"):
                total_files += 1
                try:
                    content = md_file.read_text(encoding="utf-8")

                    # 生成链接
                    modified_content = self.link_generator.generate_links(content)

                    # 确定输出路径
                    if self.output_dir:
                        relative_path = md_file.relative_to(source_dir)
                        output_path = self.output_dir / relative_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        # 默认覆盖原文件（生产环境建议使用输出目录）
                        output_path = md_file

                    output_path.write_text(modified_content, encoding="utf-8")
                    processed_files += 1
                    print(f"成功处理：{md_file}")

                except Exception as e:
                    print(f"错误：无法处理文件 {md_file}: {e}")

        print(f"\n处理完成！")
        print(f"总文件数：{total_files}")
        print(f"成功处理：{processed_files}")
        print(f"失败：{total_files - processed_files}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="将课堂笔记中知识点卡片链接化的脚本"
    )
    parser.add_argument(
        "--factor-dir",
        default="/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/00_factor",
        help="知识点卡片目录 (00_factor)",
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=[
            "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/01_Math",
            "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/02_Economy"
        ],
        help="要处理的笔记目录列表 (01_Math, 02_Economy)",
    )
    parser.add_argument(
        "--output-dir",
        help="输出目录（如果不指定，将覆盖原文件）",
    )

    args = parser.parse_args()

    linker = Linker(args.source_dirs, args.factor_dir, args.output_dir)
    linker.process_notes()


if __name__ == "__main__":
    main()
