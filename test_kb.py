#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from link_knowledge import KnowledgeBase

def test_kb():
    """测试知识库索引功能"""
    factor_dir = "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/00_factor"
    kb = KnowledgeBase(factor_dir)
    kb.load_entries()

    print(f"知识点卡片数量: {len(kb.entries)}")
    print()

    print("前 10 个知识点卡片:")
    for i, entry in enumerate(kb.entries[:10]):
        print(f"{i+1}. 文件名: {entry['filename']}")
        print(f"   别名: {entry['aliases']}")
        print(f"   科目: {entry['subject']}")
        print(f"   标签: {entry['tags']}")
        print()

    print("关键词映射示例（前 20 个）:")
    for i, (keyword, entries) in enumerate(list(kb.keyword_map.items())[:20]):
        filenames = [entry['filename'] for entry in entries]
        print(f"{i+1}. {keyword} → {filenames}")

    # 测试关键词匹配
    test_text = """
    今天学习了 OLS 回归和 DID 模型，还有工具变量 IV 的识别原理。
    异方差和自相关问题会影响回归结果的可信度。
    """

    matches = kb.find_matches(test_text)
    print()
    print("测试文本中的匹配结果:")
    for match in matches:
        print(f"- 关键词: '{match['keyword']}'")
        print(f"  位置: {match['start']} - {match['end']}")
        print(f"  匹配的文件: {[e['filename'] for e in match['entries']]}")

if __name__ == "__main__":
    test_kb()
