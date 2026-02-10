#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_factor文件夹frontmatter优化脚本
功能：检查并补充完善Markdown文件的aliases和tags字段
"""

import os
import re
import yaml
import argparse


ENGLISH_MULTIWORD_TITLE_PATTERN = re.compile(
    r"[A-Za-z0-9()'&+\-]+(?: [A-Za-z0-9()'&+\-]+)+"
)


def read_file(file_path):
    """读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path, content):
    """写入文件内容"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def extract_frontmatter(content):
    """提取YAML frontmatter"""
    frontmatter_match = re.match(r'^---\s*(.*?)\s*---\s*(.*)$', content, re.DOTALL)
    if frontmatter_match:
        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1))
            body = frontmatter_match.group(2)
            return frontmatter, body
        except Exception as e:
            print(f"Error parsing frontmatter: {e}")
            return None, content
    return None, content


def dedupe_preserve_order(items):
    """列表保序去重"""
    deduped = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def normalize_title_for_alias_check(name):
    """将标题标准化后用于alias规则判断（处理-hub后缀）"""
    if name.endswith('-hub'):
        return name[:-4]
    return name


def is_multiword_english_title(name):
    """判断是否为英文多词标题"""
    normalized = normalize_title_for_alias_check(name)
    if ' ' not in normalized:
        return False
    return bool(ENGLISH_MULTIWORD_TITLE_PATTERN.fullmatch(normalized))


def is_abbreviation_token(token):
    """判断首词是否为应保留的技术缩写"""
    if any(char.isdigit() for char in token):
        return True

    if re.fullmatch(r'[A-Z]{2,10}', token):
        return True

    if len(token) <= 10 and re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", token):
        uppercase_count = sum(char.isupper() for char in token)
        if uppercase_count >= 2:
            return True

    return False


def sanitize_aliases(aliases, filename):
    """清理错误首词alias：英文多词标题下仅移除非缩写首词"""
    name = os.path.splitext(filename)[0]
    if not is_multiword_english_title(name):
        return aliases

    first_token = normalize_title_for_alias_check(name).split()[0]
    if is_abbreviation_token(first_token):
        return aliases

    return [alias for alias in aliases if alias != first_token]


def generate_aliases(filename, folder_type):
    """根据文件名和文件夹类型生成aliases"""
    # 移除文件扩展名
    name = os.path.splitext(filename)[0]

    aliases = []

    # 添加中文别名（文件名本身）
    aliases.append(name)

    # 尝试从文件名中提取英文别名（如"资本资产定价模型" → "CAPM"）
    # 常见的经济金融术语英文缩写映射
    term_mapping = {
        "资本资产定价模型": "CAPM",
        "最小二乘法": "OLS",
        "风险价值": "VaR",
        "条件风险价值": "ES",
        "预期短缺": "ES",
        "压力测试": "Stress Testing",
        "历史模拟法": "Historical Simulation",
        "方差-协方差法": "Variance-Covariance Method",
        "蒙特卡罗模拟方法": "Monte Carlo Simulation",
        "边际VaR": "Marginal VaR",
        "递增VaR": "Incremental VaR",
        "成分VaR": "Component VaR",
        "一致性风险度量": "Coherent Risk Measure",
        "光谱风险度量": "Spectral Risk Measure",
        "Kupiec检验": "Kupiec Test",
        "聚束效应检验": "Cluster Effect Test",
        "VaR标准误": "VaR Standard Error",
        "压力VaR": "Stressed VaR",
        "绝对VaR": "Absolute VaR",
        "相对VaR": "Relative VaR"
    }

    if name in term_mapping:
        aliases.append(term_mapping[name])
    else:
        # 对纯英文多词标题，不再自动添加首词（避免Bond Valuation Model -> Bond）
        if not is_multiword_english_title(name):
            # 尝试提取文件名中的英文部分（如"CAPM模型" → "CAPM"）
            english_match = re.search(r'[A-Za-z]+', name)
            if english_match:
                aliases.append(english_match.group())

    return dedupe_preserve_order(aliases)


def generate_tags(folder_type, filename):
    """根据文件夹类型和文件名生成tags"""
    tags = []

    # 基础标签
    if folder_type == '00_hub':
        tags.append('hub')
    elif folder_type == 'framework':
        tags.append('framework')
    elif folder_type == 'concept':
        tags.append('concept')
    elif folder_type == 'system':
        tags.append('system')
    elif folder_type == 'procedure':
        tags.append('procedure')
    elif folder_type == 'proof':
        tags.append('proof')
    elif folder_type == 'writing':
        tags.append('writing')
    elif folder_type == 'undefined':
        tags.append('undefined')

    # 学科领域标签
    name = os.path.splitext(filename)[0]

    # 经济金融类标签
    finance_keywords = ['金融', '投资', '风险', '资产', '定价', '资本', '市场', '银行', '证券', '保险']
    for keyword in finance_keywords:
        if keyword in name:
            tags.append('金融')
            break

    economics_keywords = ['经济', '增长', '发展', '模型', '理论', '宏观', '微观']
    for keyword in economics_keywords:
        if keyword in name:
            tags.append('经济')
            break

    math_keywords = ['数学', '统计', '概率', '线性', '代数', '微积分', '矩阵']
    for keyword in math_keywords:
        if keyword in name:
            tags.append('数学')
            break

    cs_keywords = ['计算机', '编程', '算法', '数据', '结构']
    for keyword in cs_keywords:
        if keyword in name:
            tags.append('计算机')
            break

    return dedupe_preserve_order(tags)


def process_file(file_path, folder_type):
    """处理单个文件"""
    filename = os.path.basename(file_path)
    print(f"Processing: {file_path}")

    content = read_file(file_path)
    frontmatter, body = extract_frontmatter(content)

    # 确保frontmatter存在
    if frontmatter is None:
        frontmatter = {}

    # 处理aliases
    if 'aliases' not in frontmatter or not frontmatter['aliases']:
        frontmatter['aliases'] = sanitize_aliases(
            generate_aliases(filename, folder_type),
            filename
        )
    else:
        # 确保aliases是列表
        if isinstance(frontmatter['aliases'], str):
            frontmatter['aliases'] = [frontmatter['aliases']]
        frontmatter['aliases'] = dedupe_preserve_order(frontmatter['aliases'])
        frontmatter['aliases'] = sanitize_aliases(frontmatter['aliases'], filename)
        # 补充缺失的别名
        suggested_aliases = generate_aliases(filename, folder_type)
        for alias in suggested_aliases:
            if alias not in frontmatter['aliases']:
                frontmatter['aliases'].append(alias)
        frontmatter['aliases'] = dedupe_preserve_order(frontmatter['aliases'])

    # 处理tags
    if 'tags' not in frontmatter or not frontmatter['tags']:
        frontmatter['tags'] = generate_tags(folder_type, filename)
    else:
        # 确保tags是列表
        if isinstance(frontmatter['tags'], str):
            frontmatter['tags'] = [frontmatter['tags']]
        frontmatter['tags'] = dedupe_preserve_order(frontmatter['tags'])
        # 补充缺失的标签
        suggested_tags = generate_tags(folder_type, filename)
        for tag in suggested_tags:
            if tag not in frontmatter['tags']:
                frontmatter['tags'].append(tag)
        frontmatter['tags'] = dedupe_preserve_order(frontmatter['tags'])

    # 生成新的内容
    frontmatter_str = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False
    )
    new_content = f"---\n{frontmatter_str}---\n{body}"

    # 写入文件
    write_file(file_path, new_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化00_factor文件夹下Markdown文件的frontmatter")
    parser.add_argument("root_dir", help="00_factor文件夹路径")
    args = parser.parse_args()

    # 遍历所有子文件夹
    folder_types = ['00_hub', 'framework', 'concept', 'system', 'procedure', 'proof', 'writing', 'undefined']

    for folder_type in folder_types:
        folder_path = os.path.join(args.root_dir, folder_type)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder_path}")
            for filename in os.listdir(folder_path):
                if filename.endswith('.md'):
                    file_path = os.path.join(folder_path, filename)
                    process_file(file_path, folder_type)

    print("\n✅ Frontmatter优化完成！")


if __name__ == "__main__":
    main()
