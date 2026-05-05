"""
将 INT8 量化 scale 目录下的所有 .p (pickle) 文件读取并整合到一个 JSON 文件中。

文件命名格式: {layer_name}_{scale_type}_{layer_idx}.p
例如: q_proj_a_scale_0.p

Usage:
    python tools/pickle_to_json.py [--scale-dir PATH] [--output PATH]
"""

import os
import re
import pickle
import json
import argparse
from collections import defaultdict


def parse_filename(filename):
    """
    解析文件名，提取 layer_name, scale_type, layer_idx。
    文件名格式: {layer_name}_{scale_type}_{layer_idx}.p
    例如: q_proj_a_scale_0.p -> ('q_proj', 'a_scale', 0)
         qk_matmul_A_scale_3.p -> ('qk_matmul', 'A_scale', 3)
    """
    name = filename[:-2]  # 去掉 .p

    # scale_type 枚举（优先匹配长的）
    scale_types = ['w_scale', 'a_scale', 'o_scale', 'A_scale', 'B_scale', 'O_scale']

    for st in scale_types:
        pattern = rf'^(.+)_({re.escape(st)})_(\d+)$'
        m = re.match(pattern, name)
        if m:
            return m.group(1), m.group(2), int(m.group(3))

    return None


def load_all_scales(scale_dir):
    """
    读取目录下所有 .p 文件，整合为嵌套字典：
    {
        "q_proj": {
            0: {"a_scale": 0.085, "w_scale": 0.002, "o_scale": 0.100},
            1: {...},
            ...
        },
        ...
    }
    """
    data = defaultdict(lambda: defaultdict(dict))
    errors = []
    loaded = 0

    files = sorted([f for f in os.listdir(scale_dir) if f.endswith('.p')])
    print(f"找到 {len(files)} 个 .p 文件")

    for filename in files:
        parsed = parse_filename(filename)
        if parsed is None:
            errors.append(f"无法解析文件名: {filename}")
            continue

        layer_name, scale_type, layer_idx = parsed
        filepath = os.path.join(scale_dir, filename)

        try:
            with open(filepath, 'rb') as f:
                value = pickle.load(f)
            data[layer_name][layer_idx][scale_type] = float(value)
            loaded += 1
        except Exception as e:
            errors.append(f"读取失败 {filename}: {e}")

    print(f"成功读取: {loaded} 个文件")
    if errors:
        print(f"失败: {len(errors)} 个文件")
        for e in errors:
            print(f"  - {e}")

    return data


def build_output(data):
    """
    将嵌套字典转换为可序列化的有序结构。
    同时生成每个 layer 类型的统计摘要。
    """
    output = {}

    layer_order = ['q_proj', 'k_proj', 'v_proj', 'out_proj',
                   'fc1', 'fc2', 'qk_matmul', 'pv_matmul', 'lm_head']

    # 先按预定义顺序，再处理其他
    all_layers = layer_order + [l for l in sorted(data.keys()) if l not in layer_order]

    for layer_name in all_layers:
        if layer_name not in data:
            continue

        layer_data = data[layer_name]
        layers_by_idx = {}

        for idx in sorted(layer_data.keys()):
            layers_by_idx[str(idx)] = layer_data[idx]

        # 统计摘要
        summary = {}
        for scale_type in ['w_scale', 'a_scale', 'o_scale', 'A_scale', 'B_scale', 'O_scale']:
            values = [
                layer_data[idx][scale_type]
                for idx in layer_data
                if scale_type in layer_data[idx]
            ]
            if values:
                summary[scale_type] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }

        output[layer_name] = {
            'summary': summary,
            'layers': layers_by_idx
        }

    return output


def main():
    parser = argparse.ArgumentParser(description='将 .p 格式的量化 scale 转换为 JSON')
    parser.add_argument(
        '--scale-dir',
        type=str,
        default='/home/zyzhao/lfw_opt/opt-qt/quant_scale/Int8',
        help='存放 .p 文件的目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出 JSON 文件路径（默认保存在 scale-dir 下的 scales.json）'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.scale_dir):
        print(f"错误: 目录不存在: {args.scale_dir}")
        return

    output_path = args.output or os.path.join(args.scale_dir, 'scales.json')

    print(f"读取目录: {args.scale_dir}")
    data = load_all_scales(args.scale_dir)

    print("整合数据...")
    output = build_output(data)

    total_layers = sum(len(v['layers']) for v in output.values())
    print(f"共 {len(output)} 种层类型，{total_layers} 个层实例")

    # 写入 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n已保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")

    # 打印预览
    print("\n=== 预览（前两种层类型）===")
    for i, (layer_name, layer_info) in enumerate(output.items()):
        if i >= 2:
            break
        print(f"\n[{layer_name}]")
        print("  Summary:")
        for st, stats in layer_info['summary'].items():
            print(f"    {st}: min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
        first_idx = next(iter(layer_info['layers']))
        print(f"  Layer {first_idx}: {layer_info['layers'][first_idx]}")


if __name__ == '__main__':
    main()

