#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计评估结果的工具脚本
分别统计简单任务和复杂任务的名称、成功率等指标，并保存为CSV文件
"""

import os
import json
import csv
from pathlib import Path


def find_summary_files(folder_path):
    """
    在指定文件夹下查找所有的 *_summary.json 文件
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        summary文件路径列表
    """
    summary_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"警告: 文件夹 {folder_path} 不存在")
        return summary_files
    
    for task_folder in folder.iterdir():
        if task_folder.is_dir():
            # 查找 *_summary.json 文件
            for f in task_folder.glob("*_summary.json"):
                summary_files.append(f)
    
    return summary_files


def read_summary(summary_file):
    """
    读取summary.json文件内容
    
    Args:
        summary_file: summary文件路径
        
    Returns:
        包含task, avg_success_rate等字段的字典
    """
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                'task': data.get('task', ''),
                'avg_success_rate': data.get('avg_success_rate', 0),
                'num_episodes': data.get('num_episodes', 0),
                'total_trials': data.get('total_trials', 0),
                'avg_gt_length': data.get('avg_gt_length', 0),
                'avg_actual_length': data.get('avg_actual_length', 0),
                'avg_action_mse': data.get('avg_action_mse', 0),
                'avg_state_mse': data.get('avg_state_mse', 0),
            }
    except Exception as e:
        print(f"读取文件 {summary_file} 失败: {e}")
        return None


def collect_task_stats(folder_path, task_type):
    """
    收集指定文件夹下所有任务的统计信息
    
    Args:
        folder_path: 文件夹路径
        task_type: 任务类型 ('simple' 或 'complex')
        
    Returns:
        任务统计信息列表
    """
    summary_files = find_summary_files(folder_path)
    stats = []
    
    for summary_file in summary_files:
        data = read_summary(summary_file)
        if data:
            data['type'] = task_type
            stats.append(data)
    
    # 按任务名称排序
    stats.sort(key=lambda x: x['task'])
    return stats


def save_to_csv(stats, output_file, include_type=False):
    """
    将统计信息保存为CSV文件
    
    Args:
        stats: 统计信息列表
        output_file: 输出文件路径
        include_type: 是否包含type列
    """
    if not stats:
        print(f"警告: 没有数据可保存到 {output_file}")
        return
    
    # 定义列名
    if include_type:
        fieldnames = ['type', 'task', 'avg_success_rate',
                      'num_episodes', 'total_trials', 'avg_gt_length', 
                      'avg_actual_length', 'avg_action_mse', 'avg_state_mse']
    else:
        fieldnames = ['task', 'avg_success_rate',
                      'num_episodes', 'total_trials', 'avg_gt_length', 
                      'avg_actual_length', 'avg_action_mse', 'avg_state_mse']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(stats)
    
    print(f"已保存到: {output_file}")


def calculate_average(stats, field):
    """计算指定字段的平均值"""
    if not stats:
        return 0
    values = [s[field] for s in stats if s[field] is not None]
    return sum(values) / len(values) if values else 0


def main():
    # 获取脚本所在目录作为基础目录
    base_dir = Path(__file__).parent
    
    simple_dir = base_dir / "simple"
    complex_dir = base_dir / "complex"
    
    # 收集统计信息
    print("=" * 60)
    print("收集简单任务统计信息...")
    simple_stats = collect_task_stats(simple_dir, 'simple')
    print(f"找到 {len(simple_stats)} 个简单任务的summary文件")
    
    print("\n收集复杂任务统计信息...")
    complex_stats = collect_task_stats(complex_dir, 'complex')
    print(f"找到 {len(complex_stats)} 个复杂任务的summary文件")
    
    # 保存为单独的CSV文件
    print("\n" + "=" * 60)
    print("保存CSV文件...")
    
    # 简单任务CSV
    simple_csv = base_dir / "simple_tasks_summary.csv"
    save_to_csv(simple_stats, simple_csv, include_type=False)
    
    # 复杂任务CSV
    complex_csv = base_dir / "complex_tasks_summary.csv"
    save_to_csv(complex_stats, complex_csv, include_type=False)
    
    # 合并的CSV文件（包含所有任务）
    all_stats = simple_stats + complex_stats
    all_csv = base_dir / "all_tasks_summary.csv"
    save_to_csv(all_stats, all_csv, include_type=True)
    
    # 打印统计摘要
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)
    
    if simple_stats:
        avg_success = calculate_average(simple_stats, 'avg_success_rate')
        print(f"\n简单任务 ({len(simple_stats)} 个任务):")
        print(f"  平均成功率: {avg_success:.2%}")
        print(f"  任务列表:")
        for s in simple_stats:
            print(f"    - {s['task']}: {s['avg_success_rate']:.2%}")
    
    if complex_stats:
        avg_success = calculate_average(complex_stats, 'avg_success_rate')
        print(f"\n复杂任务 ({len(complex_stats)} 个任务):")
        print(f"  平均成功率: {avg_success:.2%}")
        print(f"  任务列表:")
        for s in complex_stats:
            print(f"    - {s['task']}: {s['avg_success_rate']:.2%}")
    
    if all_stats:
        avg_success = calculate_average(all_stats, 'avg_success_rate')
        print(f"\n总计 ({len(all_stats)} 个任务):")
        print(f"  总平均成功率: {avg_success:.2%}")


if __name__ == "__main__":
    main()
