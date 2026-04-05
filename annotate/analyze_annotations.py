import json
from pathlib import Path
from collections import defaultdict

# 任务列表和标注文件列表
TASKS = ["OpenSingleDoor", "PnPCabToCounter", "PnPMicrowaveToCounter", "PnPSinkToCounter", "TurnOnStove"]
ANNOTATION_FILES = [
    "gemini3-flash-fps2_low_trimmed_notype.json",
    "gemini3-flash-fps2_medium_trimmed_notype.json",
    "gemini3-flash-fps5_low_trimmed_notype.json",
    "gemini3-flash-fps5_medium_trimmed_notype.json",
    "gemini3-flash-fps10_low_trimmed_notype.json",
    "gemini3-flash-fps10_medium_trimmed_notype.json",
]

BASE_PATH = Path("debug/failed_videos_0321")
TEST_PATH = BASE_PATH / "test_0331"

def parse_annotation_filename(filename):
    """从文件名提取fps和thinking level"""
    # 格式: gemini3-flash-fps{N}_{level}_trimmed_notype.json
    parts = filename.replace(".json", "").replace("gemini3-flash-", "").split("_")
    fps = parts[0].replace("fps", "")
    level = parts[1]
    return fps, level

def check_annotation_correctness(failure_time_sec, fail_time_range):
    """
    检查标注是否正确
    failure_time_sec: 模型标注的失败时间（秒）
    fail_time_range: [start, end] 真实失败时间范围

    返回: (is_correct, details)
    """
    start, end = fail_time_range
    is_correct = start <= failure_time_sec <= end
    return is_correct, {
        "annotated_time": failure_time_sec,
        "actual_range": fail_time_range,
        "in_range": is_correct
    }

def load_metrics_log(task_name):
    """加载metrics_log.jsonl"""
    metrics_file = BASE_PATH / task_name / "metrics_log.jsonl"
    metrics = {}

    if not metrics_file.exists():
        return metrics

    with open(metrics_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                episode_id = data["episode_id"]
                metrics[episode_id] = data

    return metrics

def analyze_all():
    """分析所有标注文件"""
    results = defaultdict(lambda: defaultdict(dict))  # {annotation_file: {task: {episode: result}}}

    for annotation_file in ANNOTATION_FILES:
        fps, level = parse_annotation_filename(annotation_file)
        annotation_path = TEST_PATH / annotation_file

        if not annotation_path.exists():
            print(f"⚠️  文件不存在: {annotation_file}")
            continue

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        for task_name in TASKS:
            if task_name not in annotations:
                continue

            # 加载该任务的metrics_log
            metrics = load_metrics_log(task_name)

            for annotation in annotations[task_name]:
                episode_id = annotation["episode_id"]
                failure_time_sec = annotation.get("failure_time_sec")

                # 跳过没有failure_time_sec的项
                if failure_time_sec is None:
                    results[annotation_file][task_name][episode_id] = {
                        "status": "missing_annotation",
                        "failure_time_sec": failure_time_sec
                    }
                    continue

                if episode_id not in metrics:
                    results[annotation_file][task_name][episode_id] = {
                        "status": "missing_metrics",
                        "failure_time_sec": failure_time_sec
                    }
                    continue

                fail_time_range = metrics[episode_id]["fail_time"]
                is_correct, details = check_annotation_correctness(failure_time_sec, fail_time_range)

                results[annotation_file][task_name][episode_id] = {
                    "status": "correct" if is_correct else "incorrect",
                    "failure_time_sec": failure_time_sec,
                    "actual_range": fail_time_range,
                    "is_correct": is_correct
                }

    return results

def print_results(results):
    """打印分析结果"""
    print("\n" + "="*80)
    print("标注正确性分析报告")
    print("="*80)

    # 汇总统计
    total_correct = 0
    total_incorrect = 0
    total_missing = 0

    # 按标注文件统计
    print("\n【按标注文件统计】")
    print("-" * 80)

    file_stats = {}

    for annotation_file in sorted(results.keys()):
        fps, level = parse_annotation_filename(annotation_file)
        correct = 0
        incorrect = 0
        missing = 0

        task_results = results[annotation_file]

        for task_name in sorted(task_results.keys()):
            for episode_id, result in sorted(task_results[task_name].items()):
                if result["status"] == "correct":
                    correct += 1
                elif result["status"] == "incorrect":
                    incorrect += 1
                else:
                    missing += 1

        total = correct + incorrect
        accuracy = (correct / total * 100) if total > 0 else 0

        file_stats[annotation_file] = {
            "fps": fps,
            "level": level,
            "correct": correct,
            "incorrect": incorrect,
            "missing": missing,
            "accuracy": accuracy
        }

        total_correct += correct
        total_incorrect += incorrect
        total_missing += missing

        status_symbol = "✓" if accuracy >= 80 else "✗" if accuracy < 50 else "~"
        print(f"{status_symbol} {annotation_file}")
        print(f"   FPS: {fps}, Level: {level}")
        print(f"   正确: {correct}, 错误: {incorrect}, 缺失: {missing}")
        print(f"   正确率: {accuracy:.1f}%")
        print()

    # 总体正确率
    total = total_correct + total_incorrect
    overall_accuracy = (total_correct / total * 100) if total > 0 else 0

    print("-" * 80)
    print(f"【总体统计】")
    print(f"总共标注: {total} 条轨迹")
    print(f"正确: {total_correct}, 错误: {total_incorrect}, 缺失: {total_missing}")
    print(f"总体正确率: {overall_accuracy:.1f}%")
    print()

    # 按任务统计
    print("="*80)
    print("【按任务统计】")
    print("-" * 80)

    task_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})

    for annotation_file in results.keys():
        for task_name in results[annotation_file].keys():
            for episode_id, result in results[annotation_file][task_name].items():
                if result["status"] != "missing_metrics":
                    task_stats[task_name]["total"] += 1
                    if result["status"] == "correct":
                        task_stats[task_name]["correct"] += 1
                    else:
                        task_stats[task_name]["incorrect"] += 1

    for task_name in sorted(TASKS):
        if task_name not in task_stats:
            print(f"❌ {task_name}: 无数据")
            continue

        stat = task_stats[task_name]
        accuracy = (stat["correct"] / stat["total"] * 100) if stat["total"] > 0 else 0
        status = "✓" if accuracy >= 80 else "✗" if accuracy < 50 else "~"

        print(f"{status} {task_name}")
        print(f"   正确: {stat['correct']}/{stat['total']} ({accuracy:.1f}%)")

    print()

    # 详细结果
    print("="*80)
    print("【详细结果 - 按任务和标注文件】")
    print("-" * 80)

    for task_name in sorted(TASKS):
        print(f"\n### {task_name}")

        for annotation_file in sorted(results.keys()):
            if task_name not in results[annotation_file]:
                continue

            fps, level = parse_annotation_filename(annotation_file)
            episodes = results[annotation_file][task_name]

            correct_count = sum(1 for e in episodes.values() if e["status"] == "correct")
            total_count = sum(1 for e in episodes.values() if e["status"] != "missing_metrics")

            if total_count == 0:
                continue

            accuracy = correct_count / total_count * 100
            status = "✓" if accuracy == 100 else "✗" if accuracy == 0 else "~"

            print(f"  {status} {annotation_file} (FPS{fps}/{level}): {correct_count}/{total_count} ({accuracy:.0f}%)")

            # 显示错误的轨迹
            for episode_id in sorted(episodes.keys()):
                result = episodes[episode_id]
                if result["status"] == "incorrect":
                    print(f"     ✗ Episode {episode_id}: 标注={result['failure_time_sec']}s, 实际范围={result['actual_range']}")

    print("\n" + "="*80)

if __name__ == "__main__":
    results = analyze_all()
    print_results(results)
