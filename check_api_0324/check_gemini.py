#!/usr/bin/env python3
"""
Gemini Video Failure Annotator
===============================
使用 Gemini 视频理解 API + 结构化输出标注 RoboCasa 失败视频。

视频为三联画：单帧内从左到右子画面顺序为 hand → right → left；failure_percent 表示根因时刻 a（非后果时刻 b）。

特点:
  - 支持本地视频上传（Gemini File API）
  - 可调节视频分析帧率 (FPS)
  - 结构化输出（Pydantic + response_schema）
  - 输出结果保存为 JSON
  - 支持断点续跑

用法:
  # 基本用法
  python check_gemini.py \
      --video_dir /path/to/videos \
      --output annotations_gemini.json

  # 自定义 FPS 和模型
  python check_gemini.py \
      --video_dir /path/to/videos \
      --fps 5 \
      --model gemini-2.0-flash

环境变量:
  GEMINI_API_KEY  — Gemini API Key (from ai.google.dev)

依赖:
  pip install google-genai pydantic
"""

from pydantic import BaseModel, Field
from typing import Literal
import argparse
import json
import os
import sys
import time
import subprocess
import re
from pathlib import Path
from google import genai
from google.genai import types


# —— 环境视角（与导出 MP4 一致：单帧内从左到右子画面顺序）———————————————————
# 若实际为上下拼接，请同步修改下方 PROMPT 中 "Camera layout" 段落。
VIEW_NAMES_ORDERED = ("hand", "right", "left")

# ── 数据模型 ─────────────────────────────────────────────────────────────────
FailureCause = Literal[
    "WRONG_TARGET", "BAD_APPROACH", "CONTACT_FAIL", "WRONG_MANIPULATION",
    "LOST_GRASP", "PLACE_FAIL", "COLLISION", "REACH_LIMIT", "FREEZE_OR_LOOP", "OTHER"
]

_EXPLANATION_FIELD_DESC = (
    "Detailed failure analysis in English (multiple sentences allowed). MUST explicitly name views "
    f"{', '.join(repr(v) for v in VIEW_NAMES_ORDERED)} when citing evidence. MUST cover: (1) which "
    "object(s) or part(s) of the scene are involved (use color/position if needed to disambiguate); "
    "(2) end-effector / gripper pose relative to the target or scene (offset, orientation); "
    "(3) motion or force direction (approach, retreat, along which axis or side); (4) spatial "
    "references (table edge, container rim, hinge side, etc.); (5) which sub-panel(s) support the "
    "conclusion. Avoid vague one-word labels like 'misalign' or 'wrong object' without this detail."
)

_RECOVERY_FIELD_DESC = (
    "Step-by-step recovery in English. Number or separate clear steps. Each step should be actionable: "
    "what to move where, relative to what reference, and which view(s) (hand / right / left) to use to "
    "verify alignment. Mirror the specificity of the failure analysis (objects, directions, poses). "
    "Do not answer with only generic phrases like 're-align', 'reset', or 'should recover'."
)


class VideoAnnotation(BaseModel):
    """结构化输出：Gemini 对失败视频的标注"""
    failure_percent: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Timeline percent (0-100) at ROOT-CAUSE moment a: the earliest time when a committed mistake "
            "or bad geometry makes task success impossible or nearly impossible (e.g., closing on a "
            "misaligned grasp, wrong object instance, wrong manipulation direction). This MUST be a, NOT "
            "moment b when a later visible outcome occurs (drop, collision stall, timeout). If an early "
            "alignment/grasp/target error is already clear at a, do NOT set this to the percent at a later b."
        ),
    )
    failure_cause: FailureCause = Field(
        ...,
        description="Exactly one ROOT CAUSE category matching the mistake at time a.",
    )
    explanation: str = Field(..., description=_EXPLANATION_FIELD_DESC)
    how_to_recover: str = Field(..., description=_RECOVERY_FIELD_DESC)


# ── 提示词 ────────────────────────────────────────────────────────────────
PROMPT = """You are observing a robot manipulation task in simulation.

Task: '{instruction}'

This episode FAILED — the robot did NOT complete the task.
IMPORTANT: The robot starts in a reasonable initial pose. The failure happens DURING the episode, not at the start.

Camera layout (each frame is a horizontal triptych; sub-panels left-to-right):
- hand: gripper / eye-in-hand camera
- right: right agent (third-person) view
- left: left agent (third-person) view
When you describe what you see or what to do, explicitly reference these view names (hand, right, left).

Two timeline concepts (critical):
- Moment a (ROOT CAUSE): the earliest time when the policy commits a mistake or geometry that already makes success impossible or nearly impossible — e.g., gripper clearly misaligned but still closes, wrong object instance engaged, wrong approach direction locked in. At a there may be NO dramatic failure yet (no drop, no big collision).
- Moment b (OUTCOME): a later time when the consequence becomes obvious — object falls, repeated slipping, collision stop, freeze, etc. Often several seconds after a.

Your job:
1. Set failure_percent to the percentage (0-100) of the video duration at moment **a** (root cause), NOT at **b**. If both exist, **a** is always earlier or equal; never choose b when a is already identifiable. Do not anchor on the first "dramatic" failure if an earlier fatal mistake is visible.

2. Classify the ROOT CAUSE into exactly ONE category (the mistake at time a):
   - WRONG_TARGET: wrong object/instance manipulated
   - BAD_APPROACH: correct target, but approach/pose wrong
   - CONTACT_FAIL: contact happens but no effective grasp formed
   - WRONG_MANIPULATION: wrong direction/force for manipulation
   - LOST_GRASP: stable hold lost before task completion
   - PLACE_FAIL: transport OK, failure at final placement
   - COLLISION: unintended collision blocks motion
   - REACH_LIMIT: repeated failure to reach due to limits
   - FREEZE_OR_LOOP: freeze, hang, or repetitive micro-motions
   - OTHER: none of the above

3. Fill `explanation`: detailed failure analysis. Use view names hand/right/left. Cover objects, gripper pose vs target, motion directions, spatial references, and which views support your judgment.

4. Fill `how_to_recover`: concrete numbered or stepwise recovery from the situation at time a, with directions, references, and which view to check — not generic recovery slogans.
"""

VALID_CAUSES = {
    "WRONG_TARGET", "BAD_APPROACH", "CONTACT_FAIL", "WRONG_MANIPULATION",
    "LOST_GRASP", "PLACE_FAIL", "COLLISION", "REACH_LIMIT", "FREEZE_OR_LOOP", "OTHER",
}


# ── 工具函数 ────────────────────────────────────────────────────────────────

def load_episodes(jsonl_path):
    """从 metrics_log.jsonl 读取 episode 元数据，按 episode_id 排序"""
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line.strip())
            # 支持两种格式：
            # 1. 有 "success" 字段：读取 success=false 的 episode
            # 2. 无 "success" 字段：读取所有 episode（视为失败视频）
            if "success" in ep:
                if not ep["success"]:
                    episodes.append(ep)
            else:
                # 没有 success 字段，假设这个 episode 就是失败的
                episodes.append(ep)
    return sorted(episodes, key=lambda e: e.get("episode_id", 0))


def match_videos_to_episodes(task_dir, episodes):
    """将 episode_id 映射到对应的 mp4 文件路径"""
    mapping = {}
    for ep in episodes:
        ep_id = ep["episode_id"]
        video_path = Path(task_dir) / f"episode{ep_id}.mp4"
        if video_path.exists():
            mapping[ep_id] = video_path
    return mapping


def get_video_duration(video_path):
    """使用 ffprobe 获取视频时长（秒），失败时返回 None"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def read_video_bytes(video_path):
    """读取视频文件为字节"""
    with open(video_path, "rb") as f:
        return f.read()


def upload_video_to_gemini(client, video_path):
    """
    上传视频到 Gemini File API 并等待处理完成。
    返回 genai.types.File 对象。
    """
    print(f"    Uploading {video_path.name} to Gemini File API...")
    video_file = client.files.upload(file=str(video_path))
    print(f"    Upload complete. File ID: {video_file.name}, State: {video_file.state.name}")

    # 轮询直到处理完成
    while video_file.state.name == "PROCESSING":
        print(f"    Processing... waiting 3 seconds")
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise RuntimeError(f"Video upload failed: {video_file.name}")

    print(f"    File ready. State: {video_file.state.name}")
    return video_file


def build_video_part(client, video_path, fps=1):
    """
    构建视频内容部分。
    < 20MB: 内联字节 + VideoMetadata(fps)
    >= 20MB: File API 上传 + 返回文件对象
    """
    file_size = os.path.getsize(video_path)

    if file_size < 20 * 1024 * 1024:  # < 20MB
        print(f"    Using inline video (size: {file_size/1024/1024:.1f}MB)")
        video_bytes = read_video_bytes(video_path)
        return types.Part(
            inline_data=types.Blob(
                data=video_bytes,
                mime_type="video/mp4"
            ),
            video_metadata=types.VideoMetadata(fps=fps)
        )
    else:  # >= 20MB
        print(f"    Using File API upload (size: {file_size/1024/1024:.1f}MB)")
        video_file = upload_video_to_gemini(client, video_path)
        # 返回文件对象，Gemini 会自动使用 VideoMetadata
        # （注：File API 目前默认 1 FPS，暂不支持 fps 调节）
        return video_file


def annotate_episode(client, model, instruction, video_path, episode_length, fps=1):
    """
    调用 Gemini API 对单个失败 episode 进行标注。
    返回 (annotation, step, raw_response_json) 或 (None, None, error_str)
    """
    try:
        # 构建视频部分
        video_part = build_video_part(client, video_path, fps)

        # 构建完整内容
        prompt_text = PROMPT.format(instruction=instruction)

        # 调用 Gemini API，使用结构化输出
        response = client.models.generate_content(
            model=model,
            contents=[video_part, prompt_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoAnnotation,
                temperature=0.35,
            ),
        )

        # 获取结构化结果
        annotation: VideoAnnotation = response.parsed

        # 计算故障步数
        step = round(annotation.failure_percent / 100.0 * episode_length)

        # 获取原始 JSON 响应
        raw_json = response.text if hasattr(response, 'text') else json.dumps(annotation.model_dump())

        return annotation, step, raw_json

    except Exception as e:
        return None, None, str(e)


def cleanup_video_file(client, file_id):
    """删除 Gemini File API 中的视频文件以节省配额"""
    try:
        client.files.delete(name=file_id)
        print(f"    Deleted file: {file_id}")
    except Exception as e:
        print(f"    Warning: Failed to delete file {file_id}: {e}", file=sys.stderr)


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="使用 Gemini 视频理解 + 结构化输出标注 RoboCasa 失败视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="环境变量 GEMINI_API_KEY 必须设置为 Gemini API Key。")

    parser.add_argument("--video_dir", type=str, required=True,
                        help="视频目录，结构: video_dir/<TaskName>/{*.mp4, metrics_log.jsonl}")
    parser.add_argument("--output", type=str, default="annotations_gemini.json",
                        help="输出 JSON 文件路径 (默认: annotations_gemini.json)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="只标注指定任务 (默认: 标注全部)")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Gemini 模型名 (默认: gemini-3-flash-preview)")
    parser.add_argument("--fps", type=float, default=1,
                        help="视频分析帧率 FPS (默认: 1, 范围: 0.5-5)")
    parser.add_argument("--max_fail_per_task", type=int, default=10,
                        help="每个任务最多标注几个失败 episode (默认: 10)")
    parser.add_argument("--api_delay", type=float, default=1.0,
                        help="每次 API 调用后的等待秒数 (默认: 1.0)")
    args = parser.parse_args()

    # API Key 验证
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: 请设置环境变量 GEMINI_API_KEY", file=sys.stderr)
        print("  export GEMINI_API_KEY='your-gemini-api-key'", file=sys.stderr)
        sys.exit(1)

    # Validate FPS
    if not (0.5 <= args.fps <= 5):
        print(f"Warning: FPS {args.fps} outside recommended range [0.5, 5]")

    # 初始化 Gemini 客户端
    client = genai.Client(api_key=api_key)
    video_base = Path(args.video_dir)

    # 输出文件路径
    out_file = video_base / args.output

    # 断点续跑
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[LOAD] Loaded {sum(len(v) for v in results.values())} existing annotations, resuming.")
    else:
        results = {}

    # 遍历任务目录
    task_dirs = sorted(video_base.iterdir())
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            print(f"[SKIP] {task_dir}: not a directory")
            continue

        task = task_dir.name

        if args.tasks and task not in args.tasks:
            print(f"[SKIP] {task}: not in --tasks")
            continue

        jsonl = task_dir / "metrics_log.jsonl"
        if not jsonl.exists():
            print(f"[SKIP] {task_dir}: no metrics_log.jsonl")
            continue

        if task in results and len(results[task]) >= args.max_fail_per_task:
            print(f"[DONE] {task}: already has {len(results[task])} annotations")
            continue

        # 加载 episode 元数据
        fail_eps = load_episodes(jsonl)
        ep_to_video = match_videos_to_episodes(task_dir, fail_eps)

        print(f"\n{'='*70}")
        print(f"Task: {task}  ({len(fail_eps)} failure episodes)")
        print(f"Model: {args.model}, FPS: {args.fps}")
        print(f"{'='*70}")

        task_results = results.get(task, [])

        for ep in fail_eps:
            ep_id = ep["episode_id"]

            # 检查是否已标注
            if any(r["episode_id"] == ep_id for r in task_results):
                print(f"  ep{ep_id:3d}: already annotated, skip")
                continue

            # 检查视频文件是否存在
            video_path = ep_to_video.get(ep_id)
            if not video_path or not video_path.exists():
                ep_length = ep.get("length") or ep.get("actual_length", 1)
                print(f"  ep{ep_id:3d}: video not found at {video_path}")
                task_results.append({
                    "episode_id": ep_id,
                    "length": ep_length,
                    "gemini_step": None,
                    "gemini_reply": "video not found",
                })
                continue

            # 获取视频时长
            duration = get_video_duration(video_path)
            instruction = ep.get("instruction", "N/A")

            # 兼容两种字段名：length 或 actual_length
            ep_length = ep.get("length") or ep.get("actual_length", 1)
            duration_str = f"{duration:.1f}s" if duration is not None else "?s"
            print(f"  ep{ep_id:3d} (len={ep_length:3d}, {duration_str})")
            print(f"         instruction: {instruction[:60]}...")

            # 调用 Gemini 标注
            annotation, gemini_step, reply = annotate_episode(
                client, args.model, instruction, video_path, ep_length, fps=args.fps
            )

            # 计算故障时间
            if annotation:
                gemini_pct = annotation.failure_percent
                failure_time = round(gemini_pct / 100.0 * duration, 2) if duration else None
                pct_str = f"{gemini_pct}%"
                time_str = f"{failure_time:.2f}s" if failure_time is not None else "N/A"
                cause_str = annotation.failure_cause
                expl_str = annotation.explanation[:60]

                print(f"         step={gemini_step:3d} ({pct_str:4s}, {time_str:8s}) cause={cause_str:20s} | {expl_str}...")

                task_results.append({
                    "episode_id": ep_id,
                    "instruction": instruction,
                    "length": ep_length,
                    "video_duration_sec": duration,
                    "gemini_pct": gemini_pct,
                    "gemini_step": gemini_step,
                    "failure_time_sec": failure_time,
                    "failure_cause": annotation.failure_cause,
                    "explanation": annotation.explanation,
                    "how_to_recover": annotation.how_to_recover,
                    "gemini_reply": reply,
                })
            else:
                print(f"         ERROR: {reply[:100]}")
                task_results.append({
                    "episode_id": ep_id,
                    "length": ep_length,
                    "gemini_step": None,
                    "gemini_reply": reply,
                })

            time.sleep(args.api_delay)

        # 保存进度
        results[task] = task_results
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✓ Done. Annotations saved to: {out_file}")
    total = sum(len(v) for v in results.values())
    print(f"  Total annotated: {total}")


if __name__ == "__main__":
    main()
