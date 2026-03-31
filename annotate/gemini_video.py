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
from pathlib import Path
from google import genai
from google.genai import types


# ── 数据模型 ─────────────────────────────────────────────────────────────────
FailureType = Literal[
    "WRONG_TARGET", 
    "BAD_APPROACH", 
    "CONTACT_FAIL", 
    "WRONG_MANIPULATION",
    "WRONG_OBJECT",
    "LOST_GRASP", 
    "PLACE_FAIL", 
    "COLLISION", 
    "REACH_LIMIT", 
    "FREEZE_OR_LOOP", 
    "OTHER"
]

class VideoAnnotation(BaseModel):

    failure_position: int = Field(
        ...,
        ge=0,
        le=100,
    )
    failure_type: FailureType = Field(
        ...,
    )
    perception: str = Field(
        ...,
    )
    summary: str = Field(
        ...,
    )
    reflection: str = Field(
        ...,
    )
    plan: str = Field(
        ...,
    )
    next_subtask: str = Field(
        ...,
    )


# ── 提示词 ────────────────────────────────────────────────────────────────
PROMPT = """You are an expert of robot manipulation task analysis.
==========
You are observing a FAILED robot manipulation video in simulation. And your job is to find the FIRST failure in the video and analyze it.

==========
INPUT (will be supplied to you):

- Task: '{instruction}'
- Video: The video of the failed task. 

The camera layout (each frame is a horizontal triptych, from left to right) in the video is: 
- hand (eye-in-hand camera view)
- right (right agent (third-person) view)
- left (left agent (third-person) view)

==========
STRICT OUTPUT FORMAT (required):

Return one valid JSON object with exactly these fields:
{{
  "failure_position": "...",
  "failure_type": "...",
  "perception": "...",
  "summary": "...",
  "reflection": "...",
  "plan": "...",
  "next_subtask": "..."
}}

==========
WHAT TO INCLUDE IN EACH FIELD (content requirements):

**failure_position**
An integer from 0 to 100 representing the timeline percent where the FIRST failure happens. 0% = first frame, 100% = last frame.

**failure_type**
Pick exactly ONE type from the labels below for the failure.

**perception**
Analysis of the environment scene. You should ONLY focus on the critical objects related to the task. Include their attributes and spatial layout when necessary, such as color, position, shape, etc.

**summary**
The subtasks that have been executed BEFORE the failure.

**reflection**
Analysis of what subtask the robot is trying to do and why the failure happens. 

**plan**
Subtasks for the remaining task. Begin with the subtask for recovery.

**next_subtask**
The immediate next subtask to be executed for recovery. 

==========
LABELS FOR FAILURE_TYPE (pick ONE):

- WRONG_TARGET: wrong object/instance manipulated
- BAD_APPROACH: correct target, but approach/pose wrong
- CONTACT_FAIL: contact happens but no effective grasp formed
- WRONG_MANIPULATION: wrong direction/force for manipulation
- WRONG_OBJECT: wrong object identity vs task (distinct instance/type confusion)
- LOST_GRASP: stable hold lost before task completion
- PLACE_FAIL: transport OK, failure at final placement
- COLLISION: unintended collision blocks motion
- REACH_LIMIT: repeated failure to reach due to limits
- FREEZE_OR_LOOP: freeze, hang, or repetitive micro-motions
- OTHER: none of the above, describe in your own words.

==========
RULES/SAFEGUARDS:

• Do not hallucinate: never assert the presence of something not visible in the video. Use words like "possibly", "uncertain", "not sure", etc. to express your uncertainty.
• Output a valid JSON with only the above fields — no extra text or markdown.
• Keep the perception, reflection, plan, and next_subtask fields in the JSON as specific, concise and plain words.

==========
EXAMPLE OUTPUT:

{{
    "failure_position": 25,
    "failure_type": "BAD_APPROACH",
    "perception": "Kitchen counter scene with a coffee machine on the counter. The upper cabinet door is open. The robot arm is at the front of the machine; the gripper is grasping the mug and is near the dispenser spout. But the mug is not upright on the tray yet and the gripper is about to lose contact with the mug.",
    "summary": "Move the arm to the cabinet; Grasp the mug in the cabinet; Move the mug to the drip tray of the coffee machine.",
    "reflection": "The robot is trying to 'Place the mug on the drip tray'. However, the gripper was tilted left and a little low and mug-tray collision was detected. The grip is slipping, leading to failed placement in the future.",
    "plan": "Grasps the mug and adjust its position for stable placement; Move the gripper to the start button; Press the start button.",
    "next_subtask": "Grasps the mug and adjust its position for stable placement."
}}
"""

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




def trim_video_to_first_half(video_path, trim_percent=0.5):
    """
    使用 ffmpeg 裁剪视频，只保留前面的部分（去掉后面的百分比）。
    使用临时文件存储中间结果，然后读入内存，最后删除临时文件。

    Args:
        video_path: 源视频路径
        trim_percent: 保留视频的百分比（默认 0.5 = 保留前50%，去掉后50%）

    Returns:
        (trimmed_video_bytes, original_duration, trimmed_duration)
    """
    import tempfile

    try:
        # 获取原始视频时长
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        original_duration = float(result.stdout.strip())

        # 计算裁剪后的时长
        trimmed_duration = original_duration * trim_percent

        print(f"    Trimming video: {original_duration:.1f}s → {trimmed_duration:.1f}s (keeping {trim_percent*100:.0f}%)")

        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            # 使用 ffmpeg 裁剪视频到临时文件
            result = subprocess.run(
                ["ffmpeg", "-i", str(video_path), "-t", str(trimmed_duration),
                 "-c:v", "libx264", "-crf", "23", "-c:a", "aac", "-y", temp_path],
                capture_output=True, timeout=120
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")

            # 读取临时文件到内存
            with open(temp_path, "rb") as f:
                trimmed_video_bytes = f.read()

            if not trimmed_video_bytes:
                raise RuntimeError("trimmed video file is empty")

            trimmed_size = len(trimmed_video_bytes)
            print(f"    Trimmed video in memory: {trimmed_size/1024/1024:.1f}MB")

            return trimmed_video_bytes, original_duration, trimmed_duration

        finally:
            # 删除临时文件
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    except Exception as e:
        print(f"    Warning: Failed to trim video: {e}")
        # 返回原始视频字节
        with open(video_path, "rb") as f:
            original_bytes = f.read()
        return original_bytes, None, None


def build_video_part(video_path, fps=1, trim_percent=0.5):
    """
    构建视频内容部分（内联方式）。

    - 自动去掉视频后面的百分比（默认50%）
    - 直接裁剪到内存，不保存中间文件
    - 视频大小 < 20MB，使用内联方式

    Args:
        video_path: 视频文件路径
        fps: 视频分析帧率
        trim_percent: 保留视频的百分比（默认0.5 = 保留前50%）

    Returns:
        (video_part, original_duration, trimmed_duration, actual_trim_percent)
    """
    # 裁剪视频到内存
    video_bytes, original_duration, trimmed_duration = trim_video_to_first_half(
        video_path, trim_percent=trim_percent
    )

    # 如果裁剪失败，actual_trim_percent = 1.0
    if original_duration is None:
        print(f"    Using original video (trim failed)")
        actual_trim_percent = 1.0
    else:
        actual_trim_percent = trim_percent

    file_size = len(video_bytes)
    print(f"    Using inline video (size: {file_size/1024/1024:.1f}MB)")

    video_part = types.Part(
        inline_data=types.Blob(
            data=video_bytes,
            mime_type="video/mp4"
        ),
        video_metadata=types.VideoMetadata(fps=fps)
    )

    return video_part, original_duration, trimmed_duration, actual_trim_percent


def annotate_episode(client, model, instruction, video_path, episode_length, fps=1, thinking_level="medium", trim_percent=0.5):
    """
    调用 Gemini API 对单个失败 episode 进行标注。

    注意：
    - 视频会被裁剪到前 trim_percent（默认50%）
    - failure_position 百分比是相对于「裁剪后」的视频
    - 需要转换回「原始」视频的步数和时间

    Args:
        client: Gemini 客户端
        model: 模型名称
        instruction: 任务指令
        video_path: 视频文件路径
        episode_length: 原始 episode 的总步数
        fps: 视频分析帧率
        thinking_level: 思考深度
        trim_percent: 保留视频的百分比

    Returns:
        (annotation, failure_step_original, failure_time_original, raw_response_json, trim_info)
        或 (None, None, None, error_str, None)

        trim_info = {
            'original_duration': 原始视频时长,
            'trimmed_duration': 裁剪后视频时长,
            'trim_percent': 裁剪百分比,
            'failure_position_trimmed': 模型输出的百分比（相对裁剪后）,
            'failure_position_original': 转换后的百分比（相对原始）
        }
    """
    try:
        # 构建视频部分（包括裁剪）
        video_part, original_duration, trimmed_duration, actual_trim_percent = build_video_part(
            video_path, fps, trim_percent=trim_percent
        )

        # 构建完整内容
        prompt_text = PROMPT.format(instruction=instruction)

        # 构建配置
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoAnnotation,
            temperature=0.35,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        )

        # 调用 Gemini API，使用结构化输出
        response = client.models.generate_content(
            model=model,
            contents=[video_part, prompt_text],
            config=config,
        )

        # 获取结构化结果
        annotation: VideoAnnotation = response.parsed

        # 重要：模型返回的 failure_position 是相对于「裁剪后」的视频百分比
        # 需要转换回「原始」视频的百分比
        failure_position_trimmed = annotation.failure_position  # 相对裁剪后视频 (0-100%)
        failure_position_original = failure_position_trimmed * actual_trim_percent  # 相对原始视频 (0-100%)

        # 计算故障步数（相对原始 episode）
        failure_step = round(failure_position_original / 100.0 * episode_length)

        # 获取原始 JSON 响应
        raw_json = response.text if hasattr(response, 'text') else json.dumps(annotation.model_dump())

        # 构建 trim_info 用于输出显示
        trim_info = {
            'original_duration': original_duration,
            'trimmed_duration': trimmed_duration,
            'trim_percent': actual_trim_percent,
            'failure_position_trimmed': failure_position_trimmed,
            'failure_position_original': failure_position_original,
        }

        return annotation, failure_step, raw_json, trim_info

    except Exception as e:
        import traceback
        return None, None, str(e) + "\n" + traceback.format_exc(), None


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
                        help="视频分析帧率 FPS")
    parser.add_argument("--trim_percent", type=float, default=0.5,
                        help="保留视频的百分比 (默认: 0.5 = 保留前50%，去掉后50%)")
    parser.add_argument("--api_delay", type=float, default=1.0,
                        help="每次 API 调用后的等待秒数 (默认: 1.0)")
    parser.add_argument("--thinking_level", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Gemini 思考深度 (默认: medium)")
    args = parser.parse_args()

    # API Key 验证
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: 请设置环境变量 GEMINI_API_KEY", file=sys.stderr)
        print("  export GEMINI_API_KEY='your-gemini-api-key'", file=sys.stderr)
        sys.exit(1)

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

        # 加载 episode 元数据
        fail_eps = load_episodes(jsonl)
        ep_to_video = match_videos_to_episodes(task_dir, fail_eps)

        print(f"\n{'='*70}")
        print(f"Task: {task}  ({len(fail_eps)} failure episodes)")
        print(f"Model: {args.model}, FPS: {args.fps}, Trim: {args.trim_percent*100:.0f}%")
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

            # 调用 Gemini 标注（视频会被裁剪）
            annotation, failure_step, reply, trim_info = annotate_episode(
                client, args.model, instruction, video_path, ep_length, fps=args.fps,
                thinking_level=args.thinking_level, trim_percent=args.trim_percent
            )

            # 计算故障时间（基于原始视频时长）
            if annotation and trim_info:
                # failure_time 基于「原始」视频时长和转换后的百分比
                failure_position_original = trim_info['failure_position_original']
                failure_time = round(failure_position_original / 100.0 * duration, 2) if duration else None

                pct_str_trimmed = f"{trim_info['failure_position_trimmed']}%"
                pct_str_original = f"{failure_position_original:.1f}%"
                time_str = f"{failure_time:.2f}s" if failure_time is not None else "N/A"
                failure_type_str = annotation.failure_type

                # 输出显示：(裁剪后%, 原始%, 秒数)
                print(f"         step={failure_step:3d} (trimmed {pct_str_trimmed:4s} → original {pct_str_original:6s}, {time_str:8s}) type={failure_type_str:20s} | {annotation.perception[:60]}...")

                task_results.append({
                    "episode_id": ep_id,
                    "instruction": instruction,
                    "length": ep_length,
                    "video_duration_sec": duration,
                    "trimmed_duration_sec": trim_info['trimmed_duration'],
                    "trim_percent": trim_info['trim_percent'],
                    "failure_position_trimmed_percent": trim_info['failure_position_trimmed'],
                    "failure_position_original_percent": failure_position_original,
                    'failure_step': failure_step,
                    "failure_time_sec": failure_time,
                    "failure_type": annotation.failure_type,
                    "perception": annotation.perception,
                    "summary": annotation.summary,
                    "reflection": annotation.reflection,
                    "plan": annotation.plan,
                    "next_subtask": annotation.next_subtask,
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
