"""
人为标注出错时间点，让大模型标注出错处的reasoning
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
class VideoAnnotation(BaseModel):

    failure_position: int = Field(
        ...,
        ge=0,
        le=100,
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


class VideoAnnotationGTFailTime(BaseModel):
    """当已知 gt_fail_time 时使用的数据模型，无需标注 failure_position"""
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
RULES/SAFEGUARDS:

• Do not hallucinate: never assert the presence of something not visible in the video. Use words like "possibly", "uncertain", "not sure", etc. to express your uncertainty.
• Output a valid JSON with only the above fields — no extra text or markdown.
• Keep the perception, reflection, plan, and next_subtask fields in the JSON as specific, concise and plain words.

==========
EXAMPLE OUTPUT:

{{
    "failure_position": 25,
    "perception": "Kitchen counter scene with a coffee machine on the counter. The upper cabinet door is open. The robot arm is at the front of the machine; the gripper is grasping the mug and is near the dispenser spout. But the mug is not upright on the tray yet and the gripper is about to lose contact with the mug.",
    "summary": "Move the arm to the cabinet; Grasp the mug in the cabinet; Move the mug to the drip tray of the coffee machine.",
    "reflection": "The robot is trying to 'Place the mug on the drip tray'. However, the gripper was tilted left and a little low and mug-tray collision was detected. The grip is slipping, leading to failed placement in the future.",
    "plan": "Grasps the mug and adjust its position for stable placement; Move the gripper to the start button; Press the start button.",
    "next_subtask": "Grasps the mug and adjust its position for stable placement."
}}
"""

# ── 提示词（已知出错时间点） ─────────────────────────────────────────────────
PROMPT_WITH_GT_FAILTIME = """You are an expert of robot manipulation task analysis.
==========
You are observing a FAILED robot manipulation video in simulation. And your job is to analyze the failure at the SPECIFIED time position.

==========
INPUT (will be supplied to you):

- Task: '{instruction}'
- Video: The video of the failed task (duration: {trimmed_duration:.1f}s).
- FAILURE TIME: The FIRST failure happens at {gt_fail_time:.2f}s ({failure_position_in_trimmed:.1f}% of the trimmed video).

The camera layout (each frame is a horizontal triptych, from left to right) in the video is:
- hand (eye-in-hand camera view)
- right (right agent (third-person) view)
- left (left agent (third-person) view)

==========
STRICT OUTPUT FORMAT (required):

Return one valid JSON object with exactly these fields:
{{
  "perception": "...",
  "summary": "...",
  "reflection": "...",
  "plan": "...",
  "next_subtask": "..."
}}

==========
WHAT TO INCLUDE IN EACH FIELD (content requirements):

**perception**
Analysis of the environment scene at the failure time ({gt_fail_time:.2f}s). You should ONLY focus on the critical objects related to the task. Include their attributes and spatial layout when necessary, such as color, position, shape, etc.

**summary**
The subtasks that have been executed BEFORE the failure at {gt_fail_time:.2f}s.

**reflection**
Analysis of what subtask the robot is trying to do and why the failure happens at the specified time.

**plan**
Subtasks for the remaining task. Begin with the subtask for recovery.

**next_subtask**
The immediate next subtask to be executed for recovery.

==========
RULES/SAFEGUARDS:

• Do not hallucinate: never assert the presence of something not visible in the video. Use words like "possibly", "uncertain", "not sure", etc. to express your uncertainty.
• Output a valid JSON with only the above fields — no extra text or markdown.
• Keep the perception, reflection, plan, and next_subtask fields in the JSON as specific, concise and plain words.
• Focus your analysis on the specified failure time: {gt_fail_time:.2f}s.

==========
EXAMPLE OUTPUT:

{{
    "perception": "Kitchen counter scene with a coffee machine on the counter. The upper cabinet door is open. The robot arm is at the front of the machine; the gripper is grasping the mug and is near the dispenser spout. But the mug is not upright on the tray yet and the gripper is about to lose contact with the mug.",
    "summary": "Move the arm to the cabinet; Grasp the mug in the cabinet; Move the mug to the drip tray of the coffee machine.",
    "reflection": "The robot is trying to 'Place the mug on the drip tray'. However, the gripper was tilted left and a little low and mug-tray collision was detected. The grip is slipping, leading to failed placement in the future.",
    "plan": "Grasps the mug and adjust its position for stable placement; Move the gripper to the start button; Press the start button.",
    "next_subtask": "Grasps the mug and adjust its position for stable placement."
}}
"""

# ── 工具函数 ────────────────────────────────────────────────────────────────
def load_episodes(jsonl_path, num_videos_per_folder=None):
    """从 metrics_log.jsonl 读取 episode 元数据，按 episode_id 排序

    Args:
        jsonl_path: jsonl 文件路径
        num_videos_per_folder: 每个文件夹内选择的视频数，从 id 从小往大选。None 表示全选
    """
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line.strip())
            if not ep.get("success", False):
                episodes.append(ep)

    episodes = sorted(episodes, key=lambda e: e.get("episode_id", 0))

    # 如果指定了 num_videos_per_folder，只取前 num_videos_per_folder 个
    if num_videos_per_folder is not None and num_videos_per_folder > 0:
        episodes = episodes[:num_videos_per_folder]

    return episodes


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
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
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


def annotate_episode(client, model, instruction, video_path, episode_data, fps=1, thinking_level="medium", trim_percent=0.5):
    """
    调用 Gemini API 对单个失败 episode 进行标注。

    当 episode_data 中有 gt_fail_time 时，使用带有 ground truth 出错时间的 prompt，
    只需要标注 reasoning，无需标注 failure_position。

    Args:
        client: Gemini 客户端
        model: 模型名称
        instruction: 任务指令
        video_path: 视频文件路径
        episode_data: episode 元数据字典，包含 actual_length 和 gt_fail_time（秒）
        fps: 视频分析帧率
        thinking_level: 思考深度
        trim_percent: 保留视频的百分比

    Returns:
        (annotation, failure_step, raw_response_json, trim_info, has_gt_fail_time)
    """
    try:
        # 构建视频部分（包括裁剪）
        video_part, original_duration, trimmed_duration, actual_trim_percent = build_video_part(
            video_path, fps, trim_percent=trim_percent
        )

        gt_fail_time = episode_data.get("gt_fail_time")
        has_gt_fail_time = gt_fail_time is not None
        ep_length = episode_data["actual_length"]

        if has_gt_fail_time:
            # 计算 gt_fail_time 在裁剪后视频中的百分比
            if trimmed_duration and trimmed_duration > 0:
                failure_position_in_trimmed = (gt_fail_time / trimmed_duration * 100)
            else:
                failure_position_in_trimmed = 0

            prompt_text = PROMPT_WITH_GT_FAILTIME.format(
                instruction=instruction,
                gt_fail_time=gt_fail_time,
                trimmed_duration=trimmed_duration or 0,
                failure_position_in_trimmed=failure_position_in_trimmed
            )
            response_schema = VideoAnnotationGTFailTime
        else:
            prompt_text = PROMPT.format(instruction=instruction)
            response_schema = VideoAnnotation

        # 构建配置
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.35,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        )

        # 调用 Gemini API
        response = client.models.generate_content(
            model=model,
            contents=[video_part, prompt_text],
            config=config,
        )

        annotation = response.parsed

        if has_gt_fail_time:
            # 使用 gt_fail_time 的百分比（相对原始视频）
            failure_position_original = (gt_fail_time / original_duration * 100) if original_duration else 0
            failure_step = round(failure_position_original / 100.0 * ep_length)
            failure_position_trimmed = None
        else:
            # 转换模型输出的百分比（相对裁剪后）到原始视频
            failure_position_trimmed = annotation.failure_position
            failure_position_original = failure_position_trimmed * actual_trim_percent
            failure_step = round(failure_position_original / 100.0 * ep_length)

        raw_json = response.text if hasattr(response, 'text') else json.dumps(annotation.model_dump())

        trim_info = {
            'original_duration': original_duration,
            'trimmed_duration': trimmed_duration,
            'trim_percent': actual_trim_percent,
            'failure_position_trimmed': failure_position_trimmed,
            'failure_position_original': failure_position_original,
            'has_gt_fail_time': has_gt_fail_time,
        }

        return annotation, failure_step, raw_json, trim_info, has_gt_fail_time

    except Exception as e:
        import traceback
        return None, None, str(e) + "\n" + traceback.format_exc(), None, False


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
    parser.add_argument("--num_videos_per_folder", type=int, default=None,
                        help="每个文件夹内选择的视频数，从 id 从小往大选。None 或 0 表示全选 (默认: None)")
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
        fail_eps = load_episodes(jsonl, num_videos_per_folder=args.num_videos_per_folder)
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
                print(f"  ep{ep_id:3d}: video not found at {video_path}")
                continue

            # 获取视频时长
            duration = get_video_duration(video_path)
            instruction = ep.get("instruction", "N/A")

            ep_length = ep["actual_length"]
            duration_str = f"{duration:.1f}s" if duration is not None else "?s"
            print(f"  ep{ep_id:3d} (len={ep_length:3d}, {duration_str})")
            print(f"         instruction: {instruction[:60]}...")

            # 调用 Gemini 标注（视频会被裁剪）
            annotation, failure_step, reply, trim_info, has_gt_fail_time = annotate_episode(
                client, args.model, instruction, video_path, ep, fps=args.fps,
                thinking_level=args.thinking_level, trim_percent=args.trim_percent
            )

            # 计算故障时间（基于原始视频时长）
            if annotation and trim_info:
                # failure_time 基于「原始」视频时长和转换后的百分比
                failure_position_original = trim_info['failure_position_original']
                failure_time = round(failure_position_original / 100.0 * duration, 2) if duration else None

                if has_gt_fail_time:
                    # 使用 gt_fail_time
                    pct_str = f"{failure_position_original:.1f}%"
                    time_str = f"{ep.get('gt_fail_time'):.2f}s (gt)"
                    print(f"         step={failure_step:3d} ({pct_str:6s}, {time_str:12s}) | {annotation.perception[:60]}...")
                else:
                    # 正常流程
                    pct_str_trimmed = f"{trim_info['failure_position_trimmed']}%" if trim_info['failure_position_trimmed'] is not None else "N/A"
                    pct_str_original = f"{failure_position_original:.1f}%"
                    time_str = f"{failure_time:.2f}s" if failure_time is not None else "N/A"
                    print(f"         step={failure_step:3d} (trimmed {pct_str_trimmed:4s} → original {pct_str_original:6s}, {time_str:8s}) | {annotation.perception[:60]}...")

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
                    "has_gt_fail_time": has_gt_fail_time,
                    "gt_fail_time": ep.get("gt_fail_time"),
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
