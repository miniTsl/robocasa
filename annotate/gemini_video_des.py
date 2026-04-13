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
class FailureAnalysis(BaseModel):
    """简单的失败原因分析"""
    analysis: str = Field(
        ...,
        description="Why and when the robot failed to complete the task in English and Chinese"
    )


# ── 提示词 ────────────────────────────────────────────────────────────────
PROMPT = """You are an expert of robot manipulation task analysis.

Observe the failed robot manipulation video. The task is: '{instruction}'

Please analyze and describe: Why and when did the robot fail to complete this task? Answer in both English and Chinese.

Return a valid JSON object with a single field:
{{
  "analysis": "Your detailed explanation of why and when the robot failed in English and Chinese"
}}

Rules:
• Do not hallucinate: only describe what you see in the video
• Be concise and specific
• Output valid JSON with only the "analysis" field, in both English and Chinese
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
    调用 Gemini API 对单个失败 episode 进行分析。

    Args:
        client: Gemini 客户端
        model: 模型名称
        instruction: 任务指令
        video_path: 视频文件路径
        episode_data: episode 元数据字典
        fps: 视频分析帧率
        thinking_level: 思考深度
        trim_percent: 保留视频的百分比

    Returns:
        (annotation, raw_response_json, trim_info)
    """
    try:
        # 构建视频部分（包括裁剪）
        video_part, original_duration, trimmed_duration, actual_trim_percent = build_video_part(
            video_path, fps, trim_percent=trim_percent
        )

        prompt_text = PROMPT.format(instruction=instruction)
        response_schema = FailureAnalysis

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
        raw_json = response.text if hasattr(response, 'text') else json.dumps(annotation.model_dump())

        trim_info = {
            'original_duration': original_duration,
            'trimmed_duration': trimmed_duration,
            'trim_percent': actual_trim_percent,
        }

        return annotation, raw_json, trim_info

    except Exception as e:
        import traceback
        return None, str(e) + "\n" + traceback.format_exc(), None


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
        description="使用 Gemini 分析 RoboCasa 失败视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="环境变量 GEMINI_API_KEY 必须设置为 Gemini API Key。")

    parser.add_argument("--video_dir", type=str, default="/home/sunyi/robocasa/debug/failed_videos_0405",
                        help="视频目录路径 (默认: /home/sunyi/robocasa/debug/failed_videos_0405)")
    parser.add_argument("--output", type=str, default="analysis_gemini.json",
                        help="输出 JSON 文件路径 (默认: analysis_gemini.json)")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Gemini 模型名 (默认: gemini-3-flash-preview)")
    parser.add_argument("--fps", type=float, default=1,
                        help="视频分析帧率 FPS")
    parser.add_argument("--trim_percent", type=float, default=0.5,
                        help="保留视频的百分比 (默认: 0.5 = 保留前50%)")
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
    video_dir = Path(args.video_dir)

    # 输出文件路径（保存到 video_dir 所在目录）
    out_file = video_dir / args.output

    # 断点续跑
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[LOAD] Loaded {len(results)} existing analyses, resuming.")
    else:
        results = []

    # 加载 episode 元数据
    jsonl = video_dir / "metrics_log.jsonl"
    if not jsonl.exists():
        print(f"ERROR: {jsonl} not found")
        sys.exit(1)

    episodes = load_episodes(jsonl)

    # 匹配视频文件
    ep_to_video = {}
    for ep in episodes:
        ep_id = ep["episode_id"]
        video_path = video_dir / f"episode{ep_id}.mp4"
        if video_path.exists():
            ep_to_video[ep_id] = video_path

    print(f"\n{'='*70}")
    print(f"Directory: {video_dir}")
    print(f"Episodes: {len(episodes)} failure videos")
    print(f"Model: {args.model}, FPS: {args.fps}, Trim: {args.trim_percent*100:.0f}%")
    print(f"{'='*70}")

    for ep in episodes:
        ep_id = ep["episode_id"]

        # 检查是否已分析
        if any(r["episode_id"] == ep_id for r in results):
            print(f"  ep{ep_id}: already analyzed, skip")
            continue

        # 检查视频文件是否存在
        video_path = ep_to_video.get(ep_id)
        if not video_path or not video_path.exists():
            print(f"  ep{ep_id}: video not found")
            continue

        # 获取视频时长和指令
        duration = get_video_duration(video_path)
        instruction = ep.get("instruction", "N/A")
        ep_length = ep["actual_length"]

        duration_str = f"{duration:.1f}s" if duration is not None else "?s"
        print(f"  ep{ep_id} (len={ep_length:3d}, {duration_str})")
        print(f"         instruction: {instruction}")

        # 调用 Gemini 分析
        annotation, reply, trim_info = annotate_episode(
            client, args.model, instruction, video_path, ep, fps=args.fps,
            thinking_level=args.thinking_level, trim_percent=args.trim_percent
        )

        if annotation and trim_info:
            print(f"         analysis: {annotation.analysis[:80]}...")

            results.append({
                "episode_id": ep_id,
                "instruction": instruction,
                "length": ep_length,
                "video_duration_sec": duration,
                "trimmed_duration_sec": trim_info['trimmed_duration'],
                "trim_percent": trim_info['trim_percent'],
                "analysis": annotation.analysis,
                "gemini_reply": reply,
            })
        else:
            print(f"         ERROR: {reply[:100]}")
            results.append({
                "episode_id": ep_id,
                "instruction": instruction,
                "length": ep_length,
                "analysis": None,
                "gemini_reply": reply,
            })

        time.sleep(args.api_delay)

        # 保存进度
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✓ Done. Analysis saved to: {out_file}")
    print(f"  Total analyzed: {len(results)}")


if __name__ == "__main__":
    main()
