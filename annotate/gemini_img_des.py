#!/usr/bin/env python3
"""
使用 Gemini 图片理解分析 RoboCasa 失败视频。
输出简洁分析：why and when 出错以及原因。
"""


from pydantic import BaseModel, Field
import argparse
import json
import os
import sys
import time
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

Observe the sequence of image frames from the failed robot manipulation video. The task is: '{instruction}'

Camera layout (each frame is a horizontal triptych, from left to right):
- hand: eye-in-hand camera view
- left: left agent (third-person) view
- right: right agent (third-person) view

Please analyze and describe: Why and when did the robot fail to complete this task? Answer in both English and Chinese.

Return a valid JSON object with a single field:
{{
  "analysis": "Your detailed explanation of why and when the robot failed in English and Chinese"
}}

Rules:
• Do not hallucinate: only describe what you see in the images
• Be concise and specific
• Output valid JSON with only the "analysis" field, in both English and Chinese
"""


# ── 工具函数 ────────────────────────────────────────────────────────────────

def get_mime_type(image_path):
    """
    根据文件扩展名获取 MIME 类型。

    Gemini API 支持的格式:
    - image/jpeg (JPEG)
    - image/png (PNG)
    - image/webp (WebP)
    - image/heic (HEIC)
    - image/heif (HEIF)
    - application/pdf (PDF)
    """
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".pdf": "application/pdf",
    }
    return mime_types.get(ext, "image/jpeg")


def read_image_bytes(image_path):
    """读取图片文件为字节"""
    with open(image_path, "rb") as f:
        return f.read()


def upload_image_to_gemini(client, image_path):
    """
    上传图片到 Gemini File API 并等待处理完成。
    返回 genai.types.File 对象。
    """
    print(f"    Uploading {Path(image_path).name} to Gemini File API...")
    image_file = client.files.upload(file=str(image_path))
    print(f"    Upload complete. File ID: {image_file.name}, State: {image_file.state.name}")

    # 轮询直到处理完成
    while image_file.state.name == "PROCESSING":
        print(f"    Processing... waiting 2 seconds")
        time.sleep(2)
        image_file = client.files.get(name=image_file.name)

    if image_file.state.name == "FAILED":
        raise RuntimeError(f"Image upload failed: {image_file.name}")

    print(f"    File ready. State: {image_file.state.name}")
    return image_file


def load_images_from_dir(image_dir, max_frames=None, sort_key=None, sample_interval=10):
    """
    从目录加载图片文件，每隔 sample_interval 张取一张。

    Gemini API 支持的图片格式: JPEG, PNG, WebP, HEIC, HEIF
    注意: 最多支持 3,600 张图片每个请求

    Args:
        image_dir: 图片目录路径
        max_frames: 最多加载多少张图片（None=无限制，Gemini API 限制最多 3,600）
        sort_key: 排序函数（默认按文件名排序）
        sample_interval: 采样间隔（默认10，即每隔10张取一张）

    Returns:
        (sampled_image_files, all_image_files, sample_indices)
        - sampled_image_files: 采样后的图片文件列表
        - all_image_files: 原始所有图片文件列表（用于百分比换算）
        - sample_indices: 采样图片在原始列表中的索引（用于百分比到秒数换算）
    """
    # Gemini API 支持的图片扩展名
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".pdf"}
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    # 排序
    image_files = sorted(image_files, key=sort_key or (lambda x: x.name))

    # 保存原始列表用于百分比换算
    all_image_files = image_files.copy()

    # 去掉后 50% 的图片（只保留前 50%）
    if image_files:
        cut_idx = int(len(image_files) * 0.5)
        # 保证至少留下一张
        if cut_idx < 1:
            cut_idx = 1
        image_files = image_files[:cut_idx]

    # 每隔 sample_interval 张取一张（包括第一张）
    sampled_files = []
    sample_indices = []
    for i, img in enumerate(image_files):
        if i % sample_interval == 0:
            sampled_files.append(img)
            sample_indices.append(i)

    # 限制数量
    if max_frames:
        sampled_files = sampled_files[:max_frames]
        sample_indices = sample_indices[:max_frames]

    print(f"    Loaded {len(image_files)} image(s) from {image_dir} (discarded last 50%)")
    print(f"    Sampled {len(sampled_files)} image(s) (every {sample_interval}th frame)")
    return sampled_files, all_image_files, sample_indices


def build_image_parts_from_dir(client, image_dir, max_frames=None, sample_interval=10):
    """
    从目录加载图片，构建图片内容部分，每隔 sample_interval 张取一张。

    根据 Gemini API 文档:
    - 文件大小: 最多 20MB (内联) 或通过 File API 上传
    - 推荐分辨率: 1024×1024 到 4096×4096 像素
    - JPEG 质量: 85-90% 质量提供最佳平衡
    - 最多支持: 3,600 张图片每个请求

    Args:
        client: Gemini 客户端
        image_dir: 图片目录路径
        max_frames: 最多加载多少张图片（默认限制为 3,600）
        sample_interval: 采样间隔（默认10）

    Returns:
        (image_parts_list, file_ids, all_image_count, sample_indices)
        - image_parts_list 为图片 Part 对象列表
        - file_ids 为上传到 File API 的文件 ID 列表
        - all_image_count 为原始所有图片数量（用于百分比换算）
        - sample_indices 为采样图片的原始索引（用于秒数换算）
    """
    # Gemini API 限制: 最多 3,600 张图片
    GEMINI_MAX_IMAGES = 3600

    # 如果未指定，使用 API 限制
    if max_frames is None:
        max_frames = GEMINI_MAX_IMAGES
    else:
        max_frames = min(max_frames, GEMINI_MAX_IMAGES)

    # 加载图片（返回采样图片、所有图片、采样索引）
    sampled_files, all_image_files, sample_indices = load_images_from_dir(
        image_dir, max_frames=max_frames, sample_interval=sample_interval
    )

    if not sampled_files:
        raise ValueError(f"No images found in {image_dir}")

    image_parts = []
    file_ids = []

    for image_path in sampled_files:
        file_size = os.path.getsize(image_path)
        mime_type = get_mime_type(image_path)

        # Gemini API 限制: 20MB 以下内联, 20MB+ 需要 File API
        MAX_INLINE_SIZE = 20 * 1024 * 1024  # 20MB

        if file_size < MAX_INLINE_SIZE:
            # 内联图片数据
            image_bytes = read_image_bytes(image_path)
            image_parts.append(
                types.Part(
                    inline_data=types.Blob(
                        data=image_bytes,
                        mime_type=mime_type
                    )
                )
            )
            print(f"    {image_path.name}: inline (size: {file_size/1024:.1f}KB, mime: {mime_type})")
        else:
            # File API 上传 (用于 >= 20MB 的文件)
            image_file = upload_image_to_gemini(client, image_path)
            image_parts.append(image_file)
            file_ids.append(image_file.name)
            print(f"    {image_path.name}: File API (size: {file_size/1024/1024:.1f}MB, mime: {mime_type})")

    print(f"  Total sampled images: {len(image_parts)} (from {len(all_image_files)} original images)")
    return image_parts, file_ids, len(all_image_files), sample_indices


def cleanup_image_files(client, file_ids):
    """删除 Gemini File API 中的图片文件以节省配额"""
    for file_id in file_ids:
        try:
            client.files.delete(name=file_id)
            print(f"    Deleted file: {file_id}")
        except Exception as e:
            print(f"    Warning: Failed to delete file {file_id}: {e}", file=sys.stderr)


def annotate_images(client, model, instruction, image_dir, max_frames=500, sample_interval=4, thinking_level="medium"):
    """
    调用 Gemini API 对图片序列进行分析。

    Args:
        client: Gemini 客户端
        model: 模型名称
        instruction: 任务指令
        image_dir: 图片目录
        max_frames: 最多分析多少张图片
        sample_interval: 采样间隔（默认4）
        thinking_level: 思考深度 (low/medium/high)

    Returns:
        (annotation, raw_response_json, total_frames) 或 (None, error_str, None)
        - annotation: 分析结果对象
        - raw_response_json: 原始 JSON 响应
        - total_frames: 视频总帧数
    """
    try:
        # 从目录加载图片
        print("  Loading images from directory...")
        image_parts, file_ids, total_frames, sample_indices = build_image_parts_from_dir(
            client, image_dir, max_frames=max_frames, sample_interval=sample_interval
        )

        # 构建完整内容：图片 + 文字提示词
        prompt_text = PROMPT.format(instruction=instruction)

        # 组合内容：所有图片 + 提示词
        contents = [prompt_text] + image_parts

        print(f"  Calling Gemini API with {len(image_parts)} image(s)...")

        # 调用 Gemini API，使用结构化输出
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FailureAnalysis,
                temperature=0.35,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
            ),
        )

        # 获取结构化结果
        annotation = response.parsed

        # 获取原始 JSON 响应
        raw_json = response.text if hasattr(response, 'text') else json.dumps(annotation.model_dump())

        # 清理已上传的文件
        if file_ids:
            print("  Cleaning up uploaded files...")
            cleanup_image_files(client, file_ids)

        return annotation, raw_json, total_frames

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return None, error_msg, None


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="使用 Gemini 分析 RoboCasa 失败视频图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="环境变量 GEMINI_API_KEY 必须设置为 Gemini API Key。")

    parser.add_argument("--base_dir", type=str, default="/home/sunyi/robocasa/debug/failed_videos_0405",
                        help="基础目录，包含episodex子文件夹 (默认: /home/sunyi/robocasa/debug/failed_videos_0405)")
    parser.add_argument("--episode", type=int, default=None,
                        help="指定分析的 episode ID（默认: None 表示分析所有）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: base_dir)")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Gemini 模型名 (默认: gemini-3-flash-preview)")
    parser.add_argument("--max_frames", type=int, default=500,
                        help="最多分析多少张图片 (默认: 500)")
    parser.add_argument("--sample_interval", type=int, default=2,
                        help="采样间隔：每隔多少张取一张 (默认: 2)")
    parser.add_argument("--thinking_level", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Gemini 思考深度 (默认: medium)")
    parser.add_argument("--api_delay", type=float, default=1.0,
                        help="API 调用后的等待秒数 (默认: 1.0)")
    args = parser.parse_args()

    # API Key 验证
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: 请设置环境变量 GEMINI_API_KEY", file=sys.stderr)
        print("  export GEMINI_API_KEY='your-gemini-api-key'", file=sys.stderr)
        sys.exit(1)

    # 初始化 Gemini 客户端
    client = genai.Client(api_key=api_key)

    # 设置输出目录
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir

    print(f"\n{'='*70}")
    print(f"Gemini Image Analysis")
    print(f"{'='*70}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Max frames: {args.max_frames}")
    print(f"Sample interval: every {args.sample_interval}th frame")
    print(f"Thinking level: {args.thinking_level}")
    print(f"{'='*70}\n")

    # 加载 episode 信息（从 metrics_log.jsonl）
    metrics_file = base_dir / "metrics_log.jsonl"
    if not metrics_file.exists():
        print(f"ERROR: {metrics_file} not found")
        sys.exit(1)

    episode_info = {}
    with open(metrics_file) as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line.strip())
            episode_info[ep["episode_id"]] = ep

    # 找到要分析的 episode 文件夹
    episode_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("episode")])

    if args.episode is not None:
        # 只分析指定的 episode
        episode_dirs = [d for d in episode_dirs if int(d.name.replace("episode", "")) == args.episode]

    results = []

    for ep_dir in episode_dirs:
        ep_id = int(ep_dir.name.replace("episode", ""))

        # 获取任务指令
        ep_data = episode_info.get(ep_id, {})
        instruction = ep_data.get("instruction", "N/A")

        print(f"ep{ep_id}: {instruction}")

        # 调用 Gemini 分析
        annotation, reply, total_frames = annotate_images(
            client, args.model, instruction, str(ep_dir),
            max_frames=args.max_frames, sample_interval=args.sample_interval,
            thinking_level=args.thinking_level
        )

        # 保存单个 episode 的结果
        if annotation:
            result = {
                "episode_id": ep_id,
                "instruction": instruction,
                "total_frames": total_frames,
                "analysis": annotation.analysis,
                "gemini_reply": reply,
            }
            print(f"  ✓ Analysis: {annotation.analysis[:100]}...")
            results.append(result)
        else:
            result = {
                "episode_id": ep_id,
                "instruction": instruction,
                "error": reply,
            }
            print(f"  ✗ Error: {reply[:100]}")
            results.append(result)

        time.sleep(args.api_delay)

    # 保存所有结果到文件
    output_file = output_dir / "analysis_images_2_medium.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
