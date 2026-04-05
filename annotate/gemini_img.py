#!/usr/bin/env python3
"""
Gemini Image Failure Annotator
===============================
使用 Gemini 图片理解 API + 结构化输出标注 RoboCasa 失败视频。

工作流程：
1. 读取视频对应的多帧图片（时间顺序）
2. 调用 Gemini，标出「最早致命迹象」的时间位置、原因、简短解释与恢复建议
3. 结构化输出（Pydantic + response_schema），写入 JSON

特点:
  - 支持本地图片上传（Gemini File API）
  - 支持 JPEG, PNG, GIF, WebP 等图片格式
  - 结构化输出（Pydantic + response_schema）
  - 多图片输入（多帧分析）
  - 输出结果保存为 JSON
  - 支持断点续跑

用法:
  # 基本用法
  python gemini_img.py \
      --image_dir /path/to/images \
      --output annotations_gemini.json

  # 自定义模型和最大帧数
  python gemini_img.py \
      --image_dir /path/to/images \
      --max_frames 10 \
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
from pathlib import Path
from google import genai
from google.genai import types


# —— 环境视角（单帧内从左到右子画面顺序）———————————————————
VIEW_NAMES_ORDERED = ("hand", "right", "left")

# ── 数据模型 ─────────────────────────────────────────────────────────────────
FailureCause = Literal[
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


class ImageAnnotation(BaseModel):
    """最早「致命迹象」时刻 + 原因 + 简短解释 + 恢复建议（英文输出）。"""

    warning_position: int = Field(
        ...,
        ge=0,
        le=100,
        description="Timeline percent (0-100) of the FIRST clear warning sign"
    )
    warning_type: FailureCause = Field(
        ...,
        description="Warning type category"
    )
    explanation: str = Field(
        ...,
        description="What is already wrong and why that makes later failure likely"
    )
    recovery: str = Field(
        ...,
        description="How to recover from current situation to avoid further failure"
    )


# ── 提示词 ────────────────────────────────────────────────────────────────
PROMPT = """You are an expert of robot manipulation tasks.
==========
You are observing a FAILED robot manipulation task in simulation.
Task: '{instruction}'

You are given a sequence of image frames extracted from the failed episode.
Camera layout (each frame is a horizontal triptych, from left to right) of images:
- hand: eye-in-hand camera view
- right: right agent (third-person) view
- left: left agent (third-person) view

==========
YOU SHOULD:
1. Find the **first moment** where a careful human would already see a **warning sign** that later failure is **certain** — the robot may still look partly fine (no big drop yet). Do **not** pick the first obvious failure, instead pick the **earliest** warning sign.
2. Output the warning type.
3. Output the explanation of the warning.
4. Output the recovery plan to avoid further failure.

==========
OUTPUT FORMAT:
MUST output ONLY a JSON object (no markdown, no extra text) with these exact fields (example):
{{
    "warning_position": An integer from 0 to 100 representing the timeline percent where the FIRST clear warning sign appears. 0% = first frame, 100% = last frame.
    "warning_type": A string - pick exactly ONE type from the labels below.
    "explanation": A string with analysis of what is wrong and why failure is likely. Answer in simple plain words.
    "recovery": A string with step-by-step recovery plan. Answer in simple plain words.
}}

==========
Labels for warning_type (pick ONE):
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
   - OTHER: none of the above
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

    # 去掉后 20% 的图片（只保留前 80%）
    if image_files:
        cut_idx = int(len(image_files) * 0.8)
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

    print(f"    Loaded {len(image_files)} image(s) from {image_dir} (discarded last 20%)")
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


def percent_to_seconds(percent, total_frames, fps=20):
    """
    将百分比位置转换为视频秒数。

    Args:
        percent: 百分比位置 (0-100)
        total_frames: 视频总帧数
        fps: 视频帧率 (默认20，即每秒20张图片)

    Returns:
        对应的秒数 (浮点数)
    """
    frame_index = int(percent / 100.0 * (total_frames - 1))
    seconds = frame_index / fps
    return seconds


def annotate_images(client, model, instruction, image_dir, max_frames=5, sample_interval=10, fps=20):
    """
    调用 Gemini API 对图片序列进行标注。

    根据 Gemini API 文档:
    - 支持多模式输入: 文本 + 多张图片
    - 使用 File API 处理大于 20MB 的图片
    - 支持结构化输出 (JSON schema)
    - 建议温度: 0.35 用于精确分析

    Args:
        client: Gemini 客户端
        model: 模型名称
        instruction: 任务指令
        image_dir: 图片目录
        max_frames: 最多分析多少张图片
        sample_interval: 采样间隔（默认10）
        fps: 视频帧率 (默认20)

    Returns:
        (annotation, raw_response_json, total_frames, failure_seconds) 或 (None, error_str, None, None)
        - annotation: 标注结果对象
        - raw_response_json: 原始 JSON 响应
        - total_frames: 视频总帧数
        - failure_seconds: 出错时刻（秒数）
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
        # 注意：Gemini API 支持混合内容 (图片 + 文本)
        contents = image_parts + [prompt_text]

        print(f"  Calling Gemini API with {len(image_parts)} image(s) and text prompt...")

        # 调用 Gemini API，使用结构化输出
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImageAnnotation,
                temperature=0.35,  # 低温度用于精确分析
                thinking_config=types.ThinkingConfig(thinking_level="medium")
            ),
        )

        # 获取结构化结果
        try:
            annotation: ImageAnnotation = response.parsed
        except Exception as e:
            print(f"  Error parsing response: {e}")
            print(f"  Response text: {response.text if hasattr(response, 'text') else 'N/A'}")
            raise

        # 获取原始 JSON 响应
        raw_json = response.text if hasattr(response, 'text') else json.dumps(annotation.model_dump())

        # 计算出错时刻的秒数
        failure_seconds = percent_to_seconds(annotation.warning_position, total_frames, fps=fps)

        # 清理已上传的文件
        if file_ids:
            print("  Cleaning up uploaded files...")
            cleanup_image_files(client, file_ids)

        return annotation, raw_json, total_frames, failure_seconds

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return None, error_msg, None, None


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="使用 Gemini 图片理解 + 结构化输出标注 RoboCasa 失败视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="环境变量 GEMINI_API_KEY 必须设置为 Gemini API Key。")

    parser.add_argument("--image_dir", type=str, required=True,
                        help="图片目录，包含视频帧图片 (*.jpg, *.png, *.gif, *.webp)")
    parser.add_argument("--instruction", type=str, default="pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button", help="任务指令")
    parser.add_argument("--output", type=str, default="annotation.json",
                        help="输出 JSON 文件路径 (默认: annotation.json)")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Gemini 模型名 (默认: gemini-3-flash-preview)")
    parser.add_argument("--max_frames", type=int, default=100,
                        help="最多分析多少张图片 (默认: 10, Gemini API 限制: 3600)")
    parser.add_argument("--sample_interval", type=int, default=10,
                        help="采样间隔：每隔多少张取一张 (默认: 10)")
    parser.add_argument("--fps", type=int, default=20,
                        help="视频帧率 (默认: 20，即每秒20张图片)")
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

    print(f"\n{'='*70}")
    print(f"Gemini Image Annotator")
    print(f"{'='*70}")
    print(f"Image directory: {args.image_dir}")
    print(f"Instruction: {args.instruction}")
    print(f"Model: {args.model}")
    print(f"Max frames: {args.max_frames} (Gemini API limit: 3600)")
    print(f"Sample interval: every {args.sample_interval}th frame")
    print(f"Video FPS: {args.fps}")
    print(f"Supported formats: JPEG, PNG, WebP, HEIC, HEIF, PDF")
    print(f"{'='*70}\n")

    # 调用 Gemini 标注
    print("Loading and analyzing images...")
    annotation, reply, total_frames, failure_seconds = annotate_images(
        client, args.model, args.instruction, args.image_dir,
        max_frames=args.max_frames, sample_interval=args.sample_interval, fps=args.fps
    )

    # 保存结果
    if annotation:
        result = {
            "instruction": args.instruction,
            "警告位置_进度百分比": annotation.warning_position,
            "警告时刻_秒数": round(failure_seconds, 2),
            "视频总帧数": total_frames,
            "警告类型": annotation.warning_type,
            "解释": annotation.explanation,
            "如何恢复": annotation.recovery,
            "gemini_reply": reply,
        }
        print(f"\n✓ Annotation successful!")
        print(f"  视频总帧数: {total_frames}")
        print(f"  警告位置（进度）: {annotation.warning_position}%")
        print(f"  警告时刻（秒数）: {failure_seconds:.2f}s")
        print(f"  警告类型: {annotation.warning_type}")
        print(f"  解释: {annotation.explanation[:120]}...")
    else:
        result = {
            "instruction": args.instruction,
            "error": reply,
        }
        print(f"\n✗ Annotation failed!")
        print(f"  Error: {reply[:200]}")

    # 保存到文件
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    time.sleep(args.api_delay)


if __name__ == "__main__":
    main()
