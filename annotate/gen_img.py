import cv2
from pathlib import Path

# 设置源目录和输出目录
video_dir = Path("/home/sunyi/robocasa/debug/failed_videos_0405")

# 检查目录是否存在
if not video_dir.exists():
    print(f"目录不存在: {video_dir}")
    exit(1)

# 找到所有 mp4 视频文件
video_files = sorted(video_dir.glob("episode*.mp4"))
print(f"找到 {len(video_files)} 个视频文件")

for video_path in video_files:
    # 提取 episode 编号
    episode_name = video_path.stem  # 例如 "episode0"

    # 创建输出目录
    output_dir = video_dir / f"{episode_name}"
    output_dir.mkdir(exist_ok=True)

    # 检查视频文件是否存在
    if not video_path.exists():
        print(f"✗ {episode_name}: 视频文件不存在")
        continue

    # 打开视频文件
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"✗ {episode_name}: 无法打开视频")
        continue

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ {episode_name} (FPS: {fps}, 帧数: {total_frames})")

    # 逐帧读取并保存
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 保存图片
        output_path = output_dir / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(output_path), frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  已保存 {frame_count} 帧")

    cap.release()
    print(f"  完成! 共保存了 {frame_count} 帧到 {output_dir}\n")

print(f"全部完成!")
