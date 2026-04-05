import cv2
import os
from pathlib import Path

video_path = "/data/zhangxinyue/robocasa_eval/1task_stage3_changeenv_recordact/PrepareCoffee/episode_2/trial_2.mp4"

# 检查视频文件是否存在
if not os.path.exists(video_path):
    print(f"视频文件不存在: {video_path}")
    exit(1)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"无法打开视频: {video_path}")
    exit(1)

# 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频帧率: {fps} FPS")
print(f"总帧数: {total_frames}")

# 创建输出目录
output_dir = Path(__file__).parent / "frames"
output_dir.mkdir(exist_ok=True)
print(f"输出目录: {output_dir}")

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
        print(f"已保存 {frame_count} 帧")

cap.release()
print(f"完成! 共保存了 {frame_count} 帧到 {output_dir}")
