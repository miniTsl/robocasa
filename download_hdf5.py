import json
import os
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- 配置项 ---
JSON_FILE = "single_stage_mg_im_links.json"  # 你的 JSON 文件名
MAX_WORKERS = 4          # 并发数
OUTPUT_DIR = "/data3/sunyi/robocasa/v0.1/single_stage_mg_hdf5"   # 目标下载文件夹

# 全局停止事件，用于向所有线程广播终止信号
stop_event = threading.Event()

def get_file_size(url):
    """
    Dry Run 核心逻辑：极速计算文件大小。
    """
    try:
        response = requests.get(url, stream=True, allow_redirects=True, timeout=10)
        size = int(response.headers.get('Content-Length', 0))
        response.close()
        return size
    except Exception:
        return 0

def download_file(name, url, total_size, position):
    """
    执行实际的下载逻辑，支持断点续传、实时进度条和安全中断
    """
    filename = f"{OUTPUT_DIR}/{name}.hdf5"
    downloaded_size = 0
    
    if os.path.exists(filename):
        downloaded_size = os.path.getsize(filename)
        
    if downloaded_size >= total_size and total_size > 0:
        return f"✅ {filename} 已完成"

    headers = {}
    if downloaded_size > 0:
        headers['Range'] = f"bytes={downloaded_size}-"

    try:
        response = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=15)
        response.raise_for_status()
        
        with open(filename, 'ab') as f, tqdm(
            desc=name[:20].ljust(20),
            total=total_size,
            initial=downloaded_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            position=position,
            leave=True
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024): 
                # 每次写入数据块前，检查是否收到了停止信号
                if stop_event.is_set():
                    return f"🛑 {filename} 已被手动强制中断"
                
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return f"✅ {filename} 成功"
    except requests.exceptions.RequestException as e:
        return f"❌ {filename} 失败或中断"

def main():
    if not os.path.exists(JSON_FILE):
        print(f"❌ 找不到文件 {JSON_FILE}！")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    print("🔍 正在执行 Dry Run，计算文件大小 (这可能需要几秒钟)...\n")
    
    file_infos = []
    total_bytes = 0
    
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for name, url in data.items():
                future = executor.submit(get_file_size, url)
                futures[future] = (name, url)
                
            for future in futures:
                name, url = futures[future]
                size = future.result()
                file_infos.append((name, url, size))
                total_bytes += size
    except KeyboardInterrupt:
        print("\n🛑 Dry Run 被手动取消。")
        return

    # --- 核心修改区：统一打印每个文件的大小明细 ---
    print("📄 文件大小明细:")
    for name, url, size in file_infos:
        size_gb = size / (1024**3)
        print(f" ├── {name}: {size_gb:.2f} GB")

    total_gb = total_bytes / (1024**3)
    print("-" * 40)
    print(f"📊 总计需要下载: {len(data)} 个文件")
    print(f"💾 总大小预计为: {total_gb:.2f} GB")
    print("-" * 40)
    # ----------------------------------------------

    user_input = input("\n🚀 是否开始下载？(y/n): ").strip().lower()
    if user_input != 'y':
        print("已取消下载。")
        return

    print("\n⏳ 开始并行下载... (随时按 Ctrl+C 可立即停止所有任务)\n")
    
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures_list = []
    for i, (name, url, size) in enumerate(file_infos):
        futures_list.append(executor.submit(download_file, name, url, size, i))

    try:
        # 主线程保持短循环监听状态
        while any(not future.done() for future in futures_list):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n🛑 接收到终止信号 (Ctrl+C)！正在通知所有线程立即停止...")
        stop_event.set()
        
        for future in futures_list:
            future.cancel()
            
        print("✅ 所有下载任务已被强制且安全地叫停。")
    finally:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    main()