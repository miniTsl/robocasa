import pandas as pd
import os

def calculate_success_rates(csv_path):
    """
    计算CSV文件中的整体成功率和topk成功率
    
    参数:
        csv_path: CSV文件的路径
    返回:
        overall_success_rate: 整体成功率
        topk_success_rate: topk成功率
    """
    # 异常处理：检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    # 1. 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 2. 计算整体成功率
    # success列中1的数量 / 总样本数
    total_trials = len(df)
    successful_trials = df['success'].sum()
    overall_success_rate = successful_trials / total_trials if total_trials > 0 else 0
    
    # 3. 计算topk成功率
    # 按episode_index分组，判断每个episode是否有至少一个成功的trial
    # groupby后用any()判断组内是否有True（success=1），sum()统计成功的episode数
    episode_success = df.groupby('episode_index')['success'].any()
    total_episodes = len(episode_success)
    successful_episodes = episode_success.sum()
    topk_success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
    
    return overall_success_rate, topk_success_rate

# ---------------------- 主执行代码 ----------------------
if __name__ == "__main__":
    # 请替换为你的CSV文件路径
    csv_file_path = "/home/zhangxinyue/robocasa/eval_trials/pi05_baseline_changeenv/complex/ArrangeVegetables/avg.csv"
    
    try:
        overall_rate, topk_rate = calculate_success_rates(csv_file_path)
        
        # 格式化输出结果（保留4位小数，更易读）
        print(f"所有测试数据的整体成功率: {overall_rate:.4f} ({overall_rate*100:.2f}%)")
        print(f"Episode的topk成功率: {topk_rate:.4f} ({topk_rate*100:.2f}%)")
        
    except Exception as e:
        print(f"执行出错: {e}")