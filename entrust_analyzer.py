import json
import matplotlib.pyplot as plt 
import seaborn as sns
import logging
from collections import defaultdict, Counter

def load_entrust_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data["entrust"]

def analyze_entrust_types(entrusts, ax):
    """分析委托类型的分布"""
    type_counter = Counter(e["order_type"] for e in entrusts)
    
    # 可视化各类型委托的数量
    types = list(type_counter.keys())
    counts = list(type_counter.values())
    
    ax.bar(types, counts, color='skyblue', alpha=0.7)
    ax.set_title("Distribution of Entrust Types")
    ax.set_xlabel("Order Type")
    ax.set_ylabel("Count")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    logging.info("=== Entrust Type Statistics ===")
    for type_name, count in type_counter.items():
        logging.info(f"{type_name}: {count}")
        
    return type_counter

def analyze_entrust_timeline(entrusts, ax):
    """分析委托的时间分布"""
    # 按时间排序委托
    timestamps = []
    counts = []
    
    # 按时间窗口统计委托数量
    time_window = defaultdict(int)
    for entrust in entrusts:
        time_key = int(int(entrust["timestamp"]) / 1e9)  # 转换为秒级时间戳
        time_window[time_key] += 1
    
    # 排序并准备绘图数据
    sorted_times = sorted(time_window.items())
    timestamps = [x[0] for x in sorted_times]
    counts = [x[1] for x in sorted_times]
    
    ax.plot(timestamps, counts, '-o', markersize=3, alpha=0.7)
    ax.set_title("Entrust Timeline Distribution")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of Entrusts")
    ax.grid(True, alpha=0.3)

def analyze_entrust_patterns(entrusts):
    """分析委托模式与规律"""
    # 统计连续相同类型的委托
    prev_type = None
    current_streak = 0
    type_streaks = defaultdict(list)
    
    for entrust in entrusts:
        current_type = entrust["order_type"]
        if current_type == prev_type:
            current_streak += 1
        else:
            if prev_type is not None:
                type_streaks[prev_type].append(current_streak)
            current_streak = 1
            prev_type = current_type
    
    # 添加最后一个streak
    if prev_type is not None:
        type_streaks[prev_type].append(current_streak)
    
    # 打印统计结果
    logging.info("\n=== Entrust Pattern Analysis ===")
    for order_type, streaks in type_streaks.items():
        avg_streak = sum(streaks) / len(streaks) if streaks else 0
        max_streak = max(streaks) if streaks else 0
        logging.info(f"{order_type}:")
        logging.info(f"  Average streak length: {avg_streak:.2f}")
        logging.info(f"  Maximum streak length: {max_streak}")

def analyze_entrusts(filepath, show_type_dist=False):
    """主分析函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    entrusts = load_entrust_data(filepath)
    logging.info("Analyzing entrust data...")
    
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    if show_type_dist:
        # 显示类型分布时使用两个子图
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        type_stats = analyze_entrust_types(entrusts, ax1)
        analyze_entrust_timeline(entrusts, ax2)
    else:
        # 不显示类型分布时只显示时间线图
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
        analyze_entrust_timeline(entrusts, ax)
    
    analyze_entrust_patterns(entrusts)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze entrust data from JSON file.')
    parser.add_argument('--file', type=str, default='./data/data.json',
                      help='Path to the JSON data file containing entrust information')
    parser.add_argument('--type', action='store_true',
                      help='Show distribution of entrust types')
    
    args = parser.parse_args()
    analyze_entrusts(args.file, args.type)