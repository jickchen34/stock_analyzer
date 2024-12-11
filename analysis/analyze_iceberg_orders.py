import json
from typing import List, Dict
from datetime import datetime
import argparse
import os  # 新增

class IcebergOrderAnalyzer:
    def __init__(self, time_window: int = 300, volume_threshold: int = 100, 
                 min_repeat_times: int = 3):
        self.time_window = time_window  # 时间窗口(秒)
        self.volume_threshold = volume_threshold  # 最小交易量阈值
        self.min_repeat_times = min_repeat_times  # 最小重复次数

    def load_trade_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['trade']

    def convert_timestamp(self, timestamp: str) -> datetime:
        # 转换时间戳为datetime对象
        return datetime.fromtimestamp(int(timestamp)/1000)

    def analyze_trades(self, trades: List[Dict]) -> List[Dict]:
        iceberg_candidates = []
        trades.sort(key=lambda x: x['timestamp'])  # 按时间排序
        
        i = 0
        while i < len(trades):
            current_trade = trades[i]
            similar_trades = []
            
            # 寻找相似特征的交易
            for j in range(i + 1, len(trades)):
                next_trade = trades[j]
                
                time_diff = (int(next_trade['timestamp']) - int(current_trade['timestamp']))/1000
                
                # 检查是否在时间窗口内
                if time_diff > self.time_window:
                    break
                    
                # 检查是否具有相似特征
                if (current_trade['price'] == next_trade['price'] and
                    current_trade['side'] == next_trade['side'] and
                    abs(current_trade['volume'] - next_trade['volume']) < self.volume_threshold):
                    similar_trades.append(next_trade)
            
            # 如果找到足够多的相似交易，可能是冰山单
            if len(similar_trades) >= self.min_repeat_times - 1:
                iceberg_pattern = {
                    'start_time': self.convert_timestamp(current_trade['timestamp']),
                    'end_time': self.convert_timestamp(similar_trades[-1]['timestamp']),
                    'price': current_trade['price'],
                    'side': current_trade['side'],
                    'avg_volume': sum([t['volume'] for t in [current_trade] + similar_trades]) / 
                                (len(similar_trades) + 1),
                    'total_volume': sum([t['volume'] for t in [current_trade] + similar_trades]),
                    'repeat_times': len(similar_trades) + 1
                }
                iceberg_candidates.append(iceberg_pattern)
                i += len(similar_trades)
            else:
                i += 1
                
        return iceberg_candidates

    def print_analysis_results(self, iceberg_orders: List[Dict]):
        print("\n=== 冰山订单分析结果 ===")
        print(f"发现 {len(iceberg_orders)} 个潜在的冰山订单模式\n")
        
        for idx, order in enumerate(iceberg_orders, 1):
            print(f"冰山订单 #{idx}:")
            print(f"时间区间: {order['start_time']} - {order['end_time']}")
            print(f"交易方向: {order['side']}")
            print(f"价格: {order['price']}")
            print(f"平均交易量: {order['avg_volume']:.2f}")
            print(f"总交易量: {order['total_volume']}")
            print(f"重复次数: {order['repeat_times']}")
            print("-" * 50)

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='分析交易数据中的冰山订单')
    parser.add_argument('--file', type=str, required=True, help='输入的JSON交易数据文件路径')
    parser.add_argument('--window', type=int, default=300, help='时间窗口(秒),默认300秒')
    parser.add_argument('--volume', type=int, default=100, help='交易量误差范围,默认100')
    parser.add_argument('--repeat', type=int, default=3, help='最少重复次数,默认3次')
    
    args = parser.parse_args()
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的完整路径
    data_path = os.path.join(os.path.dirname(current_dir), args.file)
    
    analyzer = IcebergOrderAnalyzer(
        time_window=args.window,
        volume_threshold=args.volume,
        min_repeat_times=args.repeat
    )
    
    trades = analyzer.load_trade_data(data_path)
    iceberg_orders = analyzer.analyze_trades(trades)
    analyzer.print_analysis_results(iceberg_orders)

if __name__ == "__main__":
    main()