import json
import argparse
from collections import defaultdict
from typing import List, Dict
import numpy as np
from datetime import datetime

class TwapAnalyzer:
    def __init__(self, data: Dict, instrument=None, thresholds: Dict[str, float] = None, debug=False):
        trade_data = data.get('trade', [])
        
        # 如果指定了 instrument，只处理该标的的数据
        if instrument:
            ticker, exchange = instrument.split()
            trade_data = [trade for trade in trade_data if trade['ticker'] == ticker]
            
        self.data = sorted(trade_data, key=lambda x: int(x['timestamp'])) if trade_data else []
        self.trades_by_ticker = self._group_by_ticker()
        self.debug = debug
        self.thresholds = thresholds

    def _group_by_ticker(self) -> Dict:
        """将交易按照ticker分组"""
        grouped = defaultdict(list)
        for trade in self.data:
            grouped[trade['ticker']].append(trade)
        return grouped

    def _find_continuous_trades(self, trades: List[Dict], side: str, min_trades: int = 4) -> List[List[Dict]]:
        """查找连续的相同方向交易，忽略中间的反向交易，并在时间上连续"""
        continuous_trades = []
        current_sequence = []
        last_timestamp = None
        max_gap = self.thresholds['max_gap'] * 1e9  # 将秒转换为纳秒

        for trade in trades:
            current_timestamp = int(trade['timestamp'])

            if trade['side'] != side:
                continue  # 忽略反向交易

            if last_timestamp is None:
                current_sequence.append(trade)
            else:
                time_diff = current_timestamp - last_timestamp
                if time_diff <= max_gap:
                    current_sequence.append(trade)
                else:
                    if len(current_sequence) >= min_trades:
                        continuous_trades.append(current_sequence)
                    current_sequence = [trade]

            last_timestamp = current_timestamp

        if len(current_sequence) >= min_trades:
            continuous_trades.append(current_sequence)

        return continuous_trades

    def _analyze_time_intervals(self, trades: List[Dict]) -> dict:
        """分析时间间隔的均匀性"""
        timestamps = [int(trade['timestamp']) for trade in trades]
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        # 使用阈值参数
        interval_uniformity = std_interval / mean_interval if mean_interval != 0 else float('inf')
        is_uniform = interval_uniformity < self.thresholds['time_interval_cv']

        if self.debug:
            print(f"时间间隔分析：mean={mean_interval}, std={std_interval}, uniformity={interval_uniformity}")

        return {
            'mean_interval': mean_interval / 1e9,  # 转换为秒
            'std_interval': std_interval / 1e9,
            'interval_uniformity': interval_uniformity,
            'is_uniform': is_uniform
        }

    def _analyze_volumes(self, trades: List[Dict]) -> dict:
        """分析交易量的一致性"""
        volumes = [trade['volume'] for trade in trades]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        # 使用阈值参数
        volume_consistency = std_volume / mean_volume if mean_volume != 0 else float('inf')
        is_consistent = volume_consistency < self.thresholds['volume_cv']

        if self.debug:
            print(f"交易量分析：mean={mean_volume}, std={std_volume}, consistency={volume_consistency}")

        return {
            'mean_volume': mean_volume,
            'std_volume': std_volume,
            'volume_consistency': volume_consistency,
            'is_consistent': is_consistent
        }

    def _analyze_price_movement(self, trades: List[Dict]) -> dict:
        """分析价格波动的合理性"""
        prices = [trade['price'] for trade in trades]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        price_volatility = std_price / mean_price if mean_price != 0 else float('inf')

        # 使用阈值参数
        is_reasonable_volatility = price_volatility < self.thresholds['price_volatility']

        # 检查价格趋势
        x = np.arange(len(prices))
        if len(prices) > 1:
            slope, _ = np.polyfit(x, prices, 1)
            price_trend = slope / mean_price  # 归一化斜率
        else:
            price_trend = 0
        is_stable_trend = abs(price_trend) < self.thresholds['price_trend']

        is_reasonable = is_reasonable_volatility and is_stable_trend

        if self.debug:
            print(f"价格波动分析：mean={mean_price}, std={std_price}, volatility={price_volatility}")
            print(f"价格趋势分析：trend={price_trend}")

        return {
            'price_volatility': price_volatility,
            'price_trend': price_trend,
            'is_reasonable': is_reasonable
        }

    def analyze(self, min_trades: int = 4) -> Dict:
        """分析可能的TWAP行为"""
        results = {}

        for ticker, trades in self.trades_by_ticker.items():
            buy_sequences = self._find_continuous_trades(trades, "Buy", min_trades)
            # 不再分析卖出序列

            ticker_results = {
                'buy_twap_candidates': []
            }

            # 分析买入序列
            for sequence in buy_sequences:
                time_analysis = self._analyze_time_intervals(sequence)
                volume_analysis = self._analyze_volumes(sequence)
                price_analysis = self._analyze_price_movement(sequence)

                # 所有条件都满足才认为是TWAP
                if (time_analysis['is_uniform'] and
                    volume_analysis['is_consistent'] and
                    price_analysis['is_reasonable']):
                    start_time = datetime.fromtimestamp(int(sequence[0]['timestamp']) / 1e9)
                    end_time = datetime.fromtimestamp(int(sequence[-1]['timestamp']) / 1e9)

                    ticker_results['buy_twap_candidates'].append({
                        'trades_count': len(sequence),
                        'start_time': start_time.strftime('%H:%M:%S'),
                        'end_time': end_time.strftime('%H:%M:%S'),
                        'avg_interval': time_analysis['mean_interval'],
                        'std_interval': time_analysis['std_interval'],
                        'avg_volume': volume_analysis['mean_volume'],
                        'std_volume': volume_analysis['std_volume'],
                        'price_volatility': price_analysis['price_volatility'],
                        'price_trend': price_analysis['price_trend'],
                        'price_range': [min(t['price'] for t in sequence),
                                        max(t['price'] for t in sequence)]
                    })
                elif self.debug:
                    print(f"买入序列不符合TWAP条件：")
                    print(f"时间间隔一致性：{time_analysis['is_uniform']}")
                    print(f"交易量一致性：{volume_analysis['is_consistent']}")
                    print(f"价格波动合理性：{price_analysis['is_reasonable']}")

            if ticker_results['buy_twap_candidates']:
                results[ticker] = ticker_results

        return results

if __name__ == "__main__":
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='Analyze TWAP trading patterns.')
    parser.add_argument('--file', type=str, default='./data/data.json', help='Path to the data file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--min_trades', type=int, default=4, help='Minimum number of trades to consider')
    parser.add_argument('--max_gap', type=float, default=3600, help='Maximum time gap between trades in seconds')
    parser.add_argument('--time_interval_cv', type=float, default=0.2, help='Threshold for time interval coefficient of variation')
    parser.add_argument('--volume_cv', type=float, default=0.05, help='Threshold for volume coefficient of variation')
    parser.add_argument('--price_volatility', type=float, default=0.01, help='Threshold for price volatility')
    parser.add_argument('--price_trend', type=float, default=0.005, help='Threshold for price trend')
    parser.add_argument('--instrument', type=str, help='指定要分析的标的，格式：股票代码 交易所，例如：600415 SSE')
    args = parser.parse_args()

    # 加载数据
    with open(args.file, 'r') as f:
        data = json.load(f)

    # 设置阈值
    thresholds = {
        'max_gap': args.max_gap,
        'time_interval_cv': args.time_interval_cv,
        'volume_cv': args.volume_cv,
        'price_volatility': args.price_volatility,
        'price_trend': args.price_trend
    }

    # 分析TWAP
    analyzer = TwapAnalyzer(data, args.instrument, thresholds=thresholds, debug=args.debug)
    results = analyzer.analyze(min_trades=args.min_trades)

    if args.debug:
        print(f"总交易数: {len(data)}")
        print(f"交易的股票: {list(set(item['ticker'] for item in data))}")

    # 如果没有检测到TWAP行为，只打印"无"
    if not results:
        print("无")
    else:
        for ticker, result in results.items():
            if result['buy_twap_candidates']:
                print(f"\n股票代码: {ticker}")
                print("\n可能的TWAP买入:")
                for candidate in result['buy_twap_candidates']:
                    print(f"  交易次数: {candidate['trades_count']}")
                    print(f"  时间范围: {candidate['start_time']} - {candidate['end_time']}")
                    print(f"  平均间隔: {candidate['avg_interval']:.2f}秒")
                    print(f"  间隔标准差: {candidate['std_interval']:.2f}秒")
                    print(f"  平均成交量: {candidate['avg_volume']:.0f}")
                    print(f"  成交量标准差: {candidate['std_volume']:.2f}")
                    print(f"  价格波动率: {candidate['price_volatility']:.4f}")
                    print(f"  价格趋势: {candidate['price_trend']:.6f}")
                    print(f"  价格范围: {candidate['price_range'][0]} - {candidate['price_range'][1]}")
            else:
                print("无")