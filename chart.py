import json
import matplotlib.pyplot as plt
import datetime
import argparse
import numpy as np

def load_data(filepath):
    """加载 JSON 数据文件"""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

class TradingDataViewer:
    def __init__(self, data, instrument=None, twap_results=None):
        trade_data = data.get('trade', [])
        
        # 如果指定了 instrument，只处理该标的的数据
        if instrument:
            ticker, exchange = instrument.split()
            trade_data = [trade for trade in trade_data if trade['ticker'] == ticker]
            
        self.tickers = sorted(set(trade['ticker'] for trade in trade_data)) if trade_data else []
        
        # 检查是否有数据
        if not self.tickers:
            if instrument:
                raise ValueError(f"No data found for instrument: {instrument}")
            else:
                raise ValueError("No trading data found in the input file")

        self.current_index = 0
        self.ticker_data = {}
        self.twap_results = twap_results

        # 按股票代码分组数据
        for ticker in self.tickers:
            ticker_trades = [item for item in trade_data if item['ticker'] == ticker and item['price'] != 0]
            if ticker_trades:
                self.ticker_data[ticker] = {
                    'timestamps': np.array([int(item['timestamp']) / 1e9 for item in ticker_trades]),
                    'prices': np.array([item['price'] for item in ticker_trades]),
                    'volumes': np.array([int(item['volume']) for item in ticker_trades]),
                    'sides': np.array([item['side'] for item in ticker_trades])
                }

        # 创建图形对象
        plt.ion()
        self.fig = plt.figure(figsize=(12, 18))  # 增加总高度
        self.fig2 = plt.figure(figsize=(12, 15))  # 图表2
        self.current_ticker = None
        self.plot_current_ticker()
        plt.ioff()

    def plot_current_ticker(self):
        """绘制当前选中股票的图表"""
        current_ticker = self.tickers[self.current_index]
        if current_ticker == self.current_ticker:
            return

        self.current_ticker = current_ticker
        data = self.ticker_data[current_ticker]

        # 将时间戳转换为 datetime 对象
        datetime_timestamps = [datetime.datetime.fromtimestamp(ts) for ts in data['timestamps']]

        # 将时间戳转换为可读格式
        readable_timestamps = [dt.strftime('%H:%M:%S') for dt in datetime_timestamps]

        # 控制 X 轴标签数量以避免重叠
        total_points = len(readable_timestamps)
        if total_points > 20:
            step = max(total_points // 19, 1)
            xticks_pos = list(range(0, total_points - step, step))
            xticks_pos.append(total_points - 1)
            xticks_labels = [readable_timestamps[i] for i in xticks_pos]
        else:
            xticks_pos = range(len(readable_timestamps))
            xticks_labels = readable_timestamps

        self.fig.clear()
        # 调整主标题位置和整体布局
        self.fig.suptitle(f'Trading Data for {current_ticker} ({self.current_index + 1}/{len(self.tickers)})', 
                         y=0.98, fontsize=12)
        
        # 增加子图间距
        gs = self.fig.add_gridspec(5, 1, 
                                 height_ratios=[1, 1, 1, 1, 1], 
                                 hspace=0.5,    # 增加垂直间距
                                 top=0.95,      # 顶部边距
                                 bottom=0.05,   # 底部边距
                                 left=0.10,     # 左侧边距
                                 right=0.95)    # 右侧边距
        
        # 使用gridspec替代subplot
        ax1 = self.fig.add_subplot(gs[0])  # 价格时间序列
        ax1.plot(range(len(readable_timestamps)), data['prices'], label='Price', color='blue')
        ax1.set_xticks(xticks_pos)
        ax1.set_xticklabels(xticks_labels, rotation=45)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.set_title('Price Over Time')
        ax1.legend()
        ax1.grid(True)

        # 计算 VWAP
        cumulative_volume = np.cumsum(data['volumes'])
        cumulative_volume_price = np.cumsum(data['prices'] * data['volumes'])
        vwap = cumulative_volume_price / cumulative_volume

        ax2 = self.fig.add_subplot(gs[1])  # VWAP + 成交量
        ax2.plot(range(len(readable_timestamps)), vwap, label='VWAP', color='orange')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('VWAP')
        ax2.set_title('VWAP and Volume Over Time')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        # 创建双 Y 轴以绘制成交量
        ax2_volume = ax2.twinx()
        ax2_volume.bar(range(len(readable_timestamps)), data['volumes'], label='Volume', color='gray', alpha=0.3)
        ax2_volume.set_ylabel('Volume')
        ax2_volume.legend(loc='upper right')

        ax2.set_xticks(xticks_pos)
        ax2.set_xticklabels(xticks_labels, rotation=45)

        # 定义时间窗口（例如：1分钟）
        time_window = datetime.timedelta(minutes=1)

        # 初始化聚合数据
        net_volumes = []
        window_times = []

        # 将数据转换为 DataFrame 以便于处理
        import pandas as pd
        df = pd.DataFrame({
            'datetime': datetime_timestamps,
            'volume': data['volumes'],
            'price': data['prices'],
            'side': data['sides']
        })
        df.set_index('datetime', inplace=True)

        # 以时间窗口进行聚合
        grouped = df.groupby(pd.Grouper(freq='1Min'))

        for time_period, group in grouped:
            if group.empty:
                continue
            buy_volume = group[group['side'] == 'Buy']['volume'].sum()
            sell_volume = group[group['side'] == 'Sell']['volume'].sum()
            net_volumes.append(buy_volume - sell_volume)
            window_times.append(time_period.strftime('%H:%M:%S'))

        # 控制聚合数据的 X 轴标签
        total_window_points = len(window_times)
        if total_window_points > 20:
            step = max(total_window_points // 19, 1)
            window_xticks_pos = list(range(0, total_window_points - step, step))
            window_xticks_pos.append(total_window_points - 1)
            window_xticks_labels = [window_times[i] for i in window_xticks_pos]
        else:
            window_xticks_pos = range(len(window_times))
            window_xticks_labels = window_times

        # 恢复原始的 ax3 子图：净成交量图
        ax3 = self.fig.add_subplot(gs[2])  # 净成交量

        # 创建渐变色填充
        positive_volumes = np.array([max(0, v) for v in net_volumes])
        negative_volumes = np.array([min(0, v) for v in net_volumes])

        ax3.fill_between(range(len(window_times)), positive_volumes, 0, 
                        alpha=0.2, color='red', label='Buy > Sell')
        ax3.fill_between(range(len(window_times)), negative_volumes, 0, 
                        alpha=0.2, color='green', label='Sell > Buy')

        # 使用点的堆叠创建密集的线条效果
        for i in range(len(window_times)-1):
            # 在两个点之间创建多个插值点
            num_points = 20  # 插值点数量
            x = np.linspace(i, i+1, num_points)
            y = np.linspace(net_volumes[i], net_volumes[i+1], num_points)
            sizes = np.abs(y) / max(abs(np.array(net_volumes))) * 50 + 1  # 点大小随值变化
            
            # 用渐变色区分正负值
            colors = ['red' if val >= 0 else 'green' for val in y]
            ax3.scatter(x, y, c=colors, s=sizes, alpha=0.5, linewidth=0)

        # 标记最大最小值点
        max_idx = np.argmax(net_volumes)
        min_idx = np.argmin(net_volumes)
        ax3.scatter([max_idx], [net_volumes[max_idx]], color='red', s=150, zorder=5, marker='*')
        ax3.scatter([min_idx], [net_volumes[min_idx]], color='green', s=150, zorder=5, marker='*')

        # 设置其他属性
        ax3.set_xticks(window_xticks_pos)
        ax3.set_xticklabels(window_xticks_labels, rotation=45)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Net Volume')
        ax3.set_title(f'Net Volume Over Time (Window={time_window})')
        ax3.legend()
        ax3.grid(True)

        # 优化 ax4 子图：大单交易图
        ax4 = self.fig.add_subplot(gs[3])
        large_order_threshold = 200000  # 大于200,000的订单

        # 使用布尔索引筛选大单交易
        order_values = data['volumes'] * data['prices']
        large_orders = order_values >= large_order_threshold

        # 分别获取买入和卖出的大单交易索引
        buy_indices = np.where(large_orders & (data['sides'] == 'Buy'))[0]
        sell_indices = np.where(large_orders & (data['sides'] == 'Sell'))[0]

        # 计算买卖单比例
        total_large_orders = len(buy_indices) + len(sell_indices)
        buy_ratio = len(buy_indices) / total_large_orders * 100 if total_large_orders > 0 else 0
        sell_ratio = len(sell_indices) / total_large_orders * 100 if total_large_orders > 0 else 0

        # 绘制买入和卖出的散点
        has_buy = False
        has_sell = False
        if buy_indices.size > 0:
            buy_scatter = ax4.scatter(buy_indices, order_values[buy_indices], marker='^', color='red', s=100, 
                                    label=f'Buy ({buy_ratio:.1f}%)')
            has_buy = True
        if sell_indices.size > 0:
            sell_scatter = ax4.scatter(sell_indices, order_values[sell_indices], marker='v', color='green', s=100, 
                                     label=f'Sell ({sell_ratio:.1f}%)')
            has_sell = True

        # 设置 y 轴范围和格式
        if large_orders.any():
            ax4.set_ylim([order_values[large_orders].min() * 0.9, order_values[large_orders].max() * 1.1])
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

            # 构建图例
            legend_elements = []
            if has_buy:
                legend_elements.append(buy_scatter)
            if has_sell:
                legend_elements.append(sell_scatter)
            if legend_elements:
                ax4.legend()

        # 设置标题和其他属性
        ax4.set_xticks(xticks_pos)
        ax4.set_xticklabels(xticks_labels, rotation=45)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Order Value (CNY)')
        ax4.set_title('Large Orders (>200,000)')
        ax4.grid(True)

        # 计算累计净成交量
        cumulative_net_volume = np.cumsum(np.where(data['sides'] == 'Buy', data['volumes'], -data['volumes']))

        # 添加新的子图：累计净成交量
        ax5 = self.fig.add_subplot(gs[4])
        ax5.plot(range(len(readable_timestamps)), cumulative_net_volume, label='Cumulative Net Volume', color='purple')

        # 添加零线
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 填充正负区域
        ax5.fill_between(range(len(readable_timestamps)), cumulative_net_volume, 0,
                         where=cumulative_net_volume >= 0,
                         color='red', alpha=0.2, label='Net Buy')
        ax5.fill_between(range(len(readable_timestamps)), cumulative_net_volume, 0,
                         where=cumulative_net_volume < 0,
                         color='green', alpha=0.2, label='Net Sell')

        ax5.set_xticks(xticks_pos)
        ax5.set_xticklabels(xticks_labels, rotation=45)
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Cumulative Net Volume')
        ax5.set_title('Cumulative Net Volume Over Time')
        ax5.legend()
        ax5.grid(True)

        # 调整Y轴刻度格式
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1000:.0f}K' if abs(x) >= 1000 else str(int(x))))

        # 在绘制完成后更新画布
        self.fig.canvas.draw()
        self.fig2.canvas.draw()

        # 绘制图表2
        self.plot_order_heatmap()

    def plot_order_heatmap(self):
        """绘制订单热力图和成交量分布图"""
        self.fig2.clear()
        fig2 = self.fig2

        current_ticker = self.tickers[self.current_index]
        data = self.ticker_data[current_ticker]

        # 准备热力图数据
        timestamps = data['timestamps']
        prices = data['prices']
        volumes = data['volumes']
        sides = data['sides']

        # 定义时间和价格的区间
        num_bins = 20
        time_bins = np.linspace(timestamps.min(), timestamps.max(), num_bins + 1)
        price_bins = np.linspace(prices.min(), prices.max(), num_bins + 1)

        # 初始化热力图矩阵
        buy_heatmap = np.zeros((num_bins, num_bins))
        sell_heatmap = np.zeros((num_bins, num_bins))

        # 将数据分配到区间，并确保索引在有效范围内
        time_indices = np.clip(np.digitize(timestamps, time_bins) - 1, 0, num_bins - 1)
        price_indices = np.clip(np.digitize(prices, price_bins) - 1, 0, num_bins - 1)

        # 使用 NumPy 的高级索引填充热力图矩阵
        for side, heatmap in zip(['Buy', 'Sell'], [buy_heatmap, sell_heatmap]):
            mask = sides == side
            np.add.at(heatmap, (time_indices[mask], price_indices[mask]), volumes[mask])

        # 图表2 - 子图1：买入订单热力图
        ax1 = fig2.add_subplot(311)
        im1 = ax1.imshow(buy_heatmap, aspect='auto', cmap='Reds', origin='lower')
        fig2.colorbar(im1, ax=ax1, label='Buy Volume')
        ax1.set_title('Buy Order Heatmap')
        ax1.set_xlabel('Price Levels')
        ax1.set_ylabel('Time Periods')
        ax1.set_xticks(np.arange(num_bins))
        ax1.set_xticklabels([f"{price_bins[i]:.2f}" for i in range(num_bins)], rotation=45, ha='right')
        ax1.set_yticks(np.arange(num_bins))
        time_labels = [datetime.datetime.fromtimestamp(time_bins[i]).strftime('%H:%M') for i in range(num_bins)]
        ax1.set_yticklabels(time_labels)

        # 图表2 - 子图2：卖出订单热力图
        ax2 = fig2.add_subplot(312)
        im2 = ax2.imshow(sell_heatmap, aspect='auto', cmap='Greens', origin='lower')
        fig2.colorbar(im2, ax=ax2, label='Sell Volume')
        ax2.set_title('Sell Order Heatmap')
        ax2.set_xlabel('Price Levels')
        ax2.set_ylabel('Time Periods')
        ax2.set_xticks(np.arange(num_bins))
        ax2.set_xticklabels([f"{price_bins[i]:.2f}" for i in range(num_bins)], rotation=45, ha='right')
        ax2.set_yticks(np.arange(num_bins))
        ax2.set_yticklabels(time_labels)

        # 图表2 - 子图3：按价格的成交量分布
        ax3 = fig2.add_subplot(313)
        buy_volumes_by_price = np.zeros(num_bins)
        sell_volumes_by_price = np.zeros(num_bins)

        # 使用 NumPy 的高级索引计算成交量分布
        for side, volumes_by_price in zip(['Buy', 'Sell'], [buy_volumes_by_price, sell_volumes_by_price]):
            mask = sides == side
            np.add.at(volumes_by_price, price_indices[mask], volumes[mask])

        bar_width = 0.4
        indices = np.arange(num_bins)
        ax3.bar(indices - bar_width / 2, buy_volumes_by_price, 
                width=bar_width, label='Buy', color='red', alpha=0.7)
        ax3.bar(indices + bar_width / 2, sell_volumes_by_price, 
                width=bar_width, label='Sell', color='green', alpha=0.7)
        ax3.set_xticks(indices)
        ax3.set_xticklabels([f"{price_bins[i]:.2f}" for i in range(num_bins)], rotation=45, ha='right')
        ax3.set_xlabel('Price')
        ax3.set_ylabel('Volume')
        ax3.set_title('Volume Distribution by Price')
        ax3.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # 修改标题，增加当前序号信息
        fig2.suptitle(f'Order Analysis - {current_ticker} ({self.current_index + 1}/{len(self.tickers)})', 
                      fontsize=16)
        fig2.canvas.draw_idle()

    def on_key(self, event):
        """按键事件处理器"""
        old_index = self.current_index
        if event.key == 'left' and self.current_index > 0:
            self.current_index -= 1
            self.plot_current_ticker()
        elif event.key == 'right' and self.current_index < len(self.tickers) - 1:
            self.current_index += 1
            self.plot_current_ticker()

        # 删除原来的判断条件，因为我们希望每次按键都重新绘制
        plt.draw()  # 强制重绘所有图形

def plot_data(data, instrument=None, twap_results=None):
    """使用 TradingDataViewer 类显示数据"""
    try:
        viewer = TradingDataViewer(data, instrument, twap_results)
        viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
        viewer.fig2.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
        plt.show()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制交易数据。')
    parser.add_argument('--file', type=str, default='./data/data.json', help='数据文件路径')
    parser.add_argument('--instrument', type=str, help='指定要分析的标的，格式：股票代码 交易所，例如：600415 SSE')
    args = parser.parse_args()

    try:
        data = load_data(args.file)
        plot_data(data, args.instrument)
    except FileNotFoundError:
        print(f"Error: Could not find data file: {args.file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {args.file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)