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
    def __init__(self, data, twap_results=None):
        trade_data = data.get('trade', [])
        self.tickers = sorted(set(trade['ticker'] for trade in trade_data)) if trade_data else []
        self.current_index = 0
        self.ticker_data = {}
        self.twap_results = twap_results

        # 按股票代��分组数据
        for ticker in self.tickers:
            ticker_trades = [item for item in trade_data if item['ticker'] == ticker and item['price'] != 0]
            if ticker_trades:
                self.ticker_data[ticker] = {
                    'timestamps': [int(item['timestamp']) / 1e9 for item in ticker_trades],
                    'prices': [item['price'] for item in ticker_trades],
                    'volumes': [int(item['volume']) for item in ticker_trades],
                    'sides': [item['side'] for item in ticker_trades]
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
        cumulative_volume_price = np.cumsum(np.array(data['prices']) * np.array(data['volumes']))
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

        window_start = datetime_timestamps[0]
        buy_volume = 0
        sell_volume = 0

        for dt, volume, price, side in zip(datetime_timestamps, data['volumes'], data['prices'], data['sides']):
            if dt - window_start >= time_window:
                net_volumes.append(buy_volume - sell_volume)
                window_times.append(window_start.strftime('%H:%M:%S'))

                # Reset window
                window_start = dt
                buy_volume = 0
                sell_volume = 0

            # Accumulate buy and sell volumes
            if side == 'Buy':
                buy_volume += volume
            else:
                sell_volume += volume

        # Add the last window's data
        net_volumes.append(buy_volume - sell_volume)
        window_times.append(window_start.strftime('%H:%M:%S'))

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

        ax3 = self.fig.add_subplot(gs[2])  # 净成交量
        
        # 创建渐变色填充
        positive_volumes = np.array([max(0, v) for v in net_volumes])
        negative_volumes = np.array([min(0, v) for v in net_volumes])
        
        ax3.fill_between(range(len(window_times)), positive_volumes, 0, 
                        alpha=0.2, color='green', label='Buy > Sell')
        ax3.fill_between(range(len(window_times)), negative_volumes, 0, 
                        alpha=0.2, color='red', label='Sell > Buy')
        
        # 使用点的堆叠创建密集的线条效果
        for i in range(len(window_times)-1):
            # 在两个点之间创建多个插值点
            num_points = 20  # 插值点数量
            x = np.linspace(i, i+1, num_points)
            y = np.linspace(net_volumes[i], net_volumes[i+1], num_points)
            sizes = np.abs(y) / max(abs(np.array(net_volumes))) * 50 + 1  # 点大小随值变化
            
            # 用渐变色区分正负值
            colors = ['green' if val >= 0 else 'red' for val in y]
            ax3.scatter(x, y, c=colors, s=sizes, alpha=0.5, linewidth=0)

        # 标记最大最小值点
        max_idx = np.argmax(net_volumes)
        min_idx = np.argmin(net_volumes)
        ax3.scatter([max_idx], [net_volumes[max_idx]], color='green', s=150, zorder=5, marker='*')
        ax3.scatter([min_idx], [net_volumes[min_idx]], color='red', s=150, zorder=5, marker='*')

        # 其他设置保持不变
        ax3.set_xticks(window_xticks_pos)
        ax3.set_xticklabels(window_xticks_labels, rotation=45)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Net Volume')
        ax3.set_title(f'Net Volume Over Time (Window={time_window})')
        ax3.legend()
        ax3.grid(True)

        ax4 = self.fig.add_subplot(gs[3])  # 大单交易
        large_order_threshold = 200000  # 大于200,000的订单
        order_values = []  # 存储所有大单的交易金额
        buy_scatter = None
        sell_scatter = None
        has_buy = False
        has_sell = False
        
        for idx, (volume, price, side) in enumerate(zip(data['volumes'], data['prices'], data['sides'])):
            order_value = volume * price  # 计算实际交易金额
            if order_value >= large_order_threshold:
                order_values.append(order_value)  # 收集大单金额用于设置y轴范围
                if side == 'Buy':
                    scatter = ax4.scatter(idx, order_value, marker='^', color='red', s=100)
                    if not has_buy:
                        buy_scatter = scatter
                        has_buy = True
                else:
                    scatter = ax4.scatter(idx, order_value, marker='v', color='blue', s=100)
                    if not has_sell:
                        sell_scatter = scatter
                        has_sell = True

        # 设置合适的y轴范围和格式
        if order_values:  # 如果存在大单交易
            ax4.set_ylim([min(order_values) * 0.9, max(order_values) * 1.1])
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1000000:.1f}M' if x >= 1000000 else f'{x/1000:.0f}K'))
            
            # 构建图例
            legend_elements = []
            legend_labels = []
            if has_buy:
                legend_elements.append(buy_scatter)
                legend_labels.append('Buy')
            if has_sell:
                legend_elements.append(sell_scatter)
                legend_labels.append('Sell')
            if legend_elements:
                ax4.legend(legend_elements, legend_labels)

        # 移动到这里：设置标题和其他属性
        ax4.set_xticks(xticks_pos)
        ax4.set_xticklabels(xticks_labels, rotation=45)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Order Value (CNY)')
        ax4.set_title('Large Orders (>200,000)')  # 确保标题显示
        ax4.grid(True)

        # 计算累计净成交量
        cumulative_net_volume = []
        current_net = 0
        buy_total = 0
        sell_total = 0
        
        for volume, side in zip(data['volumes'], data['sides']):
            if side == 'Buy':
                buy_total += volume
            else:
                sell_total += volume
            current_net = buy_total - sell_total
            cumulative_net_volume.append(current_net)

        # 添加新的子图：累计净成交量
        ax5 = self.fig.add_subplot(gs[4])
        ax5.plot(range(len(readable_timestamps)), cumulative_net_volume, label='Cumulative Net Volume', color='purple')
        
        # 添加零线
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 填充正负区域
        ax5.fill_between(range(len(readable_timestamps)), cumulative_net_volume, 0,
                        where=[x >= 0 for x in cumulative_net_volume],
                        color='green', alpha=0.2, label='Net Buy')
        ax5.fill_between(range(len(readable_timestamps)), cumulative_net_volume, 0,
                        where=[x < 0 for x in cumulative_net_volume],
                        color='red', alpha=0.2, label='Net Sell')

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

        # 绘制图表2
        self.plot_order_heatmap()

    def plot_order_heatmap(self):
        """绘制订单热力图和成交量分布图"""
        self.fig2.clear()
        fig2 = self.fig2

        current_ticker = self.tickers[self.current_index]
        data = self.ticker_data[current_ticker]

        # 准备热力图数据
        timestamps = np.array(data['timestamps'])
        prices = np.array(data['prices'])
        volumes = np.array(data['volumes'])
        sides = np.array(data['sides'])

        # 定义时间和价格的区间
        num_bins = 20
        time_bins = np.linspace(timestamps.min(), timestamps.max(), num_bins + 1)
        price_bins = np.linspace(prices.min(), prices.max(), num_bins + 1)

        # 初始化热力图矩阵
        buy_heatmap = np.zeros((num_bins, num_bins))
        sell_heatmap = np.zeros((num_bins, num_bins))

        # 将数据分配到区间
        time_indices = np.digitize(timestamps, time_bins) - 1
        price_indices = np.digitize(prices, price_bins) - 1

        # 填充热力图矩阵
        for t_idx, p_idx, volume, side in zip(time_indices, price_indices, volumes, sides):
            if 0 <= t_idx < num_bins and 0 <= p_idx < num_bins:
                if side == 'Buy':
                    buy_heatmap[t_idx, p_idx] += volume
                else:
                    sell_heatmap[t_idx, p_idx] += volume

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

        for p_idx, volume, side in zip(price_indices, volumes, sides):
            if 0 <= p_idx < num_bins:
                if side == 'Buy':
                    buy_volumes_by_price[p_idx] += volume
                else:
                    sell_volumes_by_price[p_idx] += volume

        bar_width = 0.4
        indices = np.arange(num_bins)
        ax3.bar(indices - bar_width / 2, buy_volumes_by_price, width=bar_width, label='Buy', color='red', alpha=0.7)
        ax3.bar(indices + bar_width / 2, sell_volumes_by_price, width=bar_width, label='Sell', color='green', alpha=0.7)
        ax3.set_xticks(indices)
        ax3.set_xticklabels([f"{price_bins[i]:.2f}" for i in range(num_bins)], rotation=45, ha='right')
        ax3.set_xlabel('Price')
        ax3.set_ylabel('Volume')
        ax3.set_title('Volume Distribution by Price')
        ax3.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig2.suptitle(f'Order Analysis - {current_ticker}', fontsize=16)
        fig2.canvas.draw_idle()

    def on_key(self, event):
        """按键事件处理器"""
        old_index = self.current_index
        if event.key == 'left' and self.current_index > 0:
            self.current_index -= 1
        elif event.key == 'right' and self.current_index < len(self.tickers) - 1:
            self.current_index += 1

        if old_index != self.current_index:
            self.plot_current_ticker()

def plot_data(data, twap_results=None):
    """使用 TradingDataViewer 类显示数据"""
    viewer = TradingDataViewer(data, twap_results)
    viewer.fig.canvas.mpl_connect('key_press_event', viewer.on_key)
    viewer.fig2.canvas.mpl_connect('key_press_event', viewer.on_key)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制交易数据。')
    parser.add_argument('--file', type=str, default='./data/data.json', help='数据文件路径')
    args = parser.parse_args()

    data = load_data(args.file)
    plot_data(data)