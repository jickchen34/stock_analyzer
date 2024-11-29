import re
import json
import os
import argparse
from pathlib import Path

def parse_log_to_json(log_file_path, output_json_path):
    transactions = []
    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                # 匹配包含 [on_transaction] 的行
                if '[on_transaction]' in line:
                    # 提取 JSON 字符串部分
                    match = re.search(r"Recorded transaction: ({.*?})", line)
                    if match:
                        try:
                            # 解析交易数据
                            transaction_data = eval(match.group(1))
                            # 添加时间戳
                            timestamp = re.search(r"\[(\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]", line)
                            if timestamp:
                                transaction_data['log_time'] = timestamp.group(1)
                            transactions.append(transaction_data)
                        except Exception as e:
                            print(f"Error parsing transaction data: {e}")
                            continue

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        # 保存为 JSON 文件
        with open(output_json_path, 'w') as json_file:
            json.dump(transactions, json_file, indent=4)
        print(f"Successfully parsed {len(transactions)} transactions to {output_json_path}")
    except Exception as e:
        print(f"Error processing log file: {e}")

# 设置命令行参数
parser = argparse.ArgumentParser(description='Convert log file to JSON')
parser.add_argument('--file', type=str, required=True, help='Path to the log file')

args = parser.parse_args()

# 获取输入文件的路径
log_file_path = args.file

# 从输入文件路径中提取文件名（不包含扩展名）
input_filename = Path(log_file_path).stem

# 构建输出JSON文件的路径
output_json_path = f'./json/{input_filename}.json'

# 运行解析
parse_log_to_json(log_file_path, output_json_path)