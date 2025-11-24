import argparse
import re
import pandas as pd
from pathlib import Path

def parse_h5ad_log(log_file):
    """解析h5ad日志文件，提取数据集信息"""
    datasets = {}
    current_dataset = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 检测新的数据集开始
        if line.startswith('📄 正在读取文件:'):
            filename = line.replace('📄 正在读取文件:', '').strip()
            dataset_id = filename.replace('.link.h5ad', '')
            current_dataset = {
                '数据名': filename,
                '细胞数目': None,
                '物种': None,
                '组织': None,
                '测序技术': None,
                '时序值': None,
                '时序单位': None
            }
            datasets[dataset_id] = current_dataset
        
        # 提取细胞数量
        elif current_dataset and '细胞数量 (obs):' in line:
            match = re.search(r'细胞数量 \(obs\): ([0-9,]+)', line)
            if match:
                current_dataset['细胞数目'] = match.group(1).replace(',', '')
        
        # 提取非结构化数据
        elif current_dataset and line.startswith('非结构化数据 (uns):'):
            i += 1
            while i < len(lines) and not lines[i].startswith('📄') and not lines[i].startswith('==='):
                uns_line = lines[i].strip()
                if 'Species: str =' in uns_line:
                    current_dataset['物种'] = uns_line.split('= ')[1] if '= ' in uns_line else uns_line.split('=')[1]
                elif 'Tissue: str =' in uns_line:
                    current_dataset['组织'] = uns_line.split('= ')[1] if '= ' in uns_line else uns_line.split('=')[1]
                elif 'Technology: str =' in uns_line:
                    current_dataset['测序技术'] = uns_line.split('= ')[1] if '= ' in uns_line else uns_line.split('=')[1]
                elif 'timepoints: str =' in uns_line:
                    current_dataset['时序值'] = uns_line.split('= ')[1] if '= ' in uns_line else uns_line.split('=')[1]
                elif 'time_unit: str =' in uns_line:
                    current_dataset['时序单位'] = uns_line.split('= ')[1] if '= ' in uns_line else uns_line.split('=')[1]
                i += 1
            continue
        
        i += 1
    
    return datasets

def parse_csv_table(csv_file):
    """解析CSV表格文件"""
    df = pd.read_csv(csv_file)
    # 清理数据，移除空行
    df = df.dropna(how='all')
    
    datasets = {}
    for _, row in df.iterrows():
        dataset_id = row['ID']
        datasets[dataset_id] = {
            '物种': row['Species'],
            '组织': row['Tissue'],
            '测序技术': row['Sequencing'],
            '时序值': row['sorted_time'],
            '时序单位': row['time_unit'],
            'Cell': row['Cell']
        }
    return datasets

def main():
    parser = argparse.ArgumentParser(description='提取单细胞数据集信息')
    parser.add_argument('log_file', help='view_h5ad.log.txt文件路径')
    parser.add_argument('csv_file', help='tedd_datasets_table_processed.csv文件路径')
    parser.add_argument('storage_path', help='数据存储路径')
    parser.add_argument('data_source', help='数据来源', default='TEDD')
    
    args = parser.parse_args()
    
    # 解析文件
    log_datasets = parse_h5ad_log(args.log_file)
    csv_datasets = parse_csv_table(args.csv_file)
    
    # 合并数据，优先使用CSV中的数据
    results = []
    
    for dataset_id, log_info in log_datasets.items():
        result = {
            '数据名': log_info['数据名'],
            '存储路径': args.storage_path,
            '数据类型': 'h5ad',
            '数据来源': args.data_source,
            '数据用途': '单细胞转录组分析',
            '存储格式': 'h5ad',
            '细胞数目': log_info['细胞数目'],
            '物种': None,
            '组织': None,
            '测序技术': None,
            '健康/疾病': '健康',  # 默认为健康，可根据需要调整
            '是否扰动': '否',     # 默认为否，可根据需要调整
            '扰动类型': '',
            '扰动数': '0',
            '是否含时序信息': '是',
            '采样点数目': None,
            '时序值': None,
            '时序单位': None
        }
        
        # 优先使用CSV中的数据，如果不存在则使用log中的数据
        if dataset_id in csv_datasets:
            csv_info = csv_datasets[dataset_id]
            result['物种'] = csv_info['物种']
            result['组织'] = csv_info['组织']
            result['测序技术'] = csv_info['测序技术']
            result['时序值'] = csv_info['时序值']
            result['时序单位'] = csv_info['时序单位']
            # 如果log中没有细胞数目，尝试使用CSV中的Cell列
            if not result['细胞数目'] and 'Cell' in csv_info:
                result['细胞数目'] = str(csv_info['Cell'])
        else:
            # 使用log中的数据
            result['物种'] = log_info['物种']
            result['组织'] = log_info['组织']
            result['测序技术'] = log_info['测序技术']
            result['时序值'] = log_info['时序值']
            result['时序单位'] = log_info['时序单位']
        
        # 计算采样点数目
        if result['时序值']:
            timepoints = str(result['时序值']).split(',')
            result['采样点数目'] = str(len(timepoints))
        
        # 如果没有时序信息，则更新相关字段
        if not result['时序值']:
            result['是否含时序信息'] = '否'
            result['采样点数目'] = '0'
            result['时序值'] = ''
            result['时序单位'] = ''
        
        results.append(result)
    
    # 创建结果DataFrame
    df_output = pd.DataFrame(results)
    
    # 重新排列列顺序
    columns_order = [
        '数据名', '存储路径', '数据类型', '数据来源', '数据用途', '存储格式',
        '细胞数目', '物种', '组织', '测序技术', '健康/疾病', '是否扰动',
        '扰动类型', '扰动数', '是否含时序信息', '采样点数目', '时序值', '时序单位'
    ]
    df_output = df_output[columns_order]
    
    # 保存结果
    output_file = 'dataset_info_summary.csv'
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {output_file}")
    
    # 打印前几行预览
    print("\n前5行数据预览:")
    print(df_output.head().to_string(index=False))

if __name__ == '__main__':
    main()
