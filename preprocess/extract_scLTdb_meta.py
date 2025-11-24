import pandas as pd
import os
import sys
import re
from typing import Dict, List, Any

def extract_dataset_info(csv_file: str, log_file: str, storage_path: str, data_source: str) -> pd.DataFrame:
    """
    从CSV和日志文件中提取数据集信息
    
    Args:
        csv_file: CSV文件路径
        log_file: 日志文件路径
        storage_path: 存储路径
        data_source: 数据来源
        
    Returns:
        包含提取信息的DataFrame
    """
    
    # 读取CSV文件
    csv_data = pd.read_csv(csv_file)
    
    # 解析日志文件提取数据名和细胞数目
    log_info = parse_log_file(log_file)
    
    # 准备结果列表
    results = []
    
    # 处理每个在日志中出现的数据库
    for dataset_file_name, cell_count in log_info.items():
        # 从文件名中提取匹配CSV Dataset列的部分
        dataset_match_name = dataset_file_name.replace('.link.h5ad', '')
        
        # 在CSV的Dataset列中查找匹配项
        matched_datasets = csv_data[csv_data['Dataset'] == dataset_match_name]
        
        if not matched_datasets.empty:
            # 取第一个匹配的数据集
            dataset_row = matched_datasets.iloc[0]
            
            # 提取时序信息
            time_info = extract_time_info(dataset_file_name, log_file)
            
            # 构建结果字典
            result = {
                '数据名': dataset_file_name,
                '存储路径': storage_path,
                '数据类型': 'h5ad',
                '数据来源': data_source,
                '数据用途': '细胞谱系追踪',
                '存储格式': 'h5ad',
                '细胞数目': cell_count,
                '物种': dataset_row['Species'],
                '组织': dataset_row['Tissue source'],
                '测序技术': dataset_row['Technology'],
                '健康/疾病': determine_health_status(dataset_row['Tissue source']),
                '是否扰动': determine_perturbation(dataset_row['Dataset']),
                '扰动类型': determine_perturbation_type(dataset_row['Dataset']),
                '扰动数': '待补充',
                '是否含时序信息': time_info['has_time'],
                '采样点数目': time_info['time_points'],
                '时序值': time_info['time_values'],
                '时序单位': time_info['time_unit']
            }
            
            results.append(result)
        else:
            print(f"警告: 在CSV的Dataset列中未找到 {dataset_match_name} 的匹配项")
    
    return pd.DataFrame(results)

def parse_log_file(log_file: str) -> Dict[str, int]:
    """
    解析日志文件，提取数据名和细胞数目
    """
    log_info = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配数据名和细胞数目
    pattern = r'📄 正在读取文件: (.*?\.link\.h5ad)[\s\S]*?细胞数量 \(obs\): ([\d,]+)'
    matches = re.findall(pattern, content)
    
    for dataset_name, cell_count_str in matches:
        cell_count = int(cell_count_str.replace(',', ''))
        log_info[dataset_name] = cell_count
    
    return log_info

def extract_time_info(dataset_name: str, log_file: str) -> Dict[str, Any]:
    """
    从日志文件中提取时序信息
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取特定数据集的信息块
    dataset_pattern = rf'📄 正在读取文件: {re.escape(dataset_name)}[\s\S]*?(?=📄 正在读取文件:|$)'
    dataset_match = re.search(dataset_pattern, content)
    
    time_info = {
        'has_time': '否',
        'time_points': 0,
        'time_values': '',
        'time_unit': ''
    }
    
    if dataset_match:
        dataset_content = dataset_match.group(0)
        
        # 查找time列的信息
        time_pattern = r"'time'[\s\S]*?唯一值数目: (\d+)[\s\S]*?唯一值: (\[.*?\])"
        time_match = re.search(time_pattern, dataset_content)
        
        if time_match:
            time_points = int(time_match.group(1))
            time_values_str = time_match.group(2)
            
            if time_points > 1:
                time_info['has_time'] = '是'
                time_info['time_points'] = time_points
                
                # 提取并排序时间值
                time_values = eval(time_values_str)  # 将字符串转换为列表
                sorted_time_values = sorted(time_values, key=lambda x: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.', '').isdigit() else x)
                time_info['time_values'] = ', '.join(map(str, sorted_time_values))
                
                # 推断时间单位
                time_info['time_unit'] = infer_time_unit_from_log(dataset_content)
    
    return time_info

def infer_time_unit_from_log(dataset_content: str) -> str:
    """
    从日志内容中推断时间单位
    """
    # 查找可能包含时间单位的列名
    time_related_columns = re.findall(r"'(.*?(?:time|day|hour|week|month|year|sample|point).*?)'", dataset_content, re.IGNORECASE)
    
    for col in time_related_columns:
        col_lower = col.lower()
        if 'day' in col_lower or any(re.search(r'\bD\d', col) for col in time_related_columns):
            return '天'
        elif 'hour' in col_lower or 'hr' in col_lower:
            return '小时'
        elif 'week' in col_lower:
            return '周'
        elif 'month' in col_lower:
            return '月'
        elif 'year' in col_lower:
            return '年'
        elif 'minute' in col_lower or 'min' in col_lower:
            return '分钟'
    
    # 如果从列名无法推断，检查时间值
    time_values_pattern = r"唯一值: (\[.*?\])"
    time_values_match = re.search(time_values_pattern, dataset_content)
    if time_values_match:
        time_values_str = time_values_match.group(1)
        if any(unit in time_values_str.lower() for unit in ['day', 'd']):
            return '天'
        elif any(unit in time_values_str.lower() for unit in ['hour', 'h']):
            return '小时'
        elif any(unit in time_values_str.lower() for unit in ['week', 'w']):
            return '周'
        elif any(unit in time_values_str.lower() for unit in ['month', 'm']):
            return '月'
    
    return '未知'

def determine_health_status(tissue_source: str) -> str:
    """
    根据组织来源判断健康/疾病状态
    """
    if 'Tumor' in tissue_source:
        return '疾病'
    elif 'Organoid' in tissue_source:
        return '体外模型(类器官)'
    elif 'Cell Line' in tissue_source:
        return '体外模型(细胞系)'
    elif 'Bone Marrow' in tissue_source or 'Hematopoietic' in tissue_source:
        return '健康'
    else:
        return '待确认'

def determine_perturbation(dataset_name: str) -> str:
    """
    根据数据集名称判断是否扰动
    """
    perturbation_keywords = ['pertur', 'treatment', 'drug', '5FU', 'TRAIL', 'RAS']
    if any(keyword.lower() in dataset_name.lower() for keyword in perturbation_keywords):
        return '是'
    else:
        return '否'

def determine_perturbation_type(dataset_name: str) -> str:
    """
    根据数据集名称判断扰动类型
    """
    dataset_lower = dataset_name.lower()
    if '5fu' in dataset_lower:
        return '化疗药物'
    elif 'trail' in dataset_lower:
        return '凋亡诱导'
    elif 'ras' in dataset_lower:
        return '基因突变'
    elif 'pertur' in dataset_lower:
        return '扰动实验'
    else:
        return '无'

def main():
    """
    主函数
    """
    # 从命令行参数获取路径和来源
    if len(sys.argv) < 4:
        print("用法: python script.py <CSV文件路径> <日志文件路径> <存储路径> <数据来源>")
        print("示例: python script.py scLTdb_Homo.csv view_h5ad.log.txt /path/to/data scLTdb")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    log_file = sys.argv[2]
    storage_path = sys.argv[3]
    data_source = sys.argv[4] if len(sys.argv) > 4 else "scLTdb"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件 {csv_file} 不存在")
        sys.exit(1)
    
    if not os.path.exists(log_file):
        print(f"错误: 日志文件 {log_file} 不存在")
        sys.exit(1)
    
    try:
        # 提取信息
        df = extract_dataset_info(csv_file, log_file, storage_path, data_source)
        
        if df.empty:
            print("未找到匹配的数据集信息")
            return
        
        # 输出结果
        print("\n提取的数据集信息:")
        print("=" * 120)
        print(df.to_string(index=False))
        
        # 保存到文件
        output_file = "dataset_info_extracted.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"处理的数据集数量: {len(df)}")
        print(f"总细胞数目: {df['细胞数目'].sum():,}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
