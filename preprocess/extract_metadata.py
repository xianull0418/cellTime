import pandas as pd
import re
import sys
import os
from datetime import datetime

def read_tsv_file(tsv_file):
    """读取TSV文件"""
    try:
        return pd.read_csv(tsv_file, sep='\t')
    except Exception as e:
        print(f"读取TSV文件失败: {e}")
        return None

def extract_data_id_from_filename(filename):
    """从文件名中提取数据ID"""
    # 移除.link.h5ad后缀
    base_name = filename.replace('.link.h5ad', '')
    
    # 如果文件名中有点，取第一个点之前的部分作为数据ID
    if '.' in base_name:
        return base_name.split('.')[0]
    else:
        return base_name

def extract_detailed_info(log_content, filename):
    """从日志内容中提取详细信息"""
    file_pattern = rf'📄 正在读取文件: {re.escape(filename)}(.*?)(?=📄 正在读取文件:|$)'
    match = re.search(file_pattern, log_content, re.DOTALL)
    
    if not match:
        return {}
    
    file_content = match.group(1)
    info = {}
    
    # 提取细胞数量
    cell_match = re.search(r'细胞数量 \(obs\): ([\d,]+)', file_content)
    if cell_match:
        info['cell_count'] = cell_match.group(1).replace(',', '')
    
    # 提取time列信息
    time_pattern = r"'time':.*?唯一值数目: (\d+).*?唯一值: (\[.*?\])"
    time_match = re.search(time_pattern, file_content, re.DOTALL)
    if time_match:
        sampling_points = time_match.group(1)
        time_values_str = time_match.group(2)
        
        # 解析时间值
        try:
            time_values = [x.strip() for x in time_values_str[1:-1].split(',')]
            time_values = [x for x in time_values if x]
            time_values = sorted(set(time_values))
            info['sampling_points'] = sampling_points
            info['time_values'] = time_values
            info['has_temporal'] = "是" if len(time_values) > 1 else "否"
        except Exception as e:
            print(f"解析时间值时出错: {e}")
    
    # 提取gene列信息
    gene_pattern = r"'gene':.*?唯一值: (\[.*?\])"
    gene_match = re.search(gene_pattern, file_content, re.DOTALL)
    if gene_match:
        gene_values_str = gene_match.group(1)
        try:
            gene_values = [x.strip().strip("'") for x in gene_values_str[1:-1].split(',')]
            info['gene_values'] = gene_values
            
            # 推测时间单位
            gene_str = ' '.join(gene_values).lower()
            time_units = {
                'hour': ['hour', 'hr', '小时'],
                'day': ['day', '天'],
                'week': ['week', 'wk', '周'],
                'month': ['month', 'mon', '月'],
                'year': ['year', 'yr', '年']
            }
            
            found_unit = None
            for unit, keywords in time_units.items():
                for keyword in keywords:
                    if keyword in gene_str:
                        found_unit = unit
                        break
                if found_unit:
                    break
            
            if not found_unit:
                # 基于常见模式推测
                if any(x in gene_str for x in ['erlotinib', 'ctla4', 'pd1', 'drug', 'treatment', 'therapy']):
                    found_unit = 'day'  # 药物处理通常以天为单位
                elif any(x in gene_str for x in ['development', 'differentiation', 'maturation']):
                    found_unit = 'day'  # 发育过程通常以天为单位
            
            info['time_unit'] = found_unit or "N/A"
            
        except Exception as e:
            print(f"解析gene值时出错: {e}")
    
    return info

def main():
    if len(sys.argv) < 6:
        print("使用方法: python script.py <日志文件> <TSV文件> <存储路径> <数据来源> <扰动类型>")
        print("示例: python script.py view_h5ad.log.txt PerturBase.repository.tsv /data/h5ad PerturBase DrugTreatment")
        sys.exit(1)
    
    log_file = sys.argv[1]
    tsv_file = sys.argv[2]
    storage_path = sys.argv[3]
    data_source = sys.argv[4]
    perturbation_type = sys.argv[5]
    
    # 读取文件
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        tsv_df = read_tsv_file(tsv_file)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    if tsv_df is None:
        print("无法读取TSV文件，程序退出")
        return
    
    # 提取h5ad文件列表
    h5ad_files = re.findall(r'📄 正在读取文件: (.*?\.link\.h5ad)', log_content)
    
    results = []
    
    for h5ad_file in h5ad_files:
        # 从文件名中提取数据ID
        data_id = extract_data_id_from_filename(h5ad_file)
        print(f"处理文件: {h5ad_file}, 提取的数据ID: {data_id}")
        
        # 从TSV中匹配
        matched_row = tsv_df[tsv_df['Data Index'] == data_id]
        
        if matched_row.empty:
            print(f"警告: 未找到数据ID '{data_id}' 的匹配记录")
            continue
        
        row = matched_row.iloc[0]
        
        # 从日志中提取详细信息
        detailed_info = extract_detailed_info(log_content, h5ad_file)
        
        # 构建结果记录
        record = {
            '数据名': h5ad_file,
            '存储路径': storage_path,
            '数据类型': 'h5ad',
            '数据来源': data_source,
            '数据用途': row.get('Title', 'N/A'),
            '存储格式': 'h5ad',
            '细胞数目': detailed_info.get('cell_count', 'N/A'),
            '物种': row.get('Organisms', 'N/A'),
            '组织': row.get('Model Description', 'N/A'),
            '测序技术': row.get('Modality', 'N/A'),
            '健康/疾病': 'N/A',
            '是否扰动': '是',
            '扰动类型': perturbation_type,
            '扰动数': '1',
            '是否含时序信息': detailed_info.get('has_temporal', '否'),
            '采样点数目': detailed_info.get('sampling_points', 'N/A'),
            '时序值': ','.join(detailed_info.get('time_values', [])) if detailed_info.get('time_values') else 'N/A',
            '时序单位': detailed_info.get('time_unit', 'N/A')
        }
        
        results.append(record)
    
    # 输出到文件和屏幕
    if results:
        output_df = pd.DataFrame(results)
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"extracted_metadata_{timestamp}.csv"
        
        # 保存到CSV
        output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"元数据已保存到: {output_file}")
        
        # 打印到屏幕（制表符分隔）
        print("\n数据名\t存储路径\t数据类型\t数据来源\t数据用途\t存储格式\t细胞数目\t物种\t组织\t测序技术\t健康/疾病\t是否扰动\t扰动类型\t扰动数\t是否含时序信息\t采样点数目\t时序值\t时序单位")
        for record in results:
            line = "\t".join([str(record[key]) for key in [
                '数据名', '存储路径', '数据类型', '数据来源', '数据用途', '存储格式',
                '细胞数目', '物种', '组织', '测序技术', '健康/疾病', '是否扰动',
                '扰动类型', '扰动数', '是否含时序信息', '采样点数目', '时序值', '时序单位'
            ]])
            print(line)
    else:
        print("未提取到任何元数据")

if __name__ == "__main__":
    main()
