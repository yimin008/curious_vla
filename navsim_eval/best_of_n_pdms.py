import pandas as pd
import sys
from typing import List

def _process_single_csv(file_path: str) -> pd.DataFrame:
    """
    (辅助函数)
    从一个 EPDMS 结果 CSV 文件中，计算每个有效场景的 PDMS 分数，
    并返回包含所有原始列和新 'pdms_score' 列的 DataFrame。

    Args:
        file_path (str): 指向 EPDMS 结果 CSV 文件的路径。

    Returns:
        pd.DataFrame: 包含所有原始数据和 'pdms_score' 列。
                      如果文件无效或没有有效场景，则返回空的 DataFrame。
    """
    try:
        epdms_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Skipping.", file=sys.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}. Skipping.", file=sys.stderr)
        return pd.DataFrame()

    # 2. 检查必需的列
    required_cols = [
        'valid', 'token', 'no_at_fault_collisions', 'drivable_area_compliance',
        'ego_progress', 'time_to_collision_within_bound', 'history_comfort'
    ]
    missing_cols = [col for col in required_cols if col not in epdms_df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns in '{file_path}': {', '.join(missing_cols)}. Skipping file.", file=sys.stderr)
        return pd.DataFrame()

    # 3. 筛选出有效的、非平均值的场景数据
    valid_scenarios_df = epdms_df[
        (epdms_df['valid'] == True) &
        # 排除所有非场景token的汇总行
        (~epdms_df['token'].str.contains('average|score', case=False, na=False))
    ].copy()

    if valid_scenarios_df.empty:
        print(f"Info: No valid scenario data found in '{file_path}'.")
        return pd.DataFrame()

    # 4. 定义 PDMS v1 的权重
    W_EP = 5.0
    W_TTC = 5.0
    W_HC = 2.0
    TOTAL_WEIGHT_PDMS = W_EP + W_TTC + W_HC

    # 5. 计算 PDMS v1 的两个核心部分
    multiplier_prod = (
        valid_scenarios_df['no_at_fault_collisions'] *
        valid_scenarios_df['drivable_area_compliance']
    )
    
    weighted_sum = (
        W_EP * valid_scenarios_df['ego_progress'] +
        W_TTC * valid_scenarios_df['time_to_collision_within_bound'] +
        W_HC * valid_scenarios_df['history_comfort']
    )
    
    weighted_avg = weighted_sum / TOTAL_WEIGHT_PDMS

    # 6. 计算每个场景的 PDMS 分数并添加到 DataFrame
    valid_scenarios_df['pdms_score'] = multiplier_prod * weighted_avg
    
    return valid_scenarios_df

def generate_best_of_n_report(file_paths: List[str], output_csv_path: str):
    """
    从多个 EPDMS 结果 CSV 文件中，计算 "best-of-n" PDMS 分数，
    并保存一个只包含每个 token 最佳结果的 CSV 报告。

    Args:
        file_paths (List[str]): 包含所有 EPDMS 结果 CSV 文件路径的列表。
        output_csv_path (str): 保存最终 "best-of-n" 报告的 CSV 文件路径。
    """
    if not file_paths:
        print("Error: No input file paths provided.", file=sys.stderr)
        return

    print(f"Processing {len(file_paths)} files for Best-of-N report...")

    # 1. 循环读取所有文件，计算每个场景的 PDMS，并收集到一个列表中
    all_scenario_dfs_list = []
    for file_path in file_paths:
        scenario_df = _process_single_csv(file_path)
        if not scenario_df.empty:
            all_scenario_dfs_list.append(scenario_df)

    if not all_scenario_dfs_list:
        print("Error: No valid data could be processed from any of the files.", file=sys.stderr)
        return

    # 2. 将所有 DataFrame 合并为一个
    combined_df = pd.concat(all_scenario_dfs_list, ignore_index=True)

    if 'token' not in combined_df.columns:
         print("Error: 'token' column not found in processed data.", file=sys.stderr)
         return

    print(f"Total valid scenarios found across all files: {len(combined_df)}")

    # 3. 找到每个 token 的 "best" 行 (PDMS 分数最高)
    #    - 按 'pdms_score' 降序排序
    #    - 按 'token' 去重，保留第一个 (即分数最高的)
    best_of_n_df = combined_df.sort_values(by='pdms_score', ascending=False)
    best_of_n_df = best_of_n_df.drop_duplicates(subset=['token'], keep='first')
    
    num_unique_tokens = len(best_of_n_df)
    if num_unique_tokens == 0:
        print("Error: No unique tokens found after processing.", file=sys.stderr)
        return

    # 4. 计算这些 "best" 分数的最终平均值
    final_best_of_n_pdms_score = best_of_n_df['pdms_score'].mean()
    
    # 增加一个average行
    average_row = best_of_n_df.mean(numeric_only=True)
    average_row['token'] = 'average_best_of_n'
    best_of_n_df = pd.concat([best_of_n_df, pd.DataFrame([average_row])], ignore_index=True)
    
    
    # 5. 保存 "best-of-n" 结果到 CSV 文件
    try:
        # 确保目录存在 (可选, 但建议)
        # import os
        # os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        best_of_n_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    except Exception as e:
        print(f"Error: Failed to save output CSV to '{output_csv_path}': {e}", file=sys.stderr)
        return

    # 6. 打印最终总结
    print("\n--- Best-of-N PDMS Report Generation Complete ---")
    print(f"Total files processed (N): {len(file_paths)}")
    print(f"Unique tokens/scenarios found: {num_unique_tokens}")
    print(f"Final Best-of-N PDMS Score: {final_best_of_n_pdms_score:.4f}")
    print(f"Best-of-N results saved to: {output_csv_path}")
    print("-------------------------------------------------")


# --- 脚本主入口 ---
if __name__ == "__main__":
    list_of_csv_files = []
    output_csv_path = ""
    

    generate_best_of_n_report(list_of_csv_files, output_csv_path)