# 数据合并函数
def merge_market_data_files(data_dir="./data", output_dir="./merged_data"):
    """
    合并市场数据文件，按照 sym0-9 分别合并成 10 个 CSV 文件
    """
    import pandas as pd
    import glob
    import os

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有市场数据文件
    am_files = glob.glob(os.path.join(data_dir, "*_am.csv"))
    pm_files = glob.glob(os.path.join(data_dir, "*_pm.csv"))
    all_files = am_files + pm_files

    print(f"找到 {len(all_files)} 个市场数据文件")
    print(f"  上午文件: {len(am_files)} 个")
    print(f"  下午文件: {len(pm_files)} 个")

    if not all_files:
        print("没有找到市场数据文件")
        return None

    # 创建字典来存储每个 sym 的数据
    sym_data_dict = {f"svm{i}": [] for i in range(10)}

    # 读取所有文件并按 sym 分组
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)

            # 从文件名提取信息
            filename = os.path.basename(file_path)
            # 解析文件名：snapshot_sym<xx>_date<yy>_am/pm.csv
            parts = filename.replace("snapshot_", "").replace(".csv", "").split("_")

            if len(parts) >= 2:
                sym = parts[0].replace("sym", "")
                date = parts[1].replace("date", "")
                session = parts[2] if len(parts) > 2 else "unknown"

                df["file_sym"] = sym
                df["file_date"] = date
                df["file_session"] = session

            df["source_file"] = filename

            # 根据 sym 将数据添加到对应的列表中
            if sym in sym_data_dict:
                sym_data_dict[sym].append(df)
            else:
                # 如果发现不在 0-9 范围内的 sym，创建一个新的键
                if sym not in sym_data_dict:
                    sym_data_dict[sym] = []
                sym_data_dict[sym].append(df)

        except Exception as e:
            print(f"读取失败 {os.path.basename(file_path)}: {e}")

    # 合并每个 sym 的数据并保存
    merged_files = []

    for sym, df_list in sym_data_dict.items():
        if df_list:  # 只处理有数据的 sym
            # 合并该 sym 的所有数据
            merged_df = pd.concat(df_list, ignore_index=True)

            # 按时间和标的排序
            if (
                "date" in merged_df.columns
                and "time" in merged_df.columns
                and "sym" in merged_df.columns
            ):
                merged_df = merged_df.sort_values(["date", "time"])

            # 保存合并结果
            output_file = os.path.join(output_dir, f"merged_{sym}.csv")
            merged_df.to_csv(output_file, index=False)

            merged_files.append(
                {
                    "sym": sym,
                    "file_path": output_file,
                    "row_count": len(merged_df),
                    "file_count": len(df_list),
                }
            )

            print(f"{sym}: 合并了 {len(df_list)} 个文件，共 {len(merged_df):,} 行数据")

    # 打印汇总信息
    if merged_files:
        print(f"\n{'='*50}")
        print("市场数据合并完成!")
        print(f"总共生成了 {len(merged_files)} 个文件")
        print(f"输出目录: {output_dir}")

        total_rows = sum(f["row_count"] for f in merged_files)
        total_files = sum(f["file_count"] for f in merged_files)

        print(f"\n   汇总统计:")
        print(f"   总数据行数: {total_rows:,}")
        print(f"   总文件数: {total_files}")

        print(f"\n   详细输出文件:")
        for file_info in merged_files:
            print(
                f"     {file_info['sym']}: {file_info['row_count']:,} 行 ({file_info['file_count']} 个源文件)"
            )

    return merged_files


# 可选：提供一个函数将所有 sym 的数据合并到一个字典中，便于后续处理
def get_merged_data_dict(merged_files):
    """
    读取已合并的文件，返回一个字典，键为 sym，值为 DataFrame
    """
    import pandas as pd

    data_dict = {}
    for file_info in merged_files:
        sym = file_info["sym"]
        file_path = file_info["file_path"]
        data_dict[sym] = pd.read_csv(file_path)

    return data_dict


# 运行
if __name__ == "__main__":
    # 合并数据并生成10个文件
    merged_files = merge_market_data_files("./FBDQA2021A_MMP_Challenge/data/")

    # 如果需要将数据加载到内存中以供进一步处理
    if merged_files:
        print("\n可选：将合并的数据加载到内存中...")
        data_dict = get_merged_data_dict(merged_files)
        print(f"已加载 {len(data_dict)} 个 sym 的数据到内存中")
