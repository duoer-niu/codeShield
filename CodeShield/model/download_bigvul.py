"""
下载 Big-Vul C/C++ 漏洞数据集并保存为 CSV
数据集：https://huggingface.co/datasets/bstee615/bigvul

用法：
    pip3 install datasets
    python3 download_bigvul.py

输出：
    data/bigvul_raw/bigvul_train.csv
    data/bigvul_raw/bigvul_validation.csv
    data/bigvul_raw/bigvul_test.csv
"""

import os
import pandas as pd
from pathlib import Path


def download_via_datasets(output_dir: str):
    from datasets import load_dataset

    print("正在加载 Big-Vul 数据集（首次会下载缓存，约 250MB）...")
    dataset = load_dataset("bstee615/bigvul", trust_remote_code=True)

    os.makedirs(output_dir, exist_ok=True)

    for split_name in ["train", "validation", "test"]:
        print(f"正在转换 {split_name} 集...")
        df = dataset[split_name].to_pandas()
        output_path = os.path.join(output_dir, f"bigvul_{split_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"  ✓ {output_path} ({len(df)} 条)")

    print("\n数据集统计：")
    for split_name in ["train", "validation", "test"]:
        df = pd.read_csv(os.path.join(output_dir, f"bigvul_{split_name}.csv"))
        vul_count = df["vul"].sum()
        total = len(df)
        print(f"  {split_name:>10}: {total:>7} 条, 漏洞 {vul_count:>5} 条 ({vul_count/total*100:.1f}%)")


def convert_parquet_to_csv(parquet_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for pf in Path(parquet_dir).glob("*.parquet"):
        print(f"读取 {pf.name} ...")
        df = pd.read_parquet(str(pf))
        split_name = pf.stem.split("-")[0]
        csv_path = os.path.join(output_dir, f"bigvul_{split_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✓ {csv_path} ({len(df)} 条)")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "data", "bigvul_raw")
    parquet_files = list(Path(output_dir).glob("*.parquet"))

    if parquet_files:
        print("检测到本地 parquet 文件，直接转换...")
        convert_parquet_to_csv(output_dir, output_dir)
    else:
        try:
            download_via_datasets(output_dir)
        except ImportError:
            print("\n⚠ 请先安装：pip3 install datasets")
        except Exception as e:
            print(f"\n⚠ 下载失败: {e}")
            print("手动下载指引：访问 https://huggingface.co/datasets/bstee615/bigvul 下载 parquet 到 data/bigvul_raw/ 后重新运行")
