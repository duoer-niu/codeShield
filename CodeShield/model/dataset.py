"""
Big-Vul 数据处理：CSV → token 化 → 词汇表 → PyTorch Dataset

Chain GCN 将代码视为 token 序列（链式图），每个 token 是一个节点，
相邻 token 之间有边。用 padding 统一长度。

用法：
    python dataset.py              # 预览数据
    python dataset.py --build      # 构建词汇表并保存
"""

import re
import json
import os
import argparse
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


# ============================================================
# 1. 代码 Token 化
# ============================================================

def tokenize_code(code: str) -> list:
    """
    将 C/C++ 源代码切分为 token 序列。
    策略：正则匹配各类 token，保持语义完整性。
    """
    if not isinstance(code, str) or len(code.strip()) == 0:
        return []

    patterns = [
        (r'"(?:[^"\\]|\\.)*"', 'STR'),
        (r"'(?:[^'\\]|\\.)'", 'CHAR'),
        (r'\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?[fFlLuU]*)\b', 'NUM'),
        (r'[A-Za-z_]\w*', 'ID'),
        (r'->|\.\.\.|>>=|<<=|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--'
         r'|\+=|-=|\*=|/=|%=|&=|\|=|\^=', 'OP'),
        (r'[{}()\[\];,:.#]', 'SEP'),
        (r'[+\-*/%&|^~!<>=?@\\]', 'OP'),
        (r'\s+', None),
        (r'//[^\n]*', None),
        (r'/\*.*?\*/', None),
    ]

    tokens = []
    pos = 0
    while pos < len(code):
        best = None
        best_len = 0
        for pat, tag in patterns:
            m = re.match(pat, code[pos:])
            if m and len(m.group()) > best_len:
                best = (m.group(), tag)
                best_len = len(m.group())
        if best:
            if best[1] is not None:
                tokens.append(best[0])
            pos += best_len
        else:
            pos += 1
    return tokens


# ============================================================
# 2. 词汇表
# ============================================================

class Vocabulary:
    """token ↔ id 映射，预留特殊标记。"""

    SPECIALS = ["<PAD>", "<UNK>"]

    def __init__(self):
        self.token2id = {}
        self.id2token = {}

    def build(self, token_lists: list, min_freq: int = 2, max_vocab: int = 10000):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        vocab = [t for t, c in counter.items() if c >= min_freq]
        vocab = sorted(vocab, key=counter.get, reverse=True)
        vocab = self.SPECIALS + vocab[:max_vocab]

        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}

        coverage = sum(counter.get(t, 0) for t in vocab) / sum(counter.values()) * 100
        print(f"词汇表: {len(self)} 词, 覆盖率 {coverage:.1f}%")

    def __len__(self):
        return len(self.token2id)

    def encode(self, tokens: list, max_len: int) -> list:
        ids = [self.token2id.get(t, self.token2id["<UNK>"]) for t in tokens]
        ids = ids[:max_len]
        ids = ids + [self.token2id["<PAD>"]] * (max_len - len(ids))
        return ids

    def save(self, path: str, max_len: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"token2id": self.token2id, "max_len": max_len}
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"已保存: {path}")

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        v = cls()
        v.token2id = data["token2id"]
        v.id2token = {int(i): t for t, i in v.token2id.items()}
        return v


# ============================================================
# 3. Dataset
# ============================================================

class VulnDataset(Dataset):
    """
    漏洞检测数据集。
    返回: input_ids (max_len,), mask (max_len,), label (scalar)
    """

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = 256):
        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            tokens = tokenize_code(row["func_before"])
            if not tokens:
                skipped += 1
                continue

            ids = vocab.encode(tokens, max_len)
            valid_len = min(len(tokens), max_len)
            mask = [1] * valid_len + [0] * (max_len - valid_len)

            self.samples.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.float32),
                "label": torch.tensor(int(row["vul"]), dtype=torch.float32),
            })

        if skipped:
            print(f"  跳过 {skipped} 条空代码")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "mask": torch.stack([s["mask"] for s in batch]),
        "label": torch.stack([s["label"] for s in batch]),
    }


# ============================================================
# 4. 构建词汇表
# ============================================================

def export_vocab(csv_dir: str, vocab_path: str, max_len: int = 256, vocab_size: int = 10000):
    csv_file = os.path.join(csv_dir, "bigvul_train.csv")
    if not os.path.exists(csv_file):
        print(f"未找到 {csv_file}，请先运行 download_bigvul.py")
        return None

    print(f"读取: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  共 {len(df)} 条")

    all_tokens = [tokenize_code(code) for code in df["func_before"]]

    lens = [len(t) for t in all_tokens]
    print(f"Token 长度: avg={np.mean(lens):.0f}  p50={np.median(lens):.0f}  p90={np.percentile(lens,90):.0f}  p95={np.percentile(lens,95):.0f}  max={max(lens)}")

    vocab = Vocabulary()
    vocab.build(all_tokens, min_freq=2, max_vocab=vocab_size)
    vocab.save(vocab_path, max_len)

    return vocab


# ============================================================
# 5. 预览
# ============================================================

def preview(csv_dir: str, n: int = 3):
    csv_file = os.path.join(csv_dir, "bigvul_train.csv")
    if not os.path.exists(csv_file):
        print(f"未找到数据，请先运行 download_bigvul.py")
        return

    df = pd.read_csv(csv_file)
    print(f"训练集: {len(df)} 条, 漏洞占比 {df['vul'].mean()*100:.1f}%")
    print(f"CWE Top5:\n{df['CWE_ID'].value_counts().head(5)}\n")

    for i in range(n):
        code = df.iloc[i]["func_before"]
        toks = tokenize_code(code)
        print(f"--- 样本 {i+1} | vul={df.iloc[i]['vul']} | tokens={len(toks)} ---")
        print(code[:150])
        print(f"Tokens: {toks[:15]}\n")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="构建词汇表")
    parser.add_argument("--csv_dir", default="data/bigvul_raw")
    parser.add_argument("--vocab_path", default="checkpoints/vocab.json")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    args.csv_dir = os.path.join(base, args.csv_dir)
    args.vocab_path = os.path.join(base, args.vocab_path)

    if args.build:
        export_vocab(args.csv_dir, args.vocab_path, args.max_len, args.vocab_size)
    else:
        preview(args.csv_dir)
