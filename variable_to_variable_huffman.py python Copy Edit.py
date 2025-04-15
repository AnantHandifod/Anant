import heapq
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def estimate_mgram_frequencies(data: str, m: int, alpha: float = 0) -> Dict[str, float]:
    freq = defaultdict(float)
    n = len(data)
    for size in range(1, m+1):
        for i in range(n - size + 1):
            mgram = data[i:i+size]
            freq[mgram] += size ** alpha
    return freq


def build_huffman_tree(frequencies: Dict[str, float]) -> Dict[str, str]:
    heap = [Node(symbol, freq) for symbol, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        parent = Node(None, l.freq + r.freq)
        parent.left = l
        parent.right = r
        heapq.heappush(heap, parent)

    codebook = {}
    
    def assign_codes(node, code=""):
        if node:
            if node.symbol:
                codebook[node.symbol] = code
            assign_codes(node.left, code + "0")
            assign_codes(node.right, code + "1")
    
    assign_codes(heap[0])
    return codebook


def optimal_encoding(data: str, codebook: Dict[str, str], m: int) -> str:
    n = len(data)
    dp = [("", 0)] + [("", float('inf'))] * n
    
    for i in range(1, n+1):
        for size in range(1, min(m, i) + 1):
            mgram = data[i-size:i]
            if mgram in codebook:
                candidate = dp[i-size][0] + codebook[mgram]
                if len(candidate) < len(dp[i][0]) or dp[i][0] == "":
                    dp[i] = (candidate, len(candidate))
    return dp[n][0]


def greedy_encoding(data: str, codebook: Dict[str, str], m: int) -> str:
    n = len(data)
    result = ""
    i = 0
    while i < n:
        best_ratio, best_mgram = 0, ""
        for size in range(1, m+1):
            if i+size <= n:
                mgram = data[i:i+size]
                if mgram in codebook:
                    ratio = size / len(codebook[mgram])
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_mgram = mgram
        result += codebook[best_mgram]
        i += len(best_mgram)
    return result


def decode(encoded: str, codebook: Dict[str, str]) -> str:
    reverse_codebook = {v: k for k, v in codebook.items()}
    result, buffer = "", ""
    for bit in encoded:
        buffer += bit
        if buffer in reverse_codebook:
            result += reverse_codebook[buffer]
            buffer = ""
    return result


# === Example Run ===

data = "aaaaaaab"
m = 3
alpha = 1.0

# 1. Estimate frequencies
frequencies = estimate_mgram_frequencies(data, m, alpha)

# 2. Build Huffman tree
codebook = build_huffman_tree(frequencies)

# 3. Encode using optimal and greedy
optimal_result = optimal_encoding(data, codebook, m)
greedy_result = greedy_encoding(data, codebook, m)

# 4. Decode
decoded = decode(optimal_result, codebook)

print("Original: ", data)
print("Codebook: ", codebook)
print("Optimal Encoded: ", optimal_result)
print("Greedy Encoded: ", greedy_result)
print("Decoded: ", decoded)
