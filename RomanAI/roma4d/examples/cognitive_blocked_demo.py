#!/usr/bin/env python3
"""Blocked Kinetic loop for Phase 3 GGUF cognitive tests.

Single-statement body contains a dict literal — static analyzer: forbidden_ast:Dict.
Run:  r4d --explain examples/cognitive_blocked_demo.py
With a valid [llm] model_path + llama-cpp-python, expect [4D-AI-SUGGESTION] blocks.
"""


def main() -> None:
    total = 0.0
    n = 50
    for i in range(n):
        total += float({"k": i}["k"]) * 1.5
    print(f"cognitive_blocked_demo: total={total:.2f}")


if __name__ == "__main__":
    main()
