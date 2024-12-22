import argparse
import ast
import json
import logging
from pathlib import Path

# Constants
TARGET_VALUE = 24

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_name",
    type=str,
    default="gameof24",
    choices=["gameof24", "checkmate", "wordsorting"],
)
parser.add_argument("--test_path", type=str)
if __name__ == "__main__":
    args = parser.parse_args()
    task = args.task_name
    test_path = args.test_path
    benchmark_path_dict = {
        "gameof24": "benchmarks/gameof24.jsonl",
        "checkmate": "benchmarks/CheckmateInOne.jsonl",
        "wordsorting": "benchmarks/word_sorting.jsonl",
    }
    test_path_dict = {
        "gameof24": "test_results/BoT_gameof24.jsonl",
        "checkmate": "test_results/BoT_checkmate.jsonl",
        "wordsorting": "test_results/BoT_wordsorting.jsonl",
    }
    benchmark_path = benchmark_path_dict[task]
    correct = 0
    truth = []
    test = []
    with Path(benchmark_path).open() as f:
        for line in f:
            answer = json.loads(line)["target"]
            truth.append(answer)

    with Path(test_path).open() as f:
        for line in f:
            result = json.loads(line)["result"]
            result = result.split("\n")[0]
            if task == "gameof24":
                result = result.split("=")[0]
                try:
                    if ast.literal_eval(result) == TARGET_VALUE:
                        correct += 1
                except (ValueError, SyntaxError) as e:
                    logging.warning("Failed to evaluate result: %s, error: %s", result, e)
                    continue
            test.append(result)
    if correct == 0:
        for i in range(len(test)):
            if truth[i] == test[i]:
                correct += 1
