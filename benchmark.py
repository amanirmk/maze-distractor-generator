"""Benchmark different models on the same input to compare runtime."""
import logging
import time
import os
import csv
from main import run_stuff

MODELS = [
    ("gpt2", "transformers"),
    ("distilgpt2", "transformers"),
    ("EleutherAI/gpt-neo-125M", "transformers"),
    ("HuggingFaceTB/SmolLM-360M", "transformers"),
    ("Qwen/Qwen2-0.5B", "transformers"),
    ("distilbert-base-uncased", "transformers"),
    ("distilroberta-base", "transformers"),
]

INFILE = "input/test_ns_9_short.csv"
PARAMS = "params.txt"
OUTFORMAT = "delim"


def run_benchmark(models=MODELS, infile=INFILE, parameters=PARAMS, outformat=OUTFORMAT,
                  results_file="output/benchmark_results_short.csv"):
    os.makedirs("output", exist_ok=True)
    results = []

    for model_name, backend_name in models:
        safe_name = model_name.replace("/", "_")
        outfile = f"output/bench_short_{safe_name}.csv"
        logging.info("=== Benchmarking %s (backend=%s) ===", model_name, backend_name)

        t0 = time.perf_counter()
        try:
            run_stuff(
                infile=infile,
                outfile=outfile,
                parameters=parameters,
                outformat=outformat,
                model_override=model_name,
                backend_override=backend_name,
            )
            elapsed = time.perf_counter() - t0
            status = "ok"
            logging.info("  %s finished in %.2fs", model_name, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            status = f"error: {e}"
            logging.error("  %s failed after %.2fs: %s", model_name, elapsed, e)

        results.append({
            "model": model_name,
            "backend": backend_name,
            "elapsed_s": round(elapsed, 2),
            "status": status,
        })

    print("\n" + "=" * 60)
    print(f"{'Model':<35} {'Backend':<20} {'Time (s)':>10}  Status")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<35} {r['backend']:<20} {r['elapsed_s']:>10.2f}  {r['status']}")
    print("=" * 60)

    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "backend", "elapsed_s", "status"])
        writer.writeheader()
        writer.writerows(results)
    logging.info("Results saved to %s", results_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_benchmark()
