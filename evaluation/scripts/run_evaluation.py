import json
import os
from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

load_dotenv()

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "ragas_evaluation_dataset.json"

def load_dataset(path: Path) -> Dataset:
    with open(path, "r") as f:
        data = json.load(f)

    return Dataset.from_dict({
        "question":         [item["question"] for item in data],
        "contexts":         [item["contexts"] for item in data],
        "answer":           [item["answer"] for item in data],
        "ground_truth":     [item.get("ground_truth", item["answer"]) for item in data],
    })

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} samples.\n")

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    print("Running RAGAS evaluation...")
    result = evaluate(dataset=dataset, metrics=metrics)

    import csv
    df = result.to_pandas()

    print("\n=== Evaluation Results ===")
    metric_cols = [c for c in df.columns if c not in ("question", "contexts", "answer", "ground_truth")]
    import pandas as pd
    summary = {col: round(pd.to_numeric(df[col], errors="coerce").mean(), 4) for col in metric_cols}
    for metric, score in summary.items():
        print(f"  {metric:<25} {score:.4f}")

    output_path = Path(__file__).parent.parent / "results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
