import os

import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

import rag

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ---------------------------------------------------------------------------
# 2. Final test set
# ---------------------------------------------------------------------------
test_set = [
    {
        "question": "What specific risks did NVIDIA identify related to export restrictions or trade regulations?",
        "ground_truth": "NVIDIA identified specific risks from export restrictions and trade regulations, including competitive disadvantages against exempt competitors, non-U.S. customers 'designing-out' U.S. semiconductors, restrictions on downstream product usage, impaired ability and higher costs to provide cloud services, shipping limitations, and foreign governments encouraging customers to purchase from non-U.S. competitors.",
    },
    {
        "question": "What was Microsoft's revenue growth in its most recent fiscal year?",
        "ground_truth": "In fiscal year 2026, Microsoft's revenue increased by $50.1 billion, or 18%, driven by growth in Microsoft Cloud.",
    },
    {
        "question": "What is Apple's approach to returning capital to shareholders?",
        "ground_truth": "Apple returns capital to shareholders through an authorized share repurchase program and quarterly cash dividends, which the company intends to increase on an annual basis subject to declaration by the Board.",
    },
    {
        "question": "What was NVIDIA's revenue growth in its most recent fiscal year?",
        "ground_truth": "In fiscal year 2026, NVIDIA's revenue grew by approximately 65.5% to $215,938 million, up from $130,497 million in fiscal year 2025.",
    },
    {
        "question": "What risks did Apple disclose related to supply chain or manufacturing?",
        "ground_truth": "Apple disclosed risks regarding supply shortages, price increases, and design and manufacturing defects that could materially adversely affect its business, financial condition, stock price, operating results, and reputation.",
    },
    {
        "question": "What legal proceedings or litigation did Tesla disclose?",
        "ground_truth": "Tesla disclosed legal proceedings including stockholder derivative lawsuits concerning board oversight of its 2018 SEC settlement and director compensation awards, class action claims regarding its driver assistance technology systems, and litigation and investigations relating to alleged discrimination and harassment.",
    },
    {
        "question": "How much did Google spend on R&D, and what was the primary driver?",
        "ground_truth": "The provided context does not specify Google's total R&D expense, but it notes that Alphabet-level activities expenses were $16,760 million in 2025, which were primarily driven by shared AI research and development.",
    },
    {
        "question": "What did Meta identify as risks related to its metaverse or Reality Labs investments?",
        "ground_truth": "Meta identified risks including the potential failure of its Reality Labs strategy and investments, inability to build key relationships or interoperable products with third-party platforms, long R&D timelines, and continued substantial operating losses that could adversely affect its overall business, reputation, profitability, and financial results.",
    },
    {
        "question": "What did Amazon disclose as risks related to competition in its business?",
        "ground_truth": "Amazon disclosed that it faces intense business and industry risks from competition, as well as investigations regarding whether its store operations, fulfillment network, Prime, and AWS cloud services infringe competition-related or consumer protection rules.",
    },
    {
        "question": "What risks did Tesla disclose related to its autonomous driving or Full Self-Driving technology?",
        "ground_truth": "Tesla disclosed that its future growth and success depend on the adoption of autonomous driving solutions, and that uncertainty regarding the future of autonomous solutions could harm its business, prospects, financial condition, and operating results.",
    },
]

# ---------------------------------------------------------------------------
# 3. Run the real pipeline on every question
# ---------------------------------------------------------------------------
pipeline = rag.load_pipeline()

rows = []
for item in test_set:
    print(f"Running: {item['question']}")
    result = pipeline.answer(item["question"])
    rows.append({
        "question": item["question"],
        "answer": result.answer,
        "contexts": [p.text for p in result.passages],
        "ground_truth": item["ground_truth"],
    })

dataset = Dataset.from_list(rows)

judge = LangchainLLMWrapper(ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("Groq_api_key"),
    max_tokens=2048,
))
embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=rag.EMBEDDING_MODEL))

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge,
    embeddings=embeddings,
    run_config=RunConfig(timeout=180, max_workers=4),
)

scores = results.to_pandas()
report = pd.DataFrame({"question": [r["question"] for r in rows], "answer": [r["answer"] for r in rows]})
for col in (c for c in METRICS if c in scores.columns):
    report[col] = scores[col].values

print("\n=== Per-question scores ===")
print(report.drop(columns="answer").to_string())

print("\n=== Averages (successfully scored questions only) ===")
for col in (c for c in METRICS if c in report.columns):
    print(f"{col:<18} {report[col].mean():.4f}   (n = {report[col].notna().sum()}/{len(report)})")

report.to_csv("ragas_results.csv", index=False)