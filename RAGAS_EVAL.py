import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Import the exact, already-tested pipeline pieces from your real app.py,
# instead of duplicating them here and risking a second copy drifting out of sync.
from app import load_pipeline, retrieve, generate

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from google import genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# 1. Load the pipeline exactly as app.py does (reuses existing ./chroma_db,
#    does NOT re-embed since the collection already has data).
# ---------------------------------------------------------------------------
pipeline = load_pipeline()

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
questions, answers, contexts_list, ground_truths = [], [], [], []

for item in test_set:
    q = item["question"]
    print(f"Running: {q}")
    retrieved = retrieve(q, pipeline)          # pipeline passed in, matching app.py's signature
    answer = generate(q, retrieved, pipeline)  # pipeline passed in, matching app.py's signature

    questions.append(q)
    answers.append(answer)
    contexts_list.append(retrieved)
    ground_truths.append(item["ground_truth"])

eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts_list,
    "ground_truth": ground_truths,
})

# ---------------------------------------------------------------------------
# 4. Wire RAGAS to use Gemini + your existing local embedder
# ---------------------------------------------------------------------------
judge_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROQ_API_KEY"),
    max_tokens=2048,
)
ragas_llm = LangchainLLMWrapper(judge_llm)

local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

# Throttle concurrency + give each job more time to wait its turn in the
# free-tier rate limit queue, instead of firing all 40 jobs at once.
run_config = RunConfig(
    timeout=180,
    max_workers=4,
)

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config,
)

print("\n=== RAGAS Evaluation Results (overall averages) ===")
print(results)

# ---------------------------------------------------------------------------
# 5. Per-question breakdown
#
# results.to_pandas() does not reliably return a "question" column across
# RAGAS versions (it may use "user_input" / "question" / etc. depending on
# version). Rather than guess and crash with a KeyError, build the report
# from our own known-good eval_dataset and merge in whatever score columns
# RAGAS actually returned.
# ---------------------------------------------------------------------------
df = results.to_pandas()
print("\nColumns returned by RAGAS:", df.columns.tolist())

metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]

report = pd_df = None
try:
    import pandas as pd
    report = pd.DataFrame({"question": questions, "answer": answers})
    for col in metric_cols:
        report[col] = df[col].values
except Exception as e:
    print(f"Could not build merged report ({e}); falling back to raw RAGAS dataframe.")
    report = df

print("\n=== Per-question breakdown ===")
print(report)

report.to_csv("ragas_results.csv", index=False)
print("\nSaved detailed results to ragas_results.csv")