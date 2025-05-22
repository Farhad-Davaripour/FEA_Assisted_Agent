import os, json, re
from dotenv import load_dotenv
from typing import Callable, Dict, List
import pandas as pd
import phoenix as px
from phoenix.evals import llm_classify
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
load_dotenv(override=True)
stress_threshold = float(os.getenv("STRESS_THRESHOLD", 360.0))

import warnings
warnings.filterwarnings(
    "ignore",
    message="Series.__getitem__ treating keys as positions is deprecated",
    category=FutureWarning,
    module="phoenix.trace.dsl.query",
)

def run_eval(
    *,
    span_kind: str,
    select: Dict[str, str],
    post_process: Callable[[pd.DataFrame], pd.DataFrame],
    template: str,
    rails: List[str],
    judge,
    eval_name: str,
    num_steps: int = 3,
    retries: int = 0,                 # 👈 NEW
) -> pd.DataFrame:
    """Grade spans with an LLM, retrying up to `retries` extra times if not all pass."""
    client = px.Client()
    df = (
        client.query_spans(
            SpanQuery().where(f"span_kind == '{span_kind}'").select(**select)
        )
        .sort_values("start_time")
        .tail(num_steps)
    )

    if df.empty:
        return pd.DataFrame()

    # ------- retry loop ---------------------------------------------------
    graded = None
    for _ in range(retries + 1):                      # first try + N retries
        graded = llm_classify(
            data=post_process(df),
            template=template,
            rails=rails,
            model=judge,
            provide_explanation=True,
            run_sync=True,
        )
        graded["score"] = (graded["label"] == rails[0]).astype(float)
        if graded["score"].sum() == len(graded):      # all correct → stop
            break
    # ---------------------------------------------------------------------

    graded.index.name = "span_id"
    client.log_evaluations(
        SpanEvaluations(eval_name=eval_name, dataframe=graded)
    )
    return graded

def parse_stress_mpa(path="src/abaqus_files/max_vm_stress.txt"):
    """Return the stress value in MPa if file exists, else None."""
    try:
        with open(path) as f:
            text = f.read()
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if not m:
            return None
        value = float(m.group(1))
        if value > 1000:
            value /= 1e6
        return value
    except OSError:
        return None
    
# --- revised helper (same signature, no .order_by/.limit) -------------
def log_stress_eval_real_time(client):
    """
    Attach a Stress-Threshold evaluation to the latest agent step.
    Skips gracefully when no span or stress value is present yet.
    """
    # 1️⃣ fetch the most-recent ReActAgent step
    df_span = (
        client.query_spans(
            SpanQuery()
            .where("span_kind == 'AGENT' and name == 'ReActAgentWorker.run_step'")
            .select()
        )
        .sort_values("start_time")
        .tail(1)                 # equivalent to order_by + limit
    )

    if df_span.empty:            # nothing to tag yet
        return

    latest_span_id = df_span.index[0]

    # 2️⃣ deterministic stress check
    stress_val = parse_stress_mpa()
    if stress_val is None:       # file not created yet
        label, score = "not", 0.0
        explanation = "Stress value not available for this step."
    else:
        exceeds = stress_val > stress_threshold
        label, score = ("exceeded", 1.0) if exceeds else ("not", 0.0)
        explanation = (
            f"Stress = {stress_val:.1f} MPa exceeds threshold {stress_threshold:.1f} MPa"
            if exceeds
            else f"Stress = {stress_val:.1f} MPa is below threshold {stress_threshold:.1f} MPa"
        )

    graded = pd.DataFrame(
        {"label": [label], "score": [score], "explanation": [explanation]},
        index=[latest_span_id],
    )
    graded.index.name = "span_id"

    # 3️⃣ write to Phoenix
    client.log_evaluations(
        SpanEvaluations(eval_name="Stress Thresh", dataframe=graded)
    )