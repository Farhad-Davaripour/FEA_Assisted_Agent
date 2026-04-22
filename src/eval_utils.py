import os, json, re
from dotenv import load_dotenv
from typing import Callable, Dict, List
import pandas as pd
import phoenix as px
from phoenix.evals import create_classifier, evaluate_dataframe
try:
    from phoenix.evals import llm_classify  # Older Phoenix API
except ImportError:
    llm_classify = None
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

    eval_col_name = re.sub(r"[^a-zA-Z0-9_]+", "_", eval_name).strip("_").lower() or "eval"

    def _classify_with_current_api(processed: pd.DataFrame) -> pd.DataFrame:
        evaluator = create_classifier(
            name=eval_col_name,
            prompt_template=template,
            llm=judge,
            choices=rails,
        )
        evaluated = evaluate_dataframe(
            dataframe=processed,
            evaluators=[evaluator],
            hide_tqdm_bar=True,
        )
        score_col = f"{eval_col_name}_score"
        if score_col not in evaluated.columns:
            return pd.DataFrame(index=processed.index)
        raw = evaluated[score_col].apply(lambda x: x if isinstance(x, dict) else {})
        return pd.DataFrame(
            {
                "label": raw.apply(lambda d: d.get("label")),
                "score": raw.apply(lambda d: d.get("score")),
                "explanation": raw.apply(lambda d: d.get("explanation")),
            },
            index=processed.index,
        )

    # ------- retry loop ---------------------------------------------------
    graded = None
    for _ in range(retries + 1):                      # first try + N retries
        processed = post_process(df)
        if llm_classify is not None:
            graded = llm_classify(
                data=processed,
                template=template,
                rails=rails,
                model=judge,
                provide_explanation=True,
                run_sync=True,
            )
        else:
            graded = _classify_with_current_api(processed)

        # Phoenix v3 classifier may not return numeric score by default.
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
            .where("span_kind == 'AGENT'")
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
