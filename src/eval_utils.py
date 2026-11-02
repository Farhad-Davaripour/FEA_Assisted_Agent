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
            return pd.DataFrame()
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
        if graded.empty or "label" not in graded.columns:
            continue
        graded = graded[graded["label"].isin(rails)].copy()
        if graded.empty:
            continue
        graded["score"] = (graded["label"] == rails[0]).astype(float)
        break  # Never retry a valid result in search of a preferred answer.
    # ---------------------------------------------------------------------

    if graded is None or graded.empty:
        return pd.DataFrame()
    graded.index.name = "span_id"
    client.log_evaluations(
        SpanEvaluations(eval_name=eval_name, dataframe=graded)
    )
    return graded

def parse_stress_mpa(output):
    """Parse only a tool result explicitly reported in MPa."""
    match = re.fullmatch(
        r"von_mises stress is\s+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s+MPa",
        output.strip(),
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else None
    
# --- revised helper (same signature, no .order_by/.limit) -------------
def log_stress_eval_real_time(client, stress_val):
    """
    Attach a Stress-Threshold evaluation to the latest agent step.
    Skips gracefully when no span or stress value is present yet.
    """
    # Only log a stress measurement from this run, never a leftover file.
    if stress_val is None:
        return
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
