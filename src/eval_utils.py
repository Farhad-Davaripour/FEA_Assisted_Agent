from typing import Callable, Dict, List
import pandas as pd
import phoenix as px
from phoenix.evals import llm_classify
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery


def run_eval(
    *,
    span_kind: str,
    select: Dict[str, str],
    post_process: Callable[[pd.DataFrame], pd.DataFrame],
    template: str,
    rails: List[str],
    judge,
    eval_name: str,
) -> pd.DataFrame:
    """Collect spans, grade them with an LLM, push scores to Phoenix, and return the result."""
    client = px.Client()

    q = SpanQuery().where(f"span_kind == '{span_kind}'").select(**select)
    df = client.query_spans(q).sort_values("start_time").tail(3)

    if df.empty:
        return pd.DataFrame()  # caller can handle empty case

    graded = llm_classify(
        data=post_process(df),
        template=template,
        rails=rails,
        model=judge,
        provide_explanation=True,
    )
    # first rail is the "correct" one
    graded["score"] = (graded["label"] == rails[0]).astype(float)
    graded.index.name = "span_id"

    client.log_evaluations(
        SpanEvaluations(
            eval_name=eval_name,
            dataframe=graded,
        )
    )
    return graded