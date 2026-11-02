"""Session-scoped Phoenix evaluations for the current submitted analysis."""
import math
import os
import re
import time
from typing import Callable, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.evals import create_classifier, evaluate_dataframe

load_dotenv(override=True)
stress_threshold = float(os.getenv("STRESS_THRESHOLD", 360.0))
PHOENIX_PROJECT_NAME = "fea-assisted-agent"


def _new_client():
    return Client(base_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006").rstrip("/"))


def _scoped_spans(client, trace_id: str, span_kind: str, select: Dict[str, str]) -> pd.DataFrame:
    """Query exactly one trace and verify returned trace IDs again client-side."""
    if not isinstance(trace_id, str) or not re.fullmatch(r"[0-9a-f]{32}", trace_id) or int(trace_id, 16) == 0:
        raise ValueError("A valid trace ID for the current submission is required.")
    if span_kind not in ("AGENT", "LLM", "TOOL"):
        raise ValueError("Unsupported span kind.")
    fields = dict(select)
    fields["_fea_trace_id"] = "context.trace_id"
    query = SpanQuery().where(
        f"trace_id == '{trace_id}' and span_kind == '{span_kind}'"
    ).select(*fields.values())
    # Export completion does not guarantee the Phoenix server has indexed the
    # span yet. Briefly retry an empty result, never broadening the trace scope.
    for attempt in range(5):
        df = client.spans.get_spans_dataframe(query=query, project_identifier=PHOENIX_PROJECT_NAME)
        if not df.empty or attempt == 4:
            break
        time.sleep(0.25)
    if df.empty:
        return pd.DataFrame()
    # The current client normalizes context.trace_id but the server returns the
    # qualified column name; normalize aliases after retrieval.
    aliases = {}
    for name, field in fields.items():
        for candidate in (field, f"attributes.{field}", name):
            if candidate in df.columns:
                aliases[candidate] = name
                break
    df = df.rename(columns=aliases)
    if "_fea_trace_id" not in df.columns:
        raise ValueError("Phoenix did not return trace identifiers; evaluation was skipped.")
    df = df[df["_fea_trace_id"] == trace_id].drop(columns=["_fea_trace_id"])
    if df.empty:
        return pd.DataFrame()
    if "start_time" not in df.columns:
        raise ValueError("Phoenix did not return span timestamps; evaluation was skipped.")
    return df.sort_values("start_time")


def run_eval(*, trace_id: str, span_kind: str, select: Dict[str, str],
             post_process: Callable[[pd.DataFrame], pd.DataFrame], template: str,
             rails: List[str], judge, eval_name: str, num_steps: int = 3,
             retries: int = 0, client: Optional[Client] = None) -> pd.DataFrame:
    """Evaluate this submission's spans; retry only invalid/empty classifications."""
    client = client or _new_client()
    df = _scoped_spans(client, trace_id, span_kind, select).tail(num_steps)
    if df.empty:
        return pd.DataFrame()
    eval_col_name = re.sub(r"[^a-zA-Z0-9_]+", "_", eval_name).strip("_").lower() or "eval"
    processed = post_process(df)
    if processed.empty:
        return pd.DataFrame()
    # Postprocessing must preserve the scoped span indices before they are annotated.
    if not processed.index.isin(df.index).all():
        raise ValueError("Evaluation included spans outside the current trace.")
    evaluator = create_classifier(name=eval_col_name, prompt_template=template, llm=judge, choices=rails)
    graded = pd.DataFrame()
    for _ in range(retries + 1):
        evaluated = evaluate_dataframe(dataframe=processed, evaluators=[evaluator], hide_tqdm_bar=True)
        score_col = f"{eval_col_name}_score"
        if score_col not in evaluated.columns:
            continue
        raw = evaluated[score_col].apply(lambda x: x if isinstance(x, dict) else {})
        graded = pd.DataFrame({
            "label": raw.apply(lambda d: d.get("label")),
            "explanation": raw.apply(lambda d: d.get("explanation")),
        }, index=processed.index)
        graded = graded[graded["label"].isin(rails)].copy()
        if graded.empty:
            continue
        graded["score"] = (graded["label"] == rails[0]).astype(float)
        break  # Never retry a valid label in pursuit of a preferred answer.
    if graded.empty:
        return pd.DataFrame()
    graded.index.name = "span_id"
    client.spans.log_span_annotations_dataframe(dataframe=graded, annotation_name=eval_name,
                                                annotator_kind="LLM", sync=True)
    return graded


def parse_stress_mpa(output):
    """Parse only a tool result explicitly reported in MPa."""
    match = re.fullmatch(
        r"von_mises stress is\s+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s+MPa",
        output.strip(), flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def log_stress_eval_real_time(*, trace_id: str, stress_val: float,
                              client: Optional[Client] = None) -> bool:
    """Record threshold comparison only on this submission's latest agent span."""
    if stress_val is None or not math.isfinite(stress_val) or stress_val < 0:
        return False
    client = client or _new_client()
    df_span = _scoped_spans(client, trace_id, "AGENT", {"start_time": "start_time"}).tail(1)
    if df_span.empty:
        return False
    exceeds = stress_val > stress_threshold
    label, score = ("exceeded", 1.0) if exceeds else ("not", 0.0)
    relation = "exceeds" if exceeds else "does not exceed"
    graded = pd.DataFrame({
        "label": [label], "score": [score],
        "explanation": [f"Stress = {stress_val:g} MPa {relation} threshold {stress_threshold:g} MPa"],
    }, index=df_span.index)
    graded.index.name = "span_id"
    client.spans.log_span_annotations_dataframe(dataframe=graded, annotation_name="Stress Thresh",
                                                annotator_kind="CODE", sync=True)
    return True
