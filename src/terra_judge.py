"""Use GPT-6.1 Terra through Responses for Phoenix classification evaluators."""
import json
from typing import Any

from openai import AsyncOpenAI, OpenAI


class TerraResponsesJudge:
    """Minimal Phoenix-compatible classifier backed by the OpenAI Responses API.

    Phoenix's ClassificationEvaluator calls these two methods and reads
    ``model``. Clients can be injected for offline tests; otherwise the SDK
    reads credentials from the standard OpenAI environment.
    """

    model = "gpt-6.1-terra"

    def __init__(self, *, client: Any = None, async_client: Any = None):
        self._client = client
        self._async_client = async_client

    @staticmethod
    def _parameters(prompt, labels, include_explanation, description):
        choices = list(labels) if isinstance(labels, (list, dict)) else []
        if not choices or any(not isinstance(label, str) or not label for label in choices):
            raise ValueError("Classification labels must be nonempty strings.")
        if len(set(choices)) != len(choices):
            raise ValueError("Classification labels must be unique.")

        label_field = {"type": "string", "enum": choices}
        if isinstance(labels, dict):
            label_field["description"] = "; ".join(
                f"{label}: {labels[label]}" for label in choices
            )
        properties = {"label": label_field}
        if include_explanation:
            properties["explanation"] = {"type": "string"}
        schema = {
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        }
        if description:
            schema["description"] = description
        return {
            "model": TerraResponsesJudge.model,
            "input": prompt,
            "reasoning": {"effort": "medium"},
            "store": False,
            "text": {"format": {
                "type": "json_schema", "name": "phoenix_classification",
                "strict": True, "schema": schema,
            }},
        }, choices

    @staticmethod
    def _parse(response, choices, include_explanation):
        if getattr(response, "status", None) not in (None, "completed"):
            raise ValueError("The model did not complete the classification.")
        try:
            result = json.loads(response.output_text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("The model did not return a valid classification.") from exc
        expected = {"label", "explanation"} if include_explanation else {"label"}
        if (not isinstance(result, dict) or set(result) != expected
                or result.get("label") not in choices
                or (include_explanation and not isinstance(result.get("explanation"), str))):
            raise ValueError("The model returned an invalid classification.")
        return result

    def generate_classification(self, prompt, labels, include_explanation=True,
                                description=None, **kwargs):
        params, choices = self._parameters(prompt, labels, include_explanation, description)
        if self._client is None:
            self._client = OpenAI()
        return self._parse(self._client.responses.create(**params), choices, include_explanation)

    async def async_generate_classification(self, prompt, labels, include_explanation=True,
                                            description=None, **kwargs):
        params, choices = self._parameters(prompt, labels, include_explanation, description)
        if self._async_client is None:
            self._async_client = AsyncOpenAI()
        response = await self._async_client.responses.create(**params)
        return self._parse(response, choices, include_explanation)
