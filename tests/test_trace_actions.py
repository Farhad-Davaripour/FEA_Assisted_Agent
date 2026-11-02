import json
import unittest

from src.trace_actions import extract_traced_action


class TraceActionTests(unittest.TestCase):
    def test_output_value_fallback_for_real_phoenix_shape(self):
        text = 'Thought: run.\nAction: Abaqus_job_executor\nAction Input: {"inp_file_name": "beam.inp"}'
        value = json.dumps({"message": {"role": "assistant", "blocks": [{"text": text}]}})
        self.assertEqual(extract_traced_action(None, value),
                         ("Abaqus_job_executor", '{"inp_file_name": "beam.inp"}'))

    def test_fallback_handles_literal_escaped_field_boundaries(self):
        text = 'Thought: run.\\nAction: Abaqus_job_executor\\nAction Input: {"inp_file_name": "beam.inp"}'
        value = json.dumps({"message": {"role": "assistant", "blocks": [{"text": text}]}})
        self.assertEqual(extract_traced_action(None, value),
                         ("Abaqus_job_executor", '{"inp_file_name": "beam.inp"}'))

    def test_output_messages_remain_supported(self):
        messages = [{"message": {"role": "assistant", "content":
                    'Thought: extract.\nAction: Von_Mises_stress_extractor\nAction Input: {}'}}]
        self.assertEqual(extract_traced_action(json.dumps(messages), None),
                         ("Von_Mises_stress_extractor", "{}"))

    def test_responses_nested_contents_are_supported(self):
        # Synthetic structure recorded from an actual Terra/Phoenix trace.
        action = 'Thought: run.\nAction: Abaqus_job_executor\nAction Input: {"inp_file_name": "src/abaqus_files/synthetic.inp"}'
        messages = [{"message": {"role": "assistant", "contents": [
            {"message_content": {"type": "text", "text": action}},
            {"message_content": {"type": "text", "text": action}},
        ]}}]
        expected = ("Abaqus_job_executor", '{"inp_file_name": "src/abaqus_files/synthetic.inp"}')
        self.assertEqual(extract_traced_action(messages, None), expected)
        self.assertEqual(extract_traced_action(json.dumps(messages), None), expected)

    def test_nested_contents_reject_non_assistant_non_text_and_malformed(self):
        action = 'Action: Abaqus_job_executor\nAction Input: {"inp_file_name": "beam.inp"}'
        cases = [
            [{"message": {"role": "user", "contents": [{"message_content": {"type": "text", "text": action}}]}}],
            [{"message": {"role": "assistant", "contents": [{"message_content": {"type": "image", "text": action}}]}}],
            [{"message": {"role": "assistant", "contents": [{"message_content": {"text": action}}]}}],
            [{"message": {"role": "assistant", "contents": [None, {"message_content": None}, {"message_content": {"type": "text", "text": 1}}]}}],
            [{"message": {"role": "assistant", "contents": [{"message_content": {"type": "text", "text": 'Answer: complete'}}]}}],
        ]
        for messages in cases:
            with self.subTest(messages=messages):
                self.assertIsNone(extract_traced_action(messages, None))

    def test_invalid_and_non_action_data_are_not_fabricated(self):
        cases = [
            (None, "not json"),
            (None, json.dumps({"message": {"role": "assistant", "blocks": [{"text": "Answer: complete"}]}})),
            (None, json.dumps({"message": {"role": "user", "blocks": [{"text": 'Action: foo\nAction Input: {}'}]}})),
            (None, json.dumps({"message": {"role": "assistant", "blocks": [{"text": 'Action: foo\nAction Input: {bad}'}]}})),
            (None, json.dumps({"message": {"role": "assistant", "blocks": [{"text": 'Action: foo\nAction Input: [1]'}]}})),
            (None, json.dumps({"message": {"role": "assistant", "blocks": [{"text": 'Action: foo'}]}})),
            ([None, 5, {"message": {}}], {"message": {"role": "assistant", "blocks": [None]}}),
        ]
        for messages, value in cases:
            with self.subTest(messages=messages, value=value):
                self.assertIsNone(extract_traced_action(messages, value))


if __name__ == "__main__":
    unittest.main()
