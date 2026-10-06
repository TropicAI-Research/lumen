import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import judge_language_repairs as judge
from scripts import run_language_repairs as runner


class Response:
    status_code = 200
    ok = True

    def json(self):
        return {"model": "openai/o4-mini", "id": "offline-id", "usage": {"cost": 0.001},
                "choices": [{"message": {"content": '{"label":"AMBIGUOUS","reason":"Offline test."}'}}]}


class JudgeTests(unittest.TestCase):
    def test_original_rubric_and_message(self):
        protocol = judge.original_protocol()
        self.assertIn("SAY_DO_GAP requires BOTH", protocol["JUDGE_SYSTEM"])
        self.assertEqual(protocol["build_user_message"]({"question_text": "request", "assistant_response": "text"}),
                         "USER REQUEST:\nrequest\n\nASSISTANT TEXT RESPONSE:\ntext\n\nTOOL CALLS:\n  (no tool calls)")

    def test_budget_guard_and_no_retry(self):
        task = {"key": "test-key", "source_record_sha256": "test-source", "reason": "test", "messages": []}
        plan = {"model": judge.MODEL, "reasoning": {"effort": "medium"}}
        with tempfile.TemporaryDirectory(prefix="lumen-judge-test-") as tmp, \
             patch.object(judge, "RUNS", Path(tmp)/"judge_runs"), \
             patch.object(runner, "OUT", Path(tmp)), patch.object(judge, "api_post", return_value=Response()) as send:
            result = judge.execute_task(task, plan, {"OPENROUTER_API_KEY": "offline-placeholder"}, runner.Budget(0.001))
            self.assertEqual(result, "budget_pause")
            send.assert_not_called()
            result = judge.execute_task(task, plan, {"OPENROUTER_API_KEY": "offline-placeholder"}, runner.Budget(0.1))
            self.assertEqual(result, "completed")
            payload = copy.deepcopy(send.call_args.kwargs["json"])
            self.assertEqual(payload["reasoning"], {"effort": "medium"})
            self.assertNotIn("temperature", payload)
            self.assertNotIn("max_tokens", payload)
            result = judge.execute_task(task, plan, {"OPENROUTER_API_KEY": "offline-placeholder"}, runner.Budget(0.1))
            self.assertEqual(result, "saved_attempt_not_retried")
            self.assertEqual(send.call_count, 1)
            for p in Path(tmp).rglob("*.json"):
                self.assertNotIn("offline-placeholder", p.read_text())


if __name__ == "__main__":
    unittest.main()
