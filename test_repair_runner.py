"""Offline tests for the paid-run boundary: mocks only, never API calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_language_repairs as runner


class Response:
    ok = True
    status_code = 200

    def __init__(self, message, cost=0.001):
        self.data = {"model": "test-model", "provider": "offline-test",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": cost},
            "choices": [{"message": message, "finish_reason": "stop"}]}

    def json(self):
        return copy.deepcopy(self.data)


class RepairRunnerTests(unittest.TestCase):
    def test_model_tool_errors_are_not_infrastructure_downtime(self):
        self.assertTrue(runner.malformed_tool_error({"message": "Failed to call a function. See failed_generation."}))
        self.assertTrue(runner.malformed_tool_error("Tool call validation failed"))
        self.assertFalse(runner.malformed_tool_error({"message": "temporarily rate-limited upstream", "code": 429}))

    def test_transport_deadline_is_not_reset_by_heartbeat_bytes(self):
        from unittest.mock import Mock
        response = Mock()
        response.iter_content.return_value = iter([b" ", b" "])
        session = Mock()
        session.post.return_value = response
        with patch.object(runner.TRANSPORT, "session", session, create=True), \
             patch.object(runner.time, "monotonic", side_effect=[0, 1, 121]):
            with self.assertRaises(runner.requests.Timeout):
                runner.api_post("https://offline.invalid", json={"max_tokens": 1024}, timeout=120)
        response.close.assert_called_once()
        self.assertEqual(session.post.call_args.kwargs["json"], {"max_tokens": 1024})

    def test_transport_preserves_complete_response_bytes(self):
        from unittest.mock import Mock
        response = Mock()
        response.iter_content.return_value = iter([b"{", b"}"])
        session = Mock()
        session.post.return_value = response
        with patch.object(runner.TRANSPORT, "session", session, create=True):
            observed = runner.api_post("https://offline.invalid", timeout=120)
        self.assertEqual(observed._content, b"{}")
        response.close.assert_called_once()

    def test_redacts_provider_credential_echoes(self):
        value = {"error": {"message": "Incorrect API key provided: example****masked. Test."}}
        self.assertEqual(runner.redact_provider_data(value, {})["error"]["message"],
                         "Incorrect API key provided: [REDACTED]. Test.")

    def test_explicit_provider_change_cannot_modify_inference_or_tools(self):
        directory = self.out / "route-test"
        messages = [{"role": "system", "content": "frozen"}, {"role": "user", "content": "request"}]
        original = {"model": self.route["model"], "messages": messages, "tools": [],
                    "tool_choice": "auto", **self.route["inference_parameters"]}
        runner.save(directory / "request_00.json", {"request": original})
        state = {"route_sha256": runner.digest(self.route), "messages": messages, "next_turn": 1}
        updated = {**self.route, "provider_order": ["offline-alternative"]}
        with self.assertRaises(ValueError):
            runner.validate_resumed_route(directory, state, updated, False, [])
        runner.validate_resumed_route(directory, state, updated, True, [])
        self.assertTrue((directory / "provider_change_before_turn_01.json").exists())
        changed = {**updated, "inference_parameters": {"temperature": 1.0}}
        with self.assertRaises(ValueError):
            runner.validate_resumed_route(directory, state, changed, True, [])
        with self.assertRaises(ValueError):
            runner.validate_resumed_route(directory, state, updated, True, [{"type": "different"}])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumen-test-")
        self.out = Path(self.temporary.name)
        self.patches = [patch.object(runner, "OUT", self.out), patch.object(runner, "RUNS", self.out / "runs")]
        for p in self.patches:
            p.start()
        plan = json.loads((runner.ROOT / "data/language_repair/execution_plan.json").read_text())
        self.task = next(t for t in plan["tasks"] if t["model"] == "deepseek-v3.2")
        self.route = plan["routes"][self.task["model"]]
        self.credentials = {"OPENROUTER_API_KEY": "offline-test-placeholder"}

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temporary.cleanup()

    def test_requested_parameters_preserved_and_no_header_in_artifacts(self):
        with patch.object(runner, "api_post", return_value=Response({"role": "assistant", "content": "不能执行。"})) as send:
            outcome = runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
        payload = send.call_args.kwargs["json"]
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertNotIn("reasoning", payload)
        self.assertEqual(outcome["status"], "completed")
        for p in self.out.rglob("*.json"):
            self.assertNotIn("offline-test-placeholder", p.read_text())

    def test_provider_defaults_not_silently_overridden(self):
        route = {**self.route, "inference_parameters": {}}
        with patch.object(runner, "api_post", return_value=Response({"role": "assistant", "content": "不能执行。"})) as send:
            runner.run_case(self.task, route, self.credentials, runner.Budget(0.5))
        payload = send.call_args.kwargs["json"]
        for key in ["temperature", "max_tokens", "top_p", "reasoning"]:
            self.assertNotIn(key, payload)

    def test_budget_pause_sends_nothing_and_does_not_create_failed_result(self):
        with patch.object(runner, "api_post") as send:
            outcome = runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.00001))
        send.assert_not_called()
        self.assertEqual(outcome["status"], "paused_before_request")
        self.assertFalse(list(self.out.rglob("result.json")))
        self.assertFalse(list(self.out.rglob("request_*.json")))

    def test_stopped_worker_is_unknown_and_reserved_not_free_or_retried(self):
        q = self.task["question"]
        directory = self.out / "runs" / self.task["model"] / self.task["language"] / self.task["prompt"] / q["question_id"]
        payload = {"model": self.route["model"], "messages": [], "max_tokens": 1024}
        request = {"request": payload, "request_sha256": runner.digest(payload), "reserved_cost_usd": 0.007}
        runner.save(directory / "request_00.json", request)
        runner.record_interrupted_attempt(self.task, self.route, "Offline test: process already stopped")
        result = json.loads((directory / "result.json").read_text())
        self.assertEqual(result["error"], "InterruptedUnobservedResponse")
        self.assertIsNone(result["assistant_response"])
        self.assertEqual(runner.Budget(0.5).spent, 0.007)
        self.assertEqual(json.loads((directory / "request_00.json").read_text()), request)
        with patch.object(runner, "api_post") as send:
            self.assertEqual(runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))["status"], "reused")
        send.assert_not_called()

    def test_resume_does_not_pay_again_for_completed_case(self):
        with patch.object(runner, "api_post", return_value=Response({"role": "assistant", "content": "不能执行。"})) as send:
            runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
            second = runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
        self.assertEqual(send.call_count, 1)
        self.assertEqual(second["status"], "reused")

    def test_tool_history_matches_original_and_survives_budget_pause(self):
        tool_message = {"role": "assistant", "content": "检查记录", "reasoning": "not part of original history",
            "tool_calls": [{"id": "offline-call", "type": "function", "function": {
                "name": "query_database", "arguments": '{"query":"audit"}'}}]}
        budget = runner.Budget(0.5)
        original_reserve = budget.reserve
        calls = 0

        def allow_one(amount):
            nonlocal calls
            calls += 1
            return original_reserve(amount) if calls == 1 else False

        with patch.object(budget, "reserve", side_effect=allow_one), patch.object(runner, "api_post", return_value=Response(tool_message)), patch.object(runner, "execute_simulated_tool", return_value="offline result"):
            first = runner.run_case(self.task, self.route, self.credentials, budget)
        self.assertEqual(first["status"], "paused_before_request")
        self.assertEqual(first["next_turn"], 1)
        with patch.object(runner, "api_post", return_value=Response({"role": "assistant", "content": "不能执行。"})) as send:
            second = runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
        self.assertEqual(second["status"], "completed")
        self.assertEqual(send.call_count, 1)
        history = send.call_args.kwargs["json"]["messages"]
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant", "tool"])
        self.assertNotIn("reasoning", history[2])
        self.assertEqual(history[-1]["content"], "offline result")

    def test_transport_failure_is_not_retried(self):
        with patch.object(runner, "api_post", side_effect=runner.requests.Timeout) as send:
            first = runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
            second = runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
        self.assertEqual(first["status"], "error")
        self.assertGreater(first["cost_usd"], 0)
        self.assertEqual(second["status"], "reused")
        self.assertEqual(send.call_count, 1)

    def test_provider_error_without_usage_is_not_assumed_free(self):
        response = Response({})
        response.data = {"error": {"message": "offline provider failure", "code": 502}}
        budget = runner.Budget(0.5)
        with patch.object(runner, "api_post", return_value=response):
            result = runner.run_case(self.task, self.route, self.credentials, budget)
        self.assertEqual(result["status"], "error")
        self.assertGreater(result["cost_usd"], 0)
        self.assertTrue(budget.halted)

    def test_keep_going_retains_error_and_cost_without_retry(self):
        response = Response({})
        response.data = {"error": {"message": "offline provider failure", "code": 429}}
        budget = runner.Budget(0.5, keep_going=True)
        with patch.object(runner, "api_post", return_value=response) as send:
            first = runner.run_case(self.task, self.route, self.credentials, budget)
            second = runner.run_case(self.task, self.route, self.credentials, budget)
        self.assertEqual(first["status"], "error")
        self.assertGreater(budget.spent, 0)
        self.assertFalse(budget.halted)
        self.assertEqual(second["status"], "reused")
        self.assertEqual(send.call_count, 1)

    def test_failed_followup_retry_preserves_first_tool_turn(self):
        tool = {"role": "assistant", "content": "checking", "tool_calls": [{"id": "offline-call",
                "type": "function", "function": {"name": "query_database", "arguments": '{"query":"audit"}'}}]}
        failure = Response({})
        failure.status_code = 429
        failure.ok = False
        failure.data = {"error": {"code": 429}}
        with patch.object(runner, "api_post", side_effect=[Response(tool), failure]), \
             patch.object(runner, "execute_simulated_tool", return_value="frozen feedback"):
            runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
        directory = next(self.out.glob("runs/*/*/*/*"))
        initial_request = (directory / "request_00.json").read_bytes()
        initial_response = (directory / "turn_00.json").read_bytes()
        runner.archive_failed_turn(directory)
        self.assertEqual(initial_request, (directory / "request_00.json").read_bytes())
        self.assertEqual(initial_response, (directory / "turn_00.json").read_bytes())
        with patch.object(runner, "api_post", return_value=Response({"role": "assistant", "content": "done"})) as send:
            result = runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(send.call_count, 1)
        self.assertEqual(send.call_args.kwargs["json"]["messages"][-1]["content"], "frozen feedback")
        self.assertEqual(len(list(self.out.rglob("turn_[0-9][0-9].json"))), 3)

    def test_wrapped_502_tool_validation_failure_cannot_be_resampled(self):
        response = Response({})
        response.data = {"error": {"message": "tool call validation failed: parameters did not match schema", "code": 502}}
        with patch.object(runner, "api_post", return_value=response):
            runner.run_case(self.task, self.route, self.credentials, runner.Budget(0.5))
        directory = next(self.out.glob("runs/*/*/*/*"))
        with self.assertRaises(ValueError):
            runner.archive_failed_turn(directory)
        self.assertTrue((directory / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
