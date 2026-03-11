"""
Run all test scenarios from data/test_scenarios.yaml and report results.

Usage: python eval/run_eval.py
"""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
from agent import FoodOrderAgent


def run_eval():
    with open("data/test_scenarios.yaml") as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    results = []
    for scenario in scenarios:
        print(f"Running: {scenario['name']}...")
        agent = FoodOrderAgent()
        scenario_passed = True
        failures = []

        for turn in scenario["turns"]:
            response = agent.send(turn["user"])

            if "expect_tool_call" in turn:
                tool_calls = response.get("tool_calls", [])
                if not tool_calls:
                    scenario_passed = False
                    failures.append(
                        f"  Turn '{turn['user'][:40]}': Expected tool call '{turn['expect_tool_call']}', got none"
                    )
                else:
                    names = [tc["name"] for tc in tool_calls]
                    if turn["expect_tool_call"] not in names:
                        scenario_passed = False
                        failures.append(
                            f"  Turn '{turn['user'][:40]}': Expected tool '{turn['expect_tool_call']}', got {names}"
                        )

            if "expect_order_items" in turn:
                snapshot = agent.order_manager.get_snapshot()
                actual = len(snapshot["items"])
                expected = turn["expect_order_items"]
                if actual != expected:
                    scenario_passed = False
                    failures.append(
                        f"  Turn '{turn['user'][:40]}': Expected {expected} items, got {actual}"
                    )

            if "expect_total" in turn:
                snapshot = agent.order_manager.get_snapshot()
                actual = snapshot["total"]
                expected = turn["expect_total"]
                if abs(actual - expected) > 0.01:
                    scenario_passed = False
                    failures.append(
                        f"  Turn '{turn['user'][:40]}': Expected total ${expected:.2f}, got ${actual:.2f}"
                    )

            if "expect_error" in turn and turn["expect_error"]:
                msg = response.get("message", "").lower()
                if not any(
                    word in msg for word in ("empty", "cannot", "can't", "error", "limit", "invalid")
                ):
                    scenario_passed = False
                    failures.append(
                        f"  Turn '{turn['user'][:40]}': Expected error response, got: {response['message'][:60]}"
                    )

        results.append(
            {
                "scenario": scenario["name"],
                "passed": scenario_passed,
                "failures": failures,
            }
        )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"{'Scenario':<45} {'Result'}")
    print(f"{'=' * 60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['scenario']:<45} {status}")
        for f in r["failures"]:
            print(f)

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'=' * 60}")
    print(f"Result: {passed}/{len(results)} scenarios passed")
    return passed == len(results)


if __name__ == "__main__":
    success = run_eval()
    sys.exit(0 if success else 1)
