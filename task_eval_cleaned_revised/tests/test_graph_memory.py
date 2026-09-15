"""Focused tests for Experiment #6 graph-memory retrieval."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import graph_memory as graph  # noqa: E402


class ScriptedLlama:
    """Deterministic LLM double that exercises all three graph-memory prompts."""

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        if "Extract factual memories" in prompt:
            return (
                '[{"subject":"Alice","relation":"works at","object":"Acme",'
                '"speaker":"Alice","dia_id":"D1"},'
                '{"subject":"Acme","relation":"located in","object":"Paris",'
                '"speaker":"Alice","dia_id":"D1"}]'
            )
        if "Select the graph entities" in prompt:
            return '["alice"]'
        if "Answer the question" in prompt:
            return "Paris"
        raise AssertionError(f"Unexpected prompt: {prompt}")


class MemoryGraphTests(unittest.TestCase):
    def test_traversal_follows_outgoing_and_incoming_edges(self) -> None:
        memory = graph.MemoryGraph()
        memory.add_edge("Alice", "works at", "Acme")
        memory.add_edge("Acme", "located in", "Paris")
        memory.add_edge("Paris", "is in", "France")

        traversed = memory.traverse(["ALICE"], max_hops=3)

        self.assertEqual([edge["hop"] for edge in traversed], [1, 2, 3])
        self.assertEqual(traversed[-1]["object"], "France")
        self.assertEqual(memory.traverse(["Acme"], max_hops=1)[0]["direction"], "outgoing")
        self.assertEqual(len(memory.traverse(["Acme"], max_hops=1)), 2)

    def test_parser_discards_malformed_and_incomplete_triples(self) -> None:
        parsed = graph.parse_json(
            '[{"subject":"Alice","relation":"likes","object":"tea"},'
            '{"subject":"ignored"}, 3]'
        )
        self.assertEqual(
            parsed,
            [{"subject": "Alice", "relation": "likes", "object": "tea"}],
        )
        self.assertEqual(graph.parse_json("not json"), [])

    def test_build_graph_keeps_session_and_dialogue_provenance(self) -> None:
        conversation = {
            "session_1_date_time": "2023-01-01",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1", "text": "I work at Acme."}
            ],
        }
        memory = graph.build_graph(ScriptedLlama(), conversation)
        edge = memory.traverse(["Alice"], max_hops=1)[0]
        self.assertEqual(edge["session"], 1)
        self.assertEqual(edge["timestamp"], "2023-01-01")
        self.assertEqual(edge["dia_id"], "D1")

    def test_end_to_end_pipeline_returns_grounded_evidence(self) -> None:
        conversation = {
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1", "text": "I work at Acme."}
            ]
        }
        results = graph.run_conversation(
            ScriptedLlama(),
            conversation,
            [{"question": "Where is Alice's employer located?", "answer": "Paris"}],
            max_hops=2,
            top_k=8,
        )
        self.assertEqual(results[0]["prediction"], "Paris")
        self.assertEqual(results[0]["seed_entities"], ["Alice"])
        self.assertIn(
            "Paris",
            [memory["object"] for memory in results[0]["retrieved_memories"]],
        )


if __name__ == "__main__":
    unittest.main()
