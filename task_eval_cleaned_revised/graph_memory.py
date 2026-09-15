"""Experiment #6: graph-backed memory for the LoCoMo benchmark.

The experiment turns each LoCoMo conversation into factual graph edges, grounds a
question in graph nodes, traverses the relevant neighbourhood, and asks
Llama-3.2-3B-Instruct to answer from only that evidence. The module is kept
dependency-light at import time so graph construction can be unit-tested
without downloading the model.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Protocol, Set


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_MAX_HOPS = 2
DEFAULT_TOP_K = 8


class TextGenerator(Protocol):
    """The small interface used by graph construction and question answering."""

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        """Generate text for *prompt*."""


class LlamaModel:
    """Local Hugging Face wrapper for Llama-3.2-3B-Instruct.

    ``transformers`` and ``torch`` are deliberately imported here rather than at
    module import time. This lets contributors run the graph-only tests on a
    lightweight Python installation and gives an actionable error for a missing
    inference dependency.
    """

    def __init__(self, model_name: str = MODEL_NAME, device_map: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(
                "Graph-memory inference requires torch and transformers. "
                "Install the project dependencies before running the CLI."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: Dict[str, Any] = {"torch_dtype": "auto"}
        if device_map:
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)

        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def _normalise_entity(entity: str) -> str:
    """Create a stable lookup key without changing the display spelling."""

    return re.sub(r"\s+", " ", entity.strip()).casefold()


def _edge_identity(edge: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        _normalise_entity(edge["subject"]),
        edge["relation"].strip().casefold(),
        _normalise_entity(edge["object"]),
        edge.get("speaker"),
        edge.get("dia_id"),
        edge.get("session"),
        edge.get("timestamp"),
    )


class MemoryGraph:
    """A provenance-preserving, traversable graph of LoCoMo facts.

    Each fact is stored once in its stated direction, while incoming edges are
    indexed as well. That makes a graph node discoverable whether it was the
    subject or object of a fact without fabricating inverse relations.
    """

    def __init__(self) -> None:
        self.edges: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._incoming: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._entity_names: Dict[str, str] = {}
        self._edge_ids: Set[tuple[Any, ...]] = set()

    def add_edge(
        self,
        subject: str,
        relation: str,
        object_: str,
        speaker: Optional[str] = None,
        dia_id: Optional[str] = None,
        session: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a factual edge, ignoring exact duplicate extraction results."""

        subject, relation, object_ = subject.strip(), relation.strip(), object_.strip()
        if not (subject and relation and object_):
            return

        edge = {
            "subject": subject,
            "relation": relation,
            "object": object_,
            "speaker": speaker,
            "dia_id": dia_id,
            "session": session,
            "timestamp": timestamp,
        }
        identity = _edge_identity(edge)
        if identity in self._edge_ids:
            return
        self._edge_ids.add(identity)

        subject_key = _normalise_entity(subject)
        object_key = _normalise_entity(object_)
        self._entity_names.setdefault(subject_key, subject)
        self._entity_names.setdefault(object_key, object_)
        self.edges[subject_key].append(edge)
        self._incoming[object_key].append(edge)

    def entities(self) -> Set[str]:
        """Return display names for every entity represented by the graph."""

        return set(self._entity_names.values())

    def resolve_entity(self, entity: str) -> Optional[str]:
        """Resolve a case/whitespace-insensitive entity to its display name."""

        return self._entity_names.get(_normalise_entity(entity))

    def traverse(
        self, start_entities: Iterable[str], max_hops: int = DEFAULT_MAX_HOPS
    ) -> List[Dict[str, Any]]:
        """Breadth-first traverse adjacent facts up to ``max_hops``.

        Results retain the original subject/relation/object statement, include a
        one-based ``hop`` distance, and indicate whether the fact was entered via
        its subject or object. Repeated paths do not duplicate an edge.
        """

        if max_hops < 1:
            return []

        start_keys = {
            _normalise_entity(entity)
            for entity in start_entities
            if self.resolve_entity(entity) is not None
        }
        queue = deque((entity, 0) for entity in sorted(start_keys))
        visited: Set[str] = set()
        seen_edges: Set[tuple[Any, ...]] = set()
        results: List[Dict[str, Any]] = []

        while queue:
            entity, depth = queue.popleft()
            if entity in visited or depth >= max_hops:
                continue
            visited.add(entity)

            neighbours = [
                (edge, "outgoing", _normalise_entity(edge["object"]))
                for edge in self.edges.get(entity, [])
            ]
            neighbours.extend(
                (edge, "incoming", _normalise_entity(edge["subject"]))
                for edge in self._incoming.get(entity, [])
            )

            for edge, direction, next_entity in neighbours:
                identity = _edge_identity(edge)
                if identity not in seen_edges:
                    result = dict(edge)
                    result["hop"] = depth + 1
                    result["direction"] = direction
                    results.append(result)
                    seen_edges.add(identity)
                if next_entity not in visited:
                    queue.append((next_entity, depth + 1))

        return results


def _json_array(text: str) -> List[Any]:
    """Return the first JSON array in model output, or an empty list."""

    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        return []
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def parse_json(text: str) -> List[Dict[str, str]]:
    """Parse valid factual triples from an LLM JSON response.

    The original public starter exposed ``parse_json``; retaining this name
    avoids breaking notebooks that already import it.
    """

    triples: List[Dict[str, str]] = []
    for item in _json_array(text):
        if not isinstance(item, dict):
            continue
        subject, relation, object_ = (
            item.get("subject"),
            item.get("relation"),
            item.get("object"),
        )
        if subject and relation and object_:
            triple = {
                "subject": str(subject).strip(),
                "relation": str(relation).strip(),
                "object": str(object_).strip(),
            }
            for key in ("speaker", "dia_id"):
                if item.get(key) is not None:
                    triple[key] = str(item[key]).strip()
            triples.append(triple)
    return triples


def extract_triples(llm: TextGenerator, conversation: str) -> List[Dict[str, str]]:
    """Ask the LLM for stated facts in one LoCoMo session."""

    prompt = f"""
Extract factual memories from this conversation.

Return ONLY a JSON array in this format:
[
  {{
    "subject": "entity",
    "relation": "short relationship",
    "object": "entity or value",
    "speaker": "optional speaker name",
    "dia_id": "optional dialogue id"
  }}
]

Rules:
- Extract only information stated in the conversation; never infer a fact.
- Preserve names, places, dates, quantities, and negation where stated.
- Use concise relation names and entity strings that occur in the conversation.
- Include dia_id when the source line contains one.
- Do not include conversational filler or duplicate facts.

Conversation:
{conversation}
"""
    return parse_json(llm.generate(prompt, max_new_tokens=800))


def load_locomo(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("LoCoMo data must be a JSON list of conversations.")
    return data


def get_sessions(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return numbered LoCoMo sessions in chronological session order."""

    sessions: List[Dict[str, Any]] = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if not match or not isinstance(value, list):
            continue
        number = int(match.group(1))
        sessions.append(
            {
                "number": number,
                "timestamp": conversation.get(f"session_{number}_date_time"),
                "dialogues": value,
            }
        )
    return sorted(sessions, key=lambda session: session["number"])


def _session_text(session: Dict[str, Any]) -> str:
    lines: List[str] = []
    for turn in session["dialogues"]:
        text = turn.get("text")
        if not text:
            continue
        dialogue_id = turn.get("dia_id", turn.get("id", ""))
        source = f"[{dialogue_id}] " if dialogue_id else ""
        speaker = turn.get("speaker", "Unknown")
        lines.append(f"{source}{speaker}: {text}")
    return "\n".join(lines)


def build_graph(llm: TextGenerator, conversation: Dict[str, Any]) -> MemoryGraph:
    """Extract factual graph edges session by session from a LoCoMo conversation."""

    graph = MemoryGraph()
    for session in get_sessions(conversation):
        for triple in extract_triples(llm, _session_text(session)):
            graph.add_edge(
                triple["subject"],
                triple["relation"],
                triple["object"],
                speaker=triple.get("speaker"),
                dia_id=triple.get("dia_id"),
                session=session["number"],
                timestamp=session["timestamp"],
            )
    return graph


def _lexical_entity_matches(question: str, graph: MemoryGraph) -> List[str]:
    question_key = _normalise_entity(question)
    matches = [entity for entity in graph.entities() if _normalise_entity(entity) in question_key]
    return sorted(matches, key=lambda entity: (-len(entity), entity))


def find_question_entities(
    llm: TextGenerator, question: str, graph: MemoryGraph
) -> List[str]:
    """Ground a question in graph nodes, with deterministic lexical fallback."""

    entities = sorted(graph.entities())
    if not entities:
        return []
    prompt = f"""
Select the graph entities needed to answer this question.

Question:
{question}

Available entities:
{json.dumps(entities, ensure_ascii=False)}

Return ONLY a JSON array containing exact entity names from Available entities.
Return [] if none are relevant.
"""
    selected: List[str] = []
    for candidate in _json_array(llm.generate(prompt, max_new_tokens=100)):
        if isinstance(candidate, str):
            resolved = graph.resolve_entity(candidate)
            if resolved and resolved not in selected:
                selected.append(resolved)

    # A grounding failure must not turn a question containing a known name into
    # an empty traversal just because the model emitted a slightly different case.
    for entity in _lexical_entity_matches(question, graph):
        if entity not in selected:
            selected.append(entity)
    return selected


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b[\w'-]+\b", text.casefold()))


def rank_memories(
    question: str, memories: Iterable[Dict[str, Any]], top_k: int = DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """Rank traversal facts by lexical overlap while favouring nearby hops."""

    question_words = tokenize(question)
    scored: List[tuple[int, int, Dict[str, Any]]] = []
    for position, memory in enumerate(memories):
        fact_words = tokenize(
            " ".join(
                [memory["subject"], memory["relation"], memory["object"]]
            )
        )
        overlap = len(question_words & fact_words)
        hop_bonus = max(0, DEFAULT_MAX_HOPS + 1 - int(memory["hop"]))
        scored.append((overlap * 3 + hop_bonus, -position, memory))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [memory for _, _, memory in scored[:top_k]]


def format_memories(memories: Iterable[Dict[str, Any]]) -> str:
    """Render traversed graph facts with LoCoMo provenance for the answer prompt."""

    lines: List[str] = []
    for memory in memories:
        provenance: List[str] = []
        if memory.get("dia_id"):
            provenance.append(f"dialogue {memory['dia_id']}")
        if memory.get("session") is not None:
            provenance.append(f"session {memory['session']}")
        if memory.get("timestamp"):
            provenance.append(str(memory["timestamp"]))
        source = f" [{' | '.join(provenance)}]" if provenance else ""
        lines.append(
            f"(hop {memory['hop']}) {memory['subject']} "
            f"--{memory['relation']}--> {memory['object']}{source}"
        )
    return "\n".join(lines)


def answer_question(
    llm: TextGenerator, question: str, memories: Iterable[Dict[str, Any]]
) -> str:
    """Answer a LoCoMo question using only graph-traversal evidence."""

    memories = list(memories)
    if not memories:
        return "Not mentioned in the conversation."
    prompt = f"""
Answer the question using only the retrieved memories below.

Retrieved memories:
{format_memories(memories)}

Question:
{question}

Give a concise answer. If the memories do not establish an answer, respond
exactly: Not mentioned in the conversation.
"""
    return llm.generate(prompt, max_new_tokens=150).strip()


def run_conversation(
    llm: TextGenerator,
    conversation: Dict[str, Any],
    questions: Iterable[Dict[str, Any]],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Run the graph-nodes → graph-traversal pipeline for one LoCoMo item."""

    graph = build_graph(llm, conversation)
    results: List[Dict[str, Any]] = []
    for qa in questions:
        question = qa.get("question", "")
        entities = find_question_entities(llm, question, graph)
        traversed = graph.traverse(entities, max_hops=max_hops)
        memories = rank_memories(question, traversed, top_k=top_k)
        results.append(
            {
                "question": question,
                "prediction": answer_question(llm, question, memories),
                "gold_answer": qa.get("answer"),
                "evidence": qa.get("evidence", []),
                "category": qa.get("category"),
                "seed_entities": entities,
                "retrieved_memories": memories,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment #6: graph-memory evaluation on LoCoMo"
    )
    parser.add_argument("--data", default="locomo10.json", help="LoCoMo JSON file")
    parser.add_argument("--conversation", type=int, default=0, help="LoCoMo item index")
    parser.add_argument(
        "--all-conversations", action="store_true", help="Evaluate every LoCoMo item"
    )
    parser.add_argument("--output", default="graph_results.json", help="JSON result path")
    parser.add_argument("--model", default=MODEL_NAME, help="Hugging Face model id")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--max-questions", type=int, default=None, help="Optional per-item debug cap"
    )
    args = parser.parse_args()

    if args.max_hops < 1 or args.top_k < 1:
        parser.error("--max-hops and --top-k must be positive integers")

    data = load_locomo(args.data)
    if not data:
        parser.error("LoCoMo data contains no conversations")
    if not args.all_conversations and not 0 <= args.conversation < len(data):
        parser.error(f"--conversation must be between 0 and {len(data) - 1}")

    llm = LlamaModel(args.model, device_map=args.device_map)
    indices = range(len(data)) if args.all_conversations else [args.conversation]
    payload: List[Dict[str, Any]] = []
    for index in indices:
        item = data[index]
        questions = item.get("qa", [])
        if args.max_questions is not None:
            questions = questions[: args.max_questions]
        payload.append(
            {
                "sample_id": item.get("sample_id", index),
                "results": run_conversation(
                    llm,
                    item["conversation"],
                    questions,
                    max_hops=args.max_hops,
                    top_k=args.top_k,
                ),
            }
        )

    output: Any = payload if args.all_conversations else payload[0]["results"]
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
