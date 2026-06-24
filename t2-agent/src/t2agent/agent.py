"""Unified T2-Agent that produces BOTH outputs in one MCTS run.

For every ReMMDBench sample the agent:

1. Runs the three-subtask Monte-Carlo Tree Search (text / image / match).
2. Computes ``p_real`` and ``p_fake_*`` from the leaves' best confidence scores.
3. Bins ``p_real`` to a **5-way verdict** label
   (``True``, ``Mostly True``, ``Mixture``, ``Mostly False``, ``False``).
4. Asks the LLM for a **multi-label 8-way distortion taxonomy** prediction,
   conditioned on the verification trajectory.

Both outputs come from a single MCTS run, so all the LLM and tool calls are
shared between the two tasks. The agent also accepts XML-style responses with
a JSON fallback so different providers (GPT vs Qwen) can both be supported.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import RuntimeConfig
from .data import BenchmarkSample
from .labels import (
    REALMMDBENCH_TAXONOMY_LABELS,
    REALMMDBENCH_VERDICT_LABELS,
    format_label_list,
    normalize_taxonomy_label,
    normalize_taxonomy_labels,
    normalize_verdict_label,
)
from .llm import LLMClient, extract_json_object
from .logging_utils import CallLogger, get_default_logger
from .tools import ToolBox, ToolCall, available_actions_for_subtask


@dataclass(slots=True)
class TrajectoryStep:
    thought: str
    action: str
    observation: str
    st_score: float = 0.0
    sc_score: float = 0.0


@dataclass(slots=True)
class ActionNode:
    thought: str
    action: str
    depth: int
    visits: int = 0
    value: float = 0.0
    st_score: float = 0.0
    sc_score: float = 0.0
    observation: str = ""
    is_terminal: bool = False
    answer: str | None = None
    children: list["ActionNode"] = field(default_factory=list)
    expanded: bool = False
    trajectory_steps: list[TrajectoryStep] = field(default_factory=list)


@dataclass(slots=True)
class SubtaskNode:
    name: str
    initial_weight: float
    visits: int = 0
    value: float = 0.0
    completed: bool = False
    root_children: list[ActionNode] = field(default_factory=list)
    best_answer: str | None = None
    best_sc_score: float = 0.0
    best_value: float = 0.0
    best_trajectory: list[TrajectoryStep] = field(default_factory=list)
    failure_memory: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PredictionResult:
    """Combined 5-class verdict + 8-class taxonomy prediction."""

    benchmark_name: str
    sample_id: str
    predicted_verdict: str
    """Single 5-class verdict label, e.g. ``Mostly False``."""
    predicted_taxonomy: list[str]
    """Multi-label 8-class distortion taxonomy prediction."""
    predicted_label: str
    """Human-friendly summary string combining both predictions."""
    selected_tools: list[str]
    subtask_results: dict[str, dict[str, Any]]
    final_scores: dict[str, Any]
    taxonomy_rationale: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_xml_tag(text: str, tag: str) -> str | None:
    start_tag, end_tag = f"<{tag}>", f"</{tag}>"
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx:end_idx].strip()


def _extract_xml_list(text: str, tag: str) -> list[str]:
    start_tag, end_tag = f"<{tag}>", f"</{tag}>"
    results: list[str] = []
    cursor = 0
    while True:
        start_idx = text.find(start_tag, cursor)
        if start_idx == -1:
            break
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx == -1:
            break
        results.append(text[start_idx:end_idx].strip())
        cursor = end_idx + len(end_tag)
    return results


def _normalize_finish_label(label: str) -> str:
    return label.strip().upper().replace(" ", "_")


def _safe_float(text: str | None, default: float) -> float:
    if text is None:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _extract_score(text: str, pattern: str, default: float = 5.0) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    return float(match.group(1))


def _extract_float_list(text: str, default: tuple[float, ...] = (0.33, 0.33, 0.34)) -> list[float]:
    match = re.search(r"\[([^\]]+)\]", text)
    if match:
        try:
            return [float(item.strip()) for item in match.group(1).split(",")][:3]
        except ValueError:
            pass
    return list(default)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class T2Agent:
    """Unified T2-Agent producing both verdict and taxonomy predictions."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        logger: CallLogger | None = None,
        llm: LLMClient | None = None,
        toolbox: ToolBox | None = None,
    ) -> None:
        self.config = config
        self._logger = logger if logger is not None else get_default_logger()
        self.llm = llm if llm is not None else LLMClient(config, logger=self._logger)
        self.toolbox = (
            toolbox if toolbox is not None else ToolBox(config, self.llm, logger=self._logger)
        )

    # ------------------------------------------------------------------
    # Tool selection
    # ------------------------------------------------------------------

    def select_tools(self, benchmark_name: str) -> list[str]:
        toolkit = self.config.toolkits.get(benchmark_name.lower())
        if toolkit is None:
            toolkit = self.config.toolkits["realmmdbench"]
        selected = list(toolkit.base_tools)
        for tool_name in toolkit.candidate_tools:
            if tool_name not in selected:
                selected.append(tool_name)
        return selected

    # ------------------------------------------------------------------
    # Subtask initialization
    # ------------------------------------------------------------------

    def _initialization_prompt(self, sample: BenchmarkSample) -> str:
        return (
            "Task: Given a news text and a news image, infer the probability that the news "
            "belongs to the following three forgery types based on your experience:\n"
            "(1) Textual Veracity Distortion.\n"
            "(2) Visual Veracity Distortion.\n"
            "(3) Cross-modal Consistency Distortion.\n"
            "Strictly follow this XML-like output format:\n"
            "<probabilities>\n"
            "  <p1>probability of Textual Veracity Distortion (0.0 to 1.0)</p1>\n"
            "  <p2>probability of Visual Veracity Distortion (0.0 to 1.0)</p2>\n"
            "  <p3>probability of Cross-modal Consistency Distortion (0.0 to 1.0)</p3>\n"
            "</probabilities>\n\n"
            f"News text:\n{sample.text}"
        )

    def _build_subtask_nodes(self, sample: BenchmarkSample) -> list[SubtaskNode]:
        raw = self.llm.complete(
            "You estimate which forgery source is most relevant for verification.",
            self._initialization_prompt(sample),
            image_paths=[sample.image_paths[0]],
            purpose="agent.init",
        )
        p1_text = _extract_xml_tag(raw, "p1")
        p2_text = _extract_xml_tag(raw, "p2")
        p3_text = _extract_xml_tag(raw, "p3")
        if p1_text is None or p2_text is None or p3_text is None:
            probs = _extract_float_list(raw)
        else:
            probs = [
                _safe_float(p1_text, 0.33),
                _safe_float(p2_text, 0.33),
                _safe_float(p3_text, 0.34),
            ]
        return [
            SubtaskNode(name="text", initial_weight=probs[0], value=probs[0]),
            SubtaskNode(name="image", initial_weight=probs[1], value=probs[1]),
            SubtaskNode(name="match", initial_weight=probs[2], value=probs[2]),
        ]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_trajectory(
        self, sample: BenchmarkSample, trajectory: list[TrajectoryStep]
    ) -> float:
        serialized = "\n".join(
            f"Thought: {step.thought}\nAction: {step.action}\nObservation: {step.observation}"
            for step in trajectory
        )
        prompt = (
            "Task: Analyze Misinformation Detection Trajectories. Evaluate the trajectory's "
            "correctness based on environmental observations, thoughts, and actions.\n"
            "Strictly follow this XML-like output format:\n"
            "<correctness_score>integer from 1 to 10</correctness_score>\n\n"
            f"News:\n{sample.text}\n\nTrajectory:\n{serialized}"
        )
        output = self.llm.complete(
            "You evaluate reasoning trajectory quality for misinformation verification.",
            prompt,
            purpose="agent.score_trajectory",
        )
        raw_score = _extract_xml_tag(output, "correctness_score")
        if raw_score is None:
            raw_value = _extract_score(output, r"correctness score is\s+(\d+)", default=5.0)
        else:
            raw_value = _safe_float(raw_score, 5.0)
        return max(0.0, min(1.0, raw_value / self.config.search.score_scale))

    def _score_confidence(
        self,
        sample: BenchmarkSample,
        trajectory: list[TrajectoryStep],
        answer: str,
    ) -> float:
        observations = "\n".join(f"- {step.observation}" for step in trajectory)
        prompt = (
            "Task: Evaluate the reliability of the generated answer based on news, thoughts, "
            "and observations.\n"
            "Strictly follow this XML-like output format:\n"
            "<reliability_score>integer from 1 to 10</reliability_score>\n\n"
            f"News:\n{sample.text}\n\nObservations:\n{observations}\n\nAnswer:\n{answer}"
        )
        output = self.llm.complete(
            "You evaluate final answer reliability for misinformation verification.",
            prompt,
            purpose="agent.score_confidence",
        )
        raw_score = _extract_xml_tag(output, "reliability_score")
        if raw_score is None:
            raw_value = _extract_score(output, r"reliability score is\s+(\d+)", default=5.0)
        else:
            raw_value = _safe_float(raw_score, 5.0)
        return max(0.0, min(1.0, raw_value / self.config.search.score_scale))

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    def _build_planner_prompt(
        self,
        sample: BenchmarkSample,
        subtask: str,
        selected_tools: list[str],
        trajectory: list[TrajectoryStep],
        failure_memory: list[str],
    ) -> str:
        if subtask == "text":
            task_description = (
                "Task 1: Textual Veracity Detection. Use Wikipedia first and Google web "
                "search when Wikipedia is insufficient. Within 3 steps you MUST issue a "
                "Finish[TEXT_SUPPORT] or Finish[TEXT_REFUTE] action; do not keep searching "
                "forever. Commit to a verdict as soon as you have a reasonable amount of "
                "evidence, even if it is not perfect -- partial evidence is normal."
            )
            finish_token_a = "Finish[TEXT_SUPPORT]"
            finish_token_b = "Finish[TEXT_REFUTE]"
        elif subtask == "image":
            task_description = (
                "Task 2: Image Veracity Detection. Use Detect[image] and finish with "
                "IMAGE_SUPPORT or IMAGE_REFUTE."
            )
            finish_token_a = "Finish[IMAGE_SUPPORT]"
            finish_token_b = "Finish[IMAGE_REFUTE]"
        else:
            task_description = (
                "Task 3: Cross-modal Matching Detection. Use VQA and Entity if available, "
                "and finish with MATCH or MISMATCH."
            )
            finish_token_a = "Finish[MATCH]"
            finish_token_b = "Finish[MISMATCH]"
        actions = available_actions_for_subtask(subtask, selected_tools, sample.image_paths)
        serialized_trajectory = "\n".join(
            f"Thought: {step.thought}\nAction: {step.action}\nObservation: {step.observation}"
            for step in trajectory
        )
        failure_block = "\n".join(f"- {item}" for item in failure_memory[-3:]) or "- None"

        depth_hint = ""
        if subtask == "text" and len(trajectory) >= 2:
            depth_hint = (
                f"\n\n**CRITICAL REMINDER:** You have already taken {len(trajectory)} step(s). "
                f"At least ONE of your two candidate actions below MUST be exactly "
                f"`{finish_token_a}` or `{finish_token_b}`. Do not propose more search tools "
                f"as both candidates -- commit to a verdict now using whatever evidence you have."
            )

        return (
            f"{task_description}\n\nNews text:\n{sample.text}\n\n"
            "Available actions:\n- " + "\n- ".join(actions) + "\n\n"
            f"Previous trajectory:\n{serialized_trajectory or 'None'}\n\n"
            f"Failure memory:\n{failure_block}"
            f"{depth_hint}\n\n"
            "Generate exactly two candidate next actions.\n"
            "Strictly follow this XML-like output format:\n"
            "<candidates>\n"
            "  <candidate>\n"
            "    <thought>your thought here</thought>\n"
            "    <action>Tool[input] here</action>\n"
            "  </candidate>\n"
            "  <candidate>\n"
            "    <thought>your thought here</thought>\n"
            "    <action>Tool[input] here</action>\n"
            "  </candidate>\n"
            "</candidates>"
        )

    def _plan_actions(
        self,
        sample: BenchmarkSample,
        subtask: str,
        selected_tools: list[str],
        trajectory: list[TrajectoryStep],
        failure_memory: list[str],
    ) -> list[ActionNode]:
        prompt = self._build_planner_prompt(
            sample=sample,
            subtask=subtask,
            selected_tools=selected_tools,
            trajectory=trajectory,
            failure_memory=failure_memory,
        )
        raw = self.llm.complete(
            "You are the LVLM controller inside T2Agent. Use XML output exactly as instructed.",
            prompt,
            image_paths=[sample.image_paths[0]],
            purpose=f"agent.plan.{subtask}",
        )
        nodes: list[ActionNode] = []
        candidates = _extract_xml_list(raw, "candidate")
        if not candidates:
            try:
                payload = extract_json_object(raw)
            except ValueError:
                payload = {"candidates": []}
            for candidate in payload.get("candidates", []):
                thought = str(candidate.get("thought", "")).strip()
                action = str(candidate.get("action", "")).strip()
                if action:
                    nodes.append(
                        ActionNode(
                            thought=thought,
                            action=action,
                            depth=len(trajectory) + 1,
                        )
                    )
        else:
            for candidate_xml in candidates[: self.config.search.sampled_nodes]:
                thought = _extract_xml_tag(candidate_xml, "thought") or ""
                action = _extract_xml_tag(candidate_xml, "action") or ""
                if action:
                    nodes.append(
                        ActionNode(
                            thought=thought,
                            action=action,
                            depth=len(trajectory) + 1,
                        )
                    )

        if not nodes:
            failure_memory.append("Planner returned no parseable candidates; using fallback Finish action.")
            fallback_action = "Finish[TEXT_REFUTE]" if subtask == "text" else (
                "Finish[IMAGE_REFUTE]" if subtask == "image" else "Finish[MISMATCH]"
            )
            nodes.append(
                ActionNode(
                    thought="Fallback finish due to malformed planner output.",
                    action=fallback_action,
                    depth=len(trajectory) + 1,
                )
            )
        return nodes

    # ------------------------------------------------------------------
    # Action execution + MCTS
    # ------------------------------------------------------------------

    def _parse_action(self, action_text: str) -> tuple[str, str]:
        cleaned = action_text.strip().rstrip(")")
        match = re.match(r"([A-Za-z_]+)\[(.*)\]$", cleaned, re.DOTALL)
        if not match:
            match = re.match(r'([A-Za-z_]+)\("(.*)"\)$', action_text.strip(), re.DOTALL)
        if not match:
            match = re.match(r"([A-Za-z_]+)\(([^)]*)\)$", action_text.strip(), re.DOTALL)
        if not match:
            raise ValueError(f"Invalid action format: {action_text}")
        return match.group(1), match.group(2)

    def _execute_action(
        self,
        sample: BenchmarkSample,
        subtask: str,
        action_node: ActionNode,
    ) -> None:
        try:
            tool_name, tool_input = self._parse_action(action_node.action)
        except ValueError:
            action_node.observation = f"Could not parse action: {action_node.action}"
            return
        if tool_name.lower() == "finish":
            action_node.is_terminal = True
            action_node.answer = _normalize_finish_label(tool_input)
            action_node.observation = "Subtask finished."
            return
        try:
            result = self.toolbox.run(
                ToolCall(tool_name=tool_name, tool_input=tool_input),
                text=sample.text,
                image_paths=sample.image_paths,
            )
            action_node.observation = result.observation
        except Exception as exc:  # noqa: BLE001
            action_node.observation = f"Tool execution failed: {exc}"

    def _uct_score(self, parent_visits: int, child_value: float, child_visits: int) -> float:
        return (child_value / (child_visits + 1)) + (
            self.config.search.exploration_weight
            * math.sqrt(math.log(parent_visits + 1) / (child_visits + 1))
        )

    def _select_child(self, parent_visits: int, children: list[ActionNode]) -> ActionNode:
        return max(
            children,
            key=lambda child: self._uct_score(parent_visits, child.value, child.visits),
        )

    def _simulate_subtask(
        self,
        sample: BenchmarkSample,
        subtask_node: SubtaskNode,
        selected_tools: list[str],
    ) -> float:
        parent_children = subtask_node.root_children
        current_trajectory: list[TrajectoryStep] = []
        traversal: list[ActionNode] = []
        parent_visits = subtask_node.visits

        for _ in range(self.config.search.max_steps_per_rollout):
            if not parent_children:
                parent_children.extend(
                    self._plan_actions(
                        sample=sample,
                        subtask=subtask_node.name,
                        selected_tools=selected_tools,
                        trajectory=current_trajectory,
                        failure_memory=subtask_node.failure_memory,
                    )
                )
            if not parent_children:
                break
            action_node = self._select_child(parent_visits, parent_children)
            self._execute_action(sample, subtask_node.name, action_node)
            current_trajectory = [
                *current_trajectory,
                TrajectoryStep(
                    thought=action_node.thought,
                    action=action_node.action,
                    observation=action_node.observation,
                ),
            ]
            action_node.trajectory_steps = list(current_trajectory)
            action_node.st_score = self._score_trajectory(sample, current_trajectory)
            action_node.value = action_node.st_score
            traversal.append(action_node)

            if action_node.is_terminal:
                action_node.sc_score = self._score_confidence(
                    sample=sample,
                    trajectory=current_trajectory,
                    answer=action_node.answer or "UNKNOWN",
                )
                action_node.value = (
                    self.config.search.alpha * action_node.st_score
                    + (1 - self.config.search.alpha) * action_node.sc_score
                )
                break

            parent_children = action_node.children
            parent_visits = action_node.visits

        if not traversal:
            return 0.0

        leaf = traversal[-1]
        rollout_value = leaf.value
        if not leaf.is_terminal:
            subtask_node.failure_memory.append(
                f"Incomplete rollout on subtask {subtask_node.name} with value {rollout_value:.2f}"
            )

        for node in traversal:
            node.value = (node.value * node.visits + rollout_value) / (node.visits + 1)
            node.visits += 1

        subtask_node.value = (subtask_node.value * subtask_node.visits + rollout_value) / (
            subtask_node.visits + 1
        )
        subtask_node.visits += 1

        if leaf.is_terminal and leaf.answer is not None:
            if leaf.sc_score >= subtask_node.best_sc_score:
                subtask_node.best_answer = leaf.answer
                subtask_node.best_sc_score = leaf.sc_score
                subtask_node.best_value = leaf.value
                subtask_node.best_trajectory = list(leaf.trajectory_steps)
            if (
                leaf.answer.endswith("SUPPORT") or leaf.answer == "MATCH"
            ) and leaf.sc_score >= self.config.search.high_confidence_threshold:
                subtask_node.completed = True
        return rollout_value

    def _select_subtask(
        self, subtasks: list[SubtaskNode], root_visits: int
    ) -> SubtaskNode | None:
        open_subtasks = [subtask for subtask in subtasks if not subtask.completed]
        if not open_subtasks:
            return None
        return max(
            open_subtasks,
            key=lambda node: self._uct_score(
                root_visits,
                node.value if node.visits > 0 else node.initial_weight,
                node.visits,
            ),
        )

    # ------------------------------------------------------------------
    # Final scoring and dual prediction
    # ------------------------------------------------------------------

    def _fake_scores_from_subtasks(
        self, subtasks: list[SubtaskNode]
    ) -> dict[str, float]:
        p_fake: dict[str, float] = {}
        for subtask in subtasks:
            answer = subtask.best_answer or ""
            if answer in {"TEXT_REFUTE", "IMAGE_REFUTE", "MISMATCH"}:
                p_fake[subtask.name] = subtask.best_sc_score
            elif answer in {"TEXT_SUPPORT", "IMAGE_SUPPORT", "MATCH"}:
                p_fake[subtask.name] = 1 - subtask.best_sc_score
            else:
                p_fake[subtask.name] = 0.5
        non_neutral = [v for v in p_fake.values() if v != 0.5]
        if non_neutral:
            geometric = math.prod((1 - score) for score in p_fake.values())
            p_real = geometric ** (1 / max(len(p_fake), 1))
        else:
            p_real = 0.5
        return {
            "p_real": p_real,
            "p_fake_text": p_fake["text"],
            "p_fake_image": p_fake["image"],
            "p_fake_match": p_fake["match"],
        }

    def _verdict_label(self, final_scores: dict[str, float]) -> str:
        p_real = final_scores["p_real"]
        bins = self.config.evaluation.realmmdbench_probability_bins
        if p_real >= bins.get("true_min", 0.8):
            return "True"
        if p_real >= bins.get("mostly_true_min", 0.6):
            return "Mostly True"
        if p_real >= bins.get("mixture_min", 0.4):
            return "Mixture"
        if p_real >= bins.get("mostly_false_min", 0.2):
            return "Mostly False"
        return "False"

    def _serialize_evidence(self, subtasks: list[SubtaskNode]) -> str:
        blocks: list[str] = []
        for subtask in subtasks:
            observations = [step.observation for step in subtask.best_trajectory[-3:]]
            observation_block = "\n".join(f"  - {item}" for item in observations) or "  - None"
            blocks.append(
                f"{subtask.name}: answer={subtask.best_answer or 'UNKNOWN'}, "
                f"confidence={subtask.best_sc_score:.2f}\n{observation_block}"
            )
        return "\n\n".join(blocks)

    def _build_taxonomy_prompt(
        self,
        sample: BenchmarkSample,
        subtasks: list[SubtaskNode],
        final_scores: dict[str, float],
        verdict_label: str,
    ) -> str:
        taxonomy_definitions = "\n".join(
            [
                "- T1 Fabrication: text fabricates non-existent events, entities, data, or claims.",
                "- T2 Distortion: text distorts, exaggerates, or overstates a real fact.",
                "- T3 Misleading Context: text uses true elements in a wrong or misleading context.",
                "- V1 Synthetic Visual Content: visual material is AI-generated or otherwise synthetic.",
                "- V2 Visual Editing: visual material is edited, spliced, overlaid, or manipulated.",
                "- C1 Semantic Inconsistency: text and visuals conflict on the core semantic claim.",
                "- C2 Contextual Inconsistency: text and visuals come from mismatched time, place, event, or source context.",
                "- C3 Pragmatic Inconsistency: the pairing creates a misleading implication or stance without a direct factual conflict.",
            ]
        )
        return (
            "Task: classify this RealMMDBench sample into all applicable secondary "
            "distortion-taxonomy labels. This is a multi-label task: select zero, one, or "
            "multiple labels. Use exact label names only.\n\n"
            f"Allowed labels:\n{taxonomy_definitions}\n\n"
            f"News text:\n{sample.text}\n\n"
            f"Five-way verdict (already inferred from p_real): {verdict_label}\n"
            f"Modality scores: p_real={final_scores['p_real']:.3f}, "
            f"p_fake_text={final_scores['p_fake_text']:.3f}, "
            f"p_fake_image={final_scores['p_fake_image']:.3f}, "
            f"p_fake_match={final_scores['p_fake_match']:.3f}\n\n"
            "Verification evidence:\n"
            f"{self._serialize_evidence(subtasks)}\n\n"
            "Strictly follow this XML-like output format:\n"
            "<taxonomy_prediction>\n"
            "  <labels>\n"
            "    <selected_label>T2 Distortion</selected_label>\n"
            "    <selected_label>V2 Visual Editing</selected_label>\n"
            "  </labels>\n"
            "  <scores>\n"
            "    <score><label>T1 Fabrication</label><value>0.0</value></score>\n"
            "    <score><label>T2 Distortion</label><value>0.0</value></score>\n"
            "    <score><label>T3 Misleading Context</label><value>0.0</value></score>\n"
            "    <score><label>V1 Synthetic Visual Content</label><value>0.0</value></score>\n"
            "    <score><label>V2 Visual Editing</label><value>0.0</value></score>\n"
            "    <score><label>C1 Semantic Inconsistency</label><value>0.0</value></score>\n"
            "    <score><label>C2 Contextual Inconsistency</label><value>0.0</value></score>\n"
            "    <score><label>C3 Pragmatic Inconsistency</label><value>0.0</value></score>\n"
            "  </scores>\n"
            "  <rationale>one short sentence</rationale>\n"
            "</taxonomy_prediction>\n"
            "If no secondary distortion applies, leave <labels> empty."
        )

    def _fallback_taxonomy_labels(
        self, final_scores: dict[str, float]
    ) -> tuple[list[str], dict[str, float]]:
        scores = {label: 0.0 for label in REALMMDBENCH_TAXONOMY_LABELS}
        scores["T2 Distortion"] = final_scores["p_fake_text"]
        scores["V2 Visual Editing"] = final_scores["p_fake_image"]
        scores["C1 Semantic Inconsistency"] = final_scores["p_fake_match"]
        if final_scores["p_real"] >= self.config.evaluation.realmmdbench_probability_bins.get(
            "true_min", 0.8
        ):
            return [], scores
        labels = [
            label
            for label in REALMMDBENCH_TAXONOMY_LABELS
            if scores[label] >= 0.5
        ]
        return labels, scores

    def _parse_taxonomy_response(
        self, raw: str
    ) -> tuple[list[str], dict[str, float], str | None]:
        labels = normalize_taxonomy_labels(_extract_xml_list(raw, "selected_label"))
        scores = {label: 0.0 for label in REALMMDBENCH_TAXONOMY_LABELS}
        for score_xml in _extract_xml_list(raw, "score"):
            label_text = _extract_xml_tag(score_xml, "label")
            value_text = _extract_xml_tag(score_xml, "value")
            label = normalize_taxonomy_label(label_text or "")
            if label is None:
                continue
            try:
                score = float(value_text or "0")
            except (TypeError, ValueError):
                continue
            scores[label] = max(0.0, min(1.0, score))

        rationale = _extract_xml_tag(raw, "rationale")

        if not labels and not _extract_xml_list(raw, "selected_label"):
            try:
                payload = extract_json_object(raw)
            except ValueError:
                payload = {}
            json_labels = normalize_taxonomy_labels(payload.get("labels", []))
            if json_labels:
                labels = json_labels
            json_scores = payload.get("scores", {})
            if isinstance(json_scores, dict):
                for raw_label, raw_value in json_scores.items():
                    label = normalize_taxonomy_label(raw_label)
                    if label is None:
                        continue
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    scores[label] = max(0.0, min(1.0, value))
            if rationale is None:
                rationale = payload.get("rationale")

        if not labels and any(score >= 0.5 for score in scores.values()):
            labels = [
                label
                for label in REALMMDBENCH_TAXONOMY_LABELS
                if scores.get(label, 0.0) >= 0.5
            ]
        for label in labels:
            scores.setdefault(label, 1.0)
            if scores[label] < 0.5:
                scores[label] = max(scores[label], 0.51)

        return labels, scores, rationale

    def _predict_taxonomy(
        self,
        sample: BenchmarkSample,
        subtasks: list[SubtaskNode],
        final_scores: dict[str, float],
        verdict_label: str,
    ) -> tuple[list[str], dict[str, float], str | None]:
        prompt = self._build_taxonomy_prompt(sample, subtasks, final_scores, verdict_label)
        try:
            raw = self.llm.complete(
                "You are a strict multi-label RealMMDBench taxonomy classifier. "
                "Use XML output exactly as instructed.",
                prompt,
                image_paths=sample.image_paths[: max(1, len(sample.image_paths))],
                purpose="agent.taxonomy",
            )
        except Exception:  # noqa: BLE001
            labels, scores = self._fallback_taxonomy_labels(final_scores)
            return labels, scores, None

        labels, scores, rationale = self._parse_taxonomy_response(raw)
        if not labels and all(score == 0.0 for score in scores.values()):
            fallback_labels, fallback_scores = self._fallback_taxonomy_labels(final_scores)
            if fallback_labels:
                labels = fallback_labels
                scores = fallback_scores
        return labels, scores, rationale

    # ------------------------------------------------------------------
    # Top-level prediction loop
    # ------------------------------------------------------------------

    def predict(self, sample: BenchmarkSample) -> PredictionResult:
        selected_tools = self.select_tools(sample.benchmark_name)
        subtasks = self._build_subtask_nodes(sample)
        root_visits = 0
        for _ in range(self.config.search.simulations):
            selected_subtask = self._select_subtask(subtasks, root_visits)
            if selected_subtask is None:
                break
            try:
                self._simulate_subtask(sample, selected_subtask, selected_tools)
            except Exception as exc:  # noqa: BLE001
                selected_subtask.visits += 1
                selected_subtask.failure_memory.append(
                    f"Simulation failure: {type(exc).__name__}: {exc}"
                )
            root_visits += 1

        final_scores = self._fake_scores_from_subtasks(subtasks)
        verdict_label = self._verdict_label(final_scores)
        taxonomy_labels, taxonomy_scores, taxonomy_rationale = self._predict_taxonomy(
            sample, subtasks, final_scores, verdict_label
        )

        verdict_label = normalize_verdict_label(verdict_label) or verdict_label
        if verdict_label not in REALMMDBENCH_VERDICT_LABELS:
            verdict_label = "Mixture"

        summary_label = f"verdict={verdict_label} | taxonomy={format_label_list(taxonomy_labels)}"

        subtask_results = {
            subtask.name: {
                "initial_weight": subtask.initial_weight,
                "visits": subtask.visits,
                "completed": subtask.completed,
                "best_answer": subtask.best_answer,
                "best_sc_score": subtask.best_sc_score,
                "best_value": subtask.best_value,
                "best_trajectory": [asdict(step) for step in subtask.best_trajectory],
            }
            for subtask in subtasks
        }
        final_scores = {
            **final_scores,
            "verdict_label": verdict_label,
            "taxonomy_confidence": taxonomy_scores,
        }

        return PredictionResult(
            benchmark_name=sample.benchmark_name,
            sample_id=sample.sample_id,
            predicted_verdict=verdict_label,
            predicted_taxonomy=taxonomy_labels,
            predicted_label=summary_label,
            selected_tools=selected_tools,
            subtask_results=subtask_results,
            final_scores=final_scores,
            taxonomy_rationale=taxonomy_rationale,
            notes=[],
        )


# Re-export commonly used objects so callers do not have to import from tools.py
__all__ = [
    "ActionNode",
    "PredictionResult",
    "REALMMDBENCH_TAXONOMY_LABELS",
    "REALMMDBENCH_VERDICT_LABELS",
    "SubtaskNode",
    "T2Agent",
    "TrajectoryStep",
]
