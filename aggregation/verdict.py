"""
Aggregation layer: combine per-evidence NLI stance predictions into a single,
transparent verdict — the piece that makes this project different from a
black-box RAG answer.

Design (this is the part worth explaining well in an interview):

  1. Each retrieved evidence sentence has a stance (entailment / contradiction
     / neutral) and a confidence from the NLI classifier.
  2. We sum confidence-weighted "mass" per stance across all evidence —
     weighting by confidence means a low-confidence NLI call contributes
     less than a high-confidence one, instead of every piece of evidence
     voting equally regardless of how sure the classifier was.
  3. disagreement_index = 2 * min(support, refute) / (support + refute)
     when both are non-trivial, else 0. This is 0 when evidence is entirely
     one-sided and approaches 1 as support and refute mass become equal —
     i.e. it directly measures how split the evidence is, not just which
     side "won".
  4. Verdict logic checks disagreement BEFORE picking a winning side: a claim
     with strong evidence on both sides is labeled DISPUTED even if one side
     has a slightly higher raw mass, because reporting a confident verdict
     on genuinely contested evidence would be misleading.

This module has no model/network dependency — it's pure aggregation logic
over whatever (label, confidence) pairs the NLI stage produced, which makes
it fully unit-testable in isolation from the rest of the pipeline.
"""
from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    text: str
    stance: str          # "entailment" | "contradiction" | "neutral"
    confidence: float     # NLI classifier's confidence in `stance`, 0-1
    source_doc_id: str = ""


@dataclass
class Verdict:
    label: str                    # SUPPORTED | REFUTED | DISPUTED | NOT ENOUGH INFO
    confidence: float              # how dominant the winning stance is, 0-1
    disagreement_index: float      # 0 (one-sided) to 1 (evenly split), 0-1
    support_mass: float
    refute_mass: float
    neutral_mass: float
    evidence: list = field(default_factory=list)


# Below this, an NLI prediction is treated as too uncertain to count much —
# tune this against your validation set once you have real NLI confidences.
DISAGREEMENT_THRESHOLD = 0.35   # >= this -> evidence is contested, not one-sided
MIN_CONTESTED_MASS = 0.15        # both sides need at least this much mass to "count"


def aggregate(evidence_items: list[EvidenceItem]) -> Verdict:
    if not evidence_items:
        return Verdict(
            label="NOT ENOUGH INFO", confidence=0.0, disagreement_index=0.0,
            support_mass=0.0, refute_mass=0.0, neutral_mass=0.0, evidence=[],
        )

    support_mass = sum(e.confidence for e in evidence_items if e.stance == "entailment")
    refute_mass = sum(e.confidence for e in evidence_items if e.stance == "contradiction")
    neutral_mass = sum(e.confidence for e in evidence_items if e.stance == "neutral")
    total = support_mass + refute_mass + neutral_mass

    if total == 0:
        return Verdict(
            label="NOT ENOUGH INFO", confidence=0.0, disagreement_index=0.0,
            support_mass=0.0, refute_mass=0.0, neutral_mass=0.0, evidence=evidence_items,
        )

    contested_total = support_mass + refute_mass
    disagreement_index = (
        2 * min(support_mass, refute_mass) / contested_total if contested_total > 0 else 0.0
    )

    both_sides_material = (
        support_mass >= MIN_CONTESTED_MASS and refute_mass >= MIN_CONTESTED_MASS
    )

    if both_sides_material and disagreement_index >= DISAGREEMENT_THRESHOLD:
        label = "DISPUTED"
        confidence = 1.0 - disagreement_index  # less confident the more split it is
    elif support_mass > refute_mass and support_mass > neutral_mass:
        label = "SUPPORTED"
        confidence = support_mass / total
    elif refute_mass > support_mass and refute_mass > neutral_mass:
        label = "REFUTED"
        confidence = refute_mass / total
    else:
        label = "NOT ENOUGH INFO"
        confidence = neutral_mass / total

    return Verdict(
        label=label,
        confidence=round(confidence, 4),
        disagreement_index=round(disagreement_index, 4),
        support_mass=round(support_mass, 4),
        refute_mass=round(refute_mass, 4),
        neutral_mass=round(neutral_mass, 4),
        evidence=evidence_items,
    )
