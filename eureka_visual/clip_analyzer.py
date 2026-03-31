import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import HW

_model = None
_processor = None

def _load_clip():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = _model.to(HW["clip_device"])
        _model.eval()

def score_frames(
    frames: list,
    probes: dict,         # category → {probe_text: polarity}
    robot_type: str
) -> dict:
    """
    Returns a nested dict:
    {
      "posture": {
        "upright balanced torso": {"score": 0.81, "polarity": "positive"},
        ...
      },
      ...
      "_summary": {
        "posture": {"positive_avg": 0.76, "negative_avg": 0.22, "concern": False},
        ...
      },
      "_flags": ["reward_hacking: gait_symmetry", ...]
    }
    """
    _load_clip()
    device = HW["clip_device"]

    # Flatten all probe texts
    all_texts = []
    text_meta = []  # (category, text, polarity)
    for category, probe_dict in probes.items():
        for text, polarity in probe_dict.items():
            all_texts.append(text)
            text_meta.append((category, text, polarity))

    # Encode all frames
    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    inputs = _processor(
        text=all_texts,
        images=pil_frames,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        # logits_per_image: [n_frames, n_texts]
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()

    # Average scores across frames
    mean_scores = probs.mean(axis=0)  # [n_texts]

    # Build result structure
    results = {cat: {} for cat in probes}
    for i, (cat, text, polarity) in enumerate(text_meta):
        results[cat][text] = {
            "score": float(mean_scores[i]),
            "polarity": polarity
        }

    # Build per-category summary
    summary = {}
    for cat, probe_results in results.items():
        pos_scores = [v["score"] for v in probe_results.values()
                      if v["polarity"] == "positive"]
        neg_scores = [v["score"] for v in probe_results.values()
                      if v["polarity"] == "negative"]
        pos_avg = np.mean(pos_scores) if pos_scores else 0
        neg_avg = np.mean(neg_scores) if neg_scores else 0
        # Concern = negative probes scoring higher than positives
        concern = neg_avg > pos_avg
        summary[cat] = {
            "positive_avg": round(pos_avg, 3),
            "negative_avg": round(neg_avg, 3),
            "concern": concern
        }

    # Flag cross-signal contradictions (reward hacking detector)
    flags = []
    for cat, s in summary.items():
        if s["concern"]:
            # Find the worst offending negative probe
            worst = max(
                [(k, v["score"]) for k, v in results[cat].items()
                 if v["polarity"] == "negative"],
                key=lambda x: x[1], default=(None, 0)
            )
            if worst[0] and worst[1] > 0.5:
                flags.append(f"CONCERN [{cat}]: '{worst[0]}' score={worst[1]:.2f}")

    results["_summary"] = summary
    results["_flags"] = flags
    return results


def format_report(scores: dict, robot_type: str) -> str:
    """Converts score dict into structured text for the LLM."""
    lines = [f"== VISUAL ANALYSIS (CLIP probes, robot: {robot_type}) ==\n"]

    for category, probe_results in scores.items():
        if category.startswith("_"):
            continue
        summary = scores["_summary"][category]
        concern_marker = " [CONCERN]" if summary["concern"] else ""
        lines.append(f"{category.upper()}{concern_marker}")
        lines.append(f"  positive avg: {summary['positive_avg']:.2f}  "
                     f"negative avg: {summary['negative_avg']:.2f}")

        # Show top positive and top negative
        pos = [(k, v["score"]) for k, v in probe_results.items()
               if v["polarity"] == "positive"]
        neg = [(k, v["score"]) for k, v in probe_results.items()
               if v["polarity"] == "negative"]
        pos.sort(key=lambda x: -x[1])
        neg.sort(key=lambda x: -x[1])
        if pos:
            lines.append(f"  best positive: [{pos[0][1]:.2f}] {pos[0][0]}")
        if neg:
            lines.append(f"  worst negative: [{neg[0][1]:.2f}] {neg[0][0]}")
        lines.append("")

    if scores["_flags"]:
        lines.append("DETECTED ISSUES:")
        for flag in scores["_flags"]:
            lines.append(f"  {flag}")

    return "\n".join(lines)
