import gc
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.config import HW, CLIP_MODEL_ID

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[clip] Loading {CLIP_MODEL_ID} on {HW['clip_device']}...")
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model = _clip_model.to(HW["clip_device"])
        _clip_model.eval()
        print("[clip] Model loaded.")
    return _clip_model, _clip_processor


def unload_clip():
    """
    Free CLIP from GPU memory before LLM inference.
    Call after score_frames() / format_report(), before calling Ollama.
    On 8GB VRAM: CLIP ~0.6 GB + llama3.1:8b-q4 ~5 GB — must not overlap.
    """
    global _clip_model, _clip_processor
    _clip_model = None
    _clip_processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_frames(frames: list, probes: dict, robot_type: str) -> dict:
    """
    Scores frames against probe sentences using CLIP cosine similarity
    expressed as softmax probabilities over all probe texts.

    Args:
        frames:     list of np.ndarray (H, W, 3) uint8
        probes:     probe bank dict from src/probes/{robot}.py
        robot_type: "humanoid" or "ant"

    Scores are softmax probabilities over ALL probe texts (dim=-1 over texts).
    This means scores sum to 1.0 across all probes. Baseline per probe ≈ 1/N.
    With ~22-24 probes, baseline ≈ 0.04. CONCERN_THRESHOLD=0.45 → ~10x baseline.
    Scores are averaged across all extracted frames.

    Returns:
    {
        robot_type: {
            category: { probe_text: {"score": float, "polarity": str}, ... },
            ...
        },
        "_summary": {
            category: {
                "positive_avg": float, "negative_avg": float,
                "concern": bool,
                "top_positive": (text, score),
                "top_negative": (text, score),
            }, ...
        },
        "_flags": [ "CONCERN [cat]: 'text' score=X.XX", ... ],
        "_hacking_warnings": [ "POSSIBLE HACKING [cat]: ...", ... ],
    }
    """
    model, processor = _load_clip()
    device = HW["clip_device"]

    # Flatten all probe texts with metadata
    all_texts = []
    text_meta = []  # (category, text, polarity)
    for category, probe_dict in probes.items():
        for text, polarity in probe_dict.items():
            all_texts.append(text)
            text_meta.append((category, text, polarity))

    if not all_texts or not frames:
        return {"_summary": {}, "_flags": [], "_hacking_warnings": []}

    # Convert frames to PIL
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            pil_frames.append(Image.fromarray(f.astype(np.uint8)))
        else:
            pil_frames.append(f)

    # Single forward pass: all frames × all texts
    inputs = processor(
        text=all_texts,
        images=pil_frames,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image: [n_frames, n_texts]
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()

    # Average probabilities across frames → [n_texts]
    mean_scores = probs.mean(axis=0)

    # Build per-category results
    results = {robot_type: {cat: {} for cat in probes}}
    for i, (cat, text, polarity) in enumerate(text_meta):
        results[robot_type][cat][text] = {
            "score": float(mean_scores[i]),
            "polarity": polarity,
        }

    # Per-category summaries
    summary = {}
    for cat, probe_results in results[robot_type].items():
        pos_items = [(k, v["score"]) for k, v in probe_results.items() if v["polarity"] == "positive"]
        neg_items = [(k, v["score"]) for k, v in probe_results.items() if v["polarity"] == "negative"]

        pos_avg = float(np.mean([s for _, s in pos_items])) if pos_items else 0.0
        neg_avg = float(np.mean([s for _, s in neg_items])) if neg_items else 0.0

        top_pos = max(pos_items, key=lambda x: x[1]) if pos_items else ("none", 0.0)
        top_neg = max(neg_items, key=lambda x: x[1]) if neg_items else ("none", 0.0)

        summary[cat] = {
            "positive_avg": round(pos_avg, 4),
            "negative_avg": round(neg_avg, 4),
            "concern": neg_avg > pos_avg,
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

    # Load thresholds from probe module
    from src.probes import humanoid as h_probes, ant as a_probes
    probe_module = h_probes if robot_type == "humanoid" else a_probes
    concern_threshold = probe_module.CONCERN_THRESHOLD
    hacking_threshold = probe_module.HACKING_THRESHOLD

    flags, hacking_warnings = [], []
    for cat, s in summary.items():
        top_neg_text, top_neg_score = s["top_negative"]
        if top_neg_score > concern_threshold:
            flags.append(f"CONCERN [{cat}]: '{top_neg_text}' score={top_neg_score:.2f}")
        delta = s["negative_avg"] - s["positive_avg"]
        if delta > hacking_threshold:
            hacking_warnings.append(
                f"POSSIBLE HACKING [{cat}]: "
                f"negative_avg={s['negative_avg']:.2f} >> "
                f"positive_avg={s['positive_avg']:.2f} (delta={delta:.2f})"
            )

    results["_summary"] = summary
    results["_flags"] = flags
    results["_hacking_warnings"] = hacking_warnings
    return results


def format_report(scores: dict, robot_type: str) -> str:
    """
    Converts CLIP score dict into a structured text block for LLM consumption.
    Categories map to reward component names so the LLM can make targeted edits.
    """
    lines = [f"== VISUAL ANALYSIS REPORT (CLIP, robot: {robot_type}) ==\n"]

    robot_scores = scores.get(robot_type, {})
    summary = scores.get("_summary", {})

    for category, probe_results in robot_scores.items():
        s = summary.get(category, {})
        concern_str = " *** CONCERN ***" if s.get("concern", False) else ""
        lines.append(f"[{category.upper()}]{concern_str}")
        lines.append(
            f"  positive_avg={s.get('positive_avg', 0):.3f}  "
            f"negative_avg={s.get('negative_avg', 0):.3f}"
        )
        top_pos = s.get("top_positive", ("n/a", 0))
        top_neg = s.get("top_negative", ("n/a", 0))
        lines.append(f"  strongest positive: [{top_pos[1]:.3f}] {top_pos[0]}")
        lines.append(f"  strongest negative: [{top_neg[1]:.3f}] {top_neg[0]}")
        lines.append("")

    flags    = scores.get("_flags", [])
    hacking  = scores.get("_hacking_warnings", [])

    if flags or hacking:
        lines.append("DETECTED BEHAVIORAL ISSUES:")
        for f in flags:
            lines.append(f"  {f}")
        for h in hacking:
            lines.append(f"  {h}")
        lines.append("")
    else:
        lines.append("No critical behavioral issues detected.")

    return "\n".join(lines)
