"""Smoke test for CLIP probe scoring."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.clip_analyzer import score_frames, format_report
from src.probes.ant import PROBES


def test_clip_scores_ant():
    # 3 dummy frames (pure noise)
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
              for _ in range(3)]
    scores = score_frames(frames, PROBES, "ant")

    assert "ant" in scores
    assert "_summary" in scores
    assert "_flags" in scores
    assert "_hacking_warnings" in scores

    for cat in PROBES:
        assert cat in scores["ant"], f"Missing category: {cat}"
        assert cat in scores["_summary"], f"Missing summary for: {cat}"

    for cat, probe_results in scores["ant"].items():
        for text, data in probe_results.items():
            assert 0.0 <= data["score"] <= 1.0, \
                f"Score out of range: {data['score']} for '{text}'"

    report = format_report(scores, "ant")
    assert len(report) > 100, "Report too short"
    assert "ANT" in report.upper()

    print("  test_clip_scores_ant: PASS")
    print(f"  Report preview:\n{report[:400]}")


if __name__ == "__main__":
    print("Testing CLIP analyzer (downloads CLIP on first run ~340MB)...")
    try:
        test_clip_scores_ant()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
