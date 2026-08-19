"""
Market vitality scoring and barcode/dead market detection.
"""
from typing import Tuple


def vitality_grade_from_score(score: float) -> str:
    """
    Translates numeric vitality score (0..10) into letter grade (A..F).
    """
    if score >= 8.0:
        return "A"
    if score >= 6.0:
        return "B"
    if score >= 4.0:
        return "C"
    if score >= 2.0:
        return "D"
    return "F"


def compute_vitality_score(
    tpm: float,
    total_depth_usd: float,
    spread_pct: float,
    is_barcode: bool = False,
    min_slow_tpm: float = 3.0,
    min_ok_tpm: float = 15.0,
    min_good_tpm: float = 45.0,
    min_blazing_tpm: float = 120.0,
    min_thin_depth: float = 1000.0,
    min_ok_depth: float = 10000.0,
    min_good_depth: float = 50000.0,
) -> Tuple[float, str]:
    """
    Computes market vitality score (0-10) and grade (A/B/C/D/F).

    :param tpm: Trades per minute
    :param total_depth_usd: Total orderbook depth in USD within depth %
    :param spread_pct: Spread as % of price
    :param is_barcode: Flag indicating dead/barcode market
    :return: Tuple of (score, grade)
    """
    if is_barcode:
        return 0.0, "F"

    score = 0.0

    # Trade activity score (up to 4 points)
    if tpm >= min_blazing_tpm:
        score += 4.0
    elif tpm >= min_good_tpm:
        score += 3.0
    elif tpm >= min_ok_tpm:
        score += 2.0
    elif tpm >= min_slow_tpm:
        score += 1.0

    # Orderbook depth score (up to 3 points)
    if total_depth_usd >= min_good_depth:
        score += 3.0
    elif total_depth_usd >= min_ok_depth:
        score += 2.0
    elif total_depth_usd >= min_thin_depth:
        score += 1.0

    # Tightness of spread score (up to 3 points)
    if spread_pct < 0.1:
        score += 3.0
    elif spread_pct < 0.3:
        score += 2.0
    elif spread_pct < 1.0:
        score += 1.0

    score = max(0.0, min(10.0, score))
    grade = vitality_grade_from_score(score)
    return score, grade
