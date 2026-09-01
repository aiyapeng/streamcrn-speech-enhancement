"""
标准语音增强评测指标：PESQ(宽带)、STOI、SI-SDR。
这些是"语音前端/降噪"岗位与论文通用的达标口径。
"""
import numpy as np

try:
    from pesq import pesq as _pesq
    _HAS_PESQ = True
except Exception:
    _HAS_PESQ = False

try:
    from pystoi import stoi as _stoi
    _HAS_STOI = True
except Exception:
    _HAS_STOI = False


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def pesq_wb(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> float:
    if not _HAS_PESQ:
        return float("nan")
    try:
        return float(_pesq(sr, ref, est, "wb"))
    except Exception:
        return float("nan")


def stoi_score(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> float:
    if not _HAS_STOI:
        return float("nan")
    try:
        return float(_stoi(ref, est, sr, extended=False))
    except Exception:
        return float("nan")


def evaluate_pair(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> dict:
    """est/ref: 1D numpy。返回 PESQ/STOI/SI-SDR。自动对齐长度。"""
    L = min(len(est), len(ref))
    est, ref = est[:L].astype(np.float64), ref[:L].astype(np.float64)
    return {
        "pesq": pesq_wb(est, ref, sr),
        "stoi": stoi_score(est, ref, sr),
        "sisdr": si_sdr(est, ref),
    }
