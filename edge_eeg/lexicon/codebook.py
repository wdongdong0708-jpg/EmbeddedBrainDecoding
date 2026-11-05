"""12-sentence Mandarin closed-vocabulary command codebook with Hamming distance analysis (Thesis Sec 2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np


@dataclass(frozen=True)
class CommandEntry:
    """Single 4-character Mandarin command codeword with tone & articulatory place fingerprints."""

    code_id: str
    domain: str
    text: str
    pinyin: str
    tones_1based: tuple[int, int, int, int]       # Mandarin tones 1..4 (after phonological sandhi calibration)
    consonants_1based: tuple[int, int, int, int]  # Articulatory places C1..C5 (1..5)
    note: str = ""

    @property
    def tones_0based(self) -> tuple[int, int, int, int]:
        return tuple(t - 1 for t in self.tones_1based)  # type: ignore[return-value]

    @property
    def consonants_0based(self) -> tuple[int, int, int, int]:
        return tuple(c - 1 for c in self.consonants_1based)  # type: ignore[return-value]

    @property
    def joint_vector_0based(self) -> np.ndarray:
        return np.array(self.tones_0based + self.consonants_0based, dtype=np.int32)


CONSONANT_PLACE_NAMES: Dict[int, str] = {
    1: "C1:双唇/唇齿音(b,p,m,f,w)",
    2: "C2:舌尖中音(d,t,n,l)",
    3: "C3:舌尖前/后擦音与塞擦音(z,c,s,zh,ch,sh,r)",
    4: "C4:舌面音(j,q,x,y)",
    5: "C5:舌根音(g,k,h)",
}

# 12 Golden 4-Character Mandarin Commands (Smart Wearable + Accessible Care)
CODEBOOK_12: List[CommandEntry] = [
    CommandEntry("S01", "智能穿戴", "接听电话", "jie1 ting1 dian4 hua4", (1, 1, 4, 4), (4, 2, 2, 5)),
    CommandEntry("S02", "智能穿戴", "打开降噪", "da3 kai1 jiang4 zao4", (3, 1, 4, 4), (2, 5, 4, 3)),
    CommandEntry("S03", "智能穿戴", "播放音乐", "bo1 fang4 yin1 yue4", (1, 4, 1, 4), (1, 1, 4, 4)),
    CommandEntry("S04", "智能穿戴", "暂停播放", "zan4 ting2 bo1 fang4", (4, 2, 1, 4), (3, 2, 1, 1)),
    CommandEntry("S05", "智能穿戴", "调高音量", "tiao2 gao1 yin1 liang4", (2, 1, 1, 4), (2, 5, 4, 2)),
    CommandEntry(
        "S06",
        "智能穿戴",
        "切下一首",
        "qie1 xia4 yi4 shou3",
        (1, 4, 4, 3),
        (4, 4, 4, 3),
        note="“一”在第三声“首”前连续变调读去声 yi4",
    ),
    CommandEntry("S07", "无障碍看护", "呼叫医生", "hu1 jiao4 yi1 sheng1", (1, 4, 1, 1), (5, 4, 4, 3)),
    CommandEntry(
        "S08",
        "无障碍看护",
        "我想喝水",
        "wo2 xiang3 he1 shui3",
        (2, 3, 1, 3),
        (1, 4, 5, 3),
        note="“我”在第三声“想”前触发上声变调读阳平 wo2",
    ),
    CommandEntry("S09", "无障碍看护", "帮我关灯", "bang1 wo3 guan1 deng1", (1, 3, 1, 1), (1, 1, 5, 2)),
    CommandEntry("S10", "无障碍看护", "关闭窗帘", "guan1 bi4 chuang1 lian2", (1, 4, 1, 2), (5, 1, 3, 2)),
    CommandEntry("S11", "无障碍看护", "调低空调", "tiao2 di1 kong1 tiao2", (2, 1, 1, 2), (2, 2, 5, 2)),
    CommandEntry("S12", "无障碍看护", "抬高床头", "tai2 gao1 chuang2 tou2", (2, 1, 2, 2), (2, 5, 3, 2)),
]

# 0-based numpy matrices: shape (12, 4)
CODEBOOK_TONES: np.ndarray = np.array([entry.tones_0based for entry in CODEBOOK_12], dtype=np.int32)
CODEBOOK_CONSONANTS: np.ndarray = np.array([entry.consonants_0based for entry in CODEBOOK_12], dtype=np.int32)

# 4-Sentence Golden Polarized Subset for Phase 4 Self-Collected N=1 EEG (S01, S04, S08, S12)
POLARIZED_SUBSET_4_INDICES: List[int] = [0, 3, 7, 11]


def compute_hamming_report(entries: Sequence[CommandEntry] | None = None) -> Dict[str, object]:
    """Compute pairwise Hamming distances across tones (4D), consonants (4D), and joint (8D) space."""
    items = list(entries) if entries is not None else CODEBOOK_12
    n = len(items)
    tones = np.array([e.tones_0based for e in items], dtype=np.int32)
    cons = np.array([e.consonants_0based for e in items], dtype=np.int32)

    tone_dists: List[int] = []
    cons_dists: List[int] = []
    joint_dists: List[int] = []
    min_pairs: List[tuple[str, str, int, int, int]] = []

    joint_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            d_t = int(np.sum(tones[i] != tones[j]))
            d_c = int(np.sum(cons[i] != cons[j]))
            d_j = d_t + d_c
            tone_dists.append(d_t)
            cons_dists.append(d_c)
            joint_dists.append(d_j)
            joint_matrix[i, j] = d_j
            joint_matrix[j, i] = d_j
            min_pairs.append((items[i].code_id, items[j].code_id, d_t, d_c, d_j))

    d_min_joint = int(min(joint_dists))
    min_pair_details = [p for p in min_pairs if p[4] == d_min_joint]

    return {
        "num_commands": n,
        "num_pairs": len(joint_dists),
        "tone_d_min": int(min(tone_dists)),
        "tone_d_avg": float(np.mean(tone_dists)),
        "cons_d_min": int(min(cons_dists)),
        "cons_d_avg": float(np.mean(cons_dists)),
        "joint_d_min": d_min_joint,
        "joint_d_max": int(max(joint_dists)),
        "joint_d_avg": float(np.mean(joint_dists)),
        "hard_error_correction_radius": (d_min_joint - 1) // 2,
        "min_distance_pairs": min_pair_details,
        "joint_distance_matrix": joint_matrix,
    }
