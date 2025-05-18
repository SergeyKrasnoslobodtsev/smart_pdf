import math
from typing import List, Tuple

import numpy as np

def cluster_coord(coords: List[int], eps: int = 5) -> List[int]:
    """Склеиваем координаты, отличающиеся < eps px."""
    if not coords:
        return []
    coords = sorted(coords)
    groups, cur = [], [coords[0]]
    for c in coords[1:]:
        if c - cur[-1] <= eps:
            cur.append(c)
        else:
            groups.append(int(np.mean(cur)))
            cur = [c]
    groups.append(int(np.mean(cur)))
    return groups

def line_in_bbox(mask: np.ndarray,
                 x0: int, y0: int, x1: int, y1: int,
                 axis: str, min_ratio: float = 0.1) -> bool:
    """
    Проверяем, проходит ли линия **внутри** bbox.
    axis='v' => оцениваем вертикаль x1; axis='h' => горизонталь y1.
    Берём полосу шириной/высотой 4 px с отступом 10 % от краёв —
    отсекает «усики» других линий и шум OCR.
    """
    
    if axis == "v":
        band = mask[y0 + (y1 - y0)//10 : y1 - (y1 - y0)//10, 
                    x1 - 1 : x1 + 3]                          
    else:
        band = mask[y1 - 1 : y1 + 3,
                    x0 + (x1 - x0)//10 : x1 - (x1 - x0)//10]
    return (band > 0).mean() >= min_ratio

def merge_intervals(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Объединяет пересекающиеся или смежные отрезки [y0,y1)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for curr in intervals[1:]:
        last = merged[-1]
        if curr[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], curr[1]))
        else:
            merged.append(curr)
    return merged

def merge_close_lines(lines, threshold=2):
    """Объединяет близкие линии в одну."""
    merged_lines = []
    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
        else:
            last_line = merged_lines[-1]
            if math.isclose(last_line, line, abs_tol=threshold):
                last_line = (last_line + line) / 2.0
                merged_lines[-1] = last_line
            else:
                merged_lines.append(line)
    return merged_lines

def bboxes_overlap(inner, outer) -> bool:
    """Проверка, вложен ли один прямоугольник в другой"""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return ox <= ix and oy <= iy and (ix + iw) <= (ox + ow) and (iy + ih) <= (oy + oh)
