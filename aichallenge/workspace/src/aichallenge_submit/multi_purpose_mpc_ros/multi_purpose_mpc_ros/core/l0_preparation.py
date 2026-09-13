"""Distance-based approach to L0 priority zones on the circular Center path."""
import math


def approaching_zone(wp, zones, segment_lengths, preview):
    count = len(segment_lengths)
    if not count or not math.isfinite(preview) or preview <= 0:
        return False
    wp = int(wp) % count
    for start, _ in zones:
        distance = 0.
        index = wp
        for _ in range(count):
            if index == int(start) % count:
                if distance <= preview:
                    return True
                break
            length = float(segment_lengths[index])
            if not math.isfinite(length) or length <= 0:
                break
            distance += length
            if distance > preview:
                break
            index = (index+1) % count
    return False


def l2_exception(target_id, slow, passable, conflicts):
    """Keep a feasible slow-car pass; presence of a slow car alone is insufficient."""
    return bool(target_id is not None and slow and passable
                and not any(conflicts.get(key) for key in ('front', 'side', 'rear')))
