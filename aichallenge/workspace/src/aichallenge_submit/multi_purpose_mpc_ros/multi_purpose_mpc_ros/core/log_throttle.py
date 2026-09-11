"""Cheap pre-format log gating, independent of ROS logger call-site inspection."""
from time import monotonic


def periodic_log_due(owner, key, interval):
    now = monotonic()
    stamps = getattr(owner, '_periodic_log_stamps', None)
    if stamps is None:
        stamps = owner._periodic_log_stamps = {}
    previous = stamps.get(key)
    if previous is not None and 0 <= now - previous < interval:
        return False
    if key not in stamps and len(stamps) >= 64:
        del stamps[next(iter(stamps))]
    stamps[key] = now
    return True
