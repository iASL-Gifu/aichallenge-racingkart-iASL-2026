#!/usr/bin/env python3
"""Run on the simulator HOST, separately from ROS. JSONL CPU/GPU telemetry."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def gpu_command(arguments):
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return {'error': 'nvidia-smi unavailable'}
    try:
        result = subprocess.run([executable]+arguments, capture_output=True,
                                text=True, timeout=3, check=False)
        return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'error': str(error)}


def processes():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            end = text.rindex(')')
            fields = text[end+2:].split()
            result[path.parent.name] = (
                text[text.index('(')+1:end], int(fields[11])+int(fields[12]),
                int(fields[19]), int(fields[21]))
        except (OSError, ValueError, IndexError):
            continue
    return result


def process_load(previous, current, elapsed):
    ticks = os.sysconf('SC_CLK_TCK')
    page_size = os.sysconf('SC_PAGE_SIZE')
    rows = []
    for pid, (name, cpu, born, rss) in current.items():
        old = previous.get(pid)
        if old is None or old[2] != born or elapsed <= 0:
            continue
        rows.append({'pid': int(pid), 'name': name,
                     'cpu_pct': round(100*max(0, cpu-old[1])/ticks/elapsed, 1),
                     'rss_mib': round(max(0, rss)*page_size/(1024**2), 1)})
    return sorted(rows, key=lambda r: r['cpu_pct'], reverse=True)[:15]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=600.)
    parser.add_argument('--interval', type=float, default=5.)
    args = parser.parse_args()
    if args.interval < 2 or args.duration < 0:
        parser.error('interval must be >=2 seconds and duration >=0')
    previous = processes()
    last = start = time.monotonic()
    # Append to avoid destroying a previous run's measurements.
    with open(args.output, 'a', buffering=1) as output:
        while time.monotonic()-start < args.duration:
            time.sleep(min(args.interval, max(0, args.duration-(time.monotonic()-start))))
            now = time.monotonic()
            current = processes()
            row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'window_sec': round(now-last, 3), 'loadavg': os.getloadavg(),
                   'cpu_count': os.cpu_count(), 'top_cpu': process_load(previous, current, now-last)}
            for name, path in [('cpu_pressure', '/proc/pressure/cpu'),
                               ('memory_pressure', '/proc/pressure/memory')]:
                try:
                    row[name] = Path(path).read_text().strip()
                except OSError as error:
                    row[name] = str(error)
            row['gpu'] = gpu_command([
                '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,clocks.current.graphics,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'])
            row['gpu_processes'] = gpu_command(['pmon', '-c', '1', '-s', 'um'])
            output.write(json.dumps(row, separators=(',', ':'))+'\n')
            previous, last = current, now


if __name__ == '__main__':
    main()
