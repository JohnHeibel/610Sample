"""
Summarize an Nsight Compute report (.ncu-rep) into a per-kernel table.

Reads metrics by invoking ncu --import --csv --page raw and extracts:
  - duration (us)
  - achieved occupancy
  - SM busy %
  - Memory throughput %
  - DRAM throughput %
  - Tensor / FMA pipe utilization %
  - top bottleneck classification

The ncu raw page emits WIDE-format CSV: one row per kernel launch, with one
column per metric.

Usage:
  python summarize_ncu.py profile_output/v7_350M.ncu-rep
"""

import argparse
import csv
import io
import subprocess
import sys


NCU = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\ncu.bat"

# (column name in CSV, display label, formatter)
METRICS = [
    ('gpu__time_duration.sum',                                              'dur',     lambda v: f"{v*1000:.1f}us" if v < 1 else f"{v:.2f}ms"),
    ('sm__throughput.avg.pct_of_peak_sustained_elapsed',                    'SM%',     lambda v: f"{v:.1f}"),
    ('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',    'Mem%',    lambda v: f"{v:.1f}"),
    ('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',              'DRAM%',   lambda v: f"{v:.1f}"),
    ('sm__warps_active.avg.pct_of_peak_sustained_active',                   'Occup%',  lambda v: f"{v:.1f}"),
    ('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',     'Tensor%', lambda v: f"{v:.1f}"),
    ('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed',        'FMA%',    lambda v: f"{v:.1f}"),
    # --- detailed-set additions (silently dropped if absent in basic reports) ---
    ('l1tex__t_sector_hit_rate.pct',                                        'L1hit%',  lambda v: f"{v:.1f}"),
    ('lts__t_sector_hit_rate.pct',                                          'L2hit%',  lambda v: f"{v:.1f}"),
    ('l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed',
                                                                            'Shr%',    lambda v: f"{v:.1f}"),
]


def fetch_csv(report_path):
    cmd = [NCU, '--import', report_path, '--csv', '--page', 'raw']
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ncu --import failed:\n{e.output}", file=sys.stderr)
        sys.exit(1)


def parse(csv_text):
    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows:
        return [], []

    # Find header row — first row containing "Kernel Name".
    hdr_idx = next((i for i, r in enumerate(rows) if 'Kernel Name' in r), None)
    if hdr_idx is None:
        print("Could not find ncu CSV header (no 'Kernel Name' column).", file=sys.stderr)
        sys.exit(1)

    header = rows[hdr_idx]
    name_c = header.index('Kernel Name')
    # ncu emits a "units" row right after the header (empty kernel name, units like "ms", "%")
    body = [r for r in rows[hdr_idx + 1:]
            if r and len(r) == len(header) and r[name_c].strip()]
    return header, body


def short_name(full):
    base = full.split('(')[0]
    if '<' in base:
        base = base.split('<')[0]
    return base.split('::')[-1].strip()


def to_float(s):
    if s is None:
        return None
    s = s.replace(',', '').strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def classify(m):
    sm  = m.get('sm__throughput.avg.pct_of_peak_sustained_elapsed') or 0
    mem = m.get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed') or 0
    occ = m.get('sm__warps_active.avg.pct_of_peak_sustained_active') or 0
    if mem > 70 and mem > sm:
        return 'memory-bound'
    if sm > 70 and sm > mem:
        return 'compute-bound'
    if occ < 30:
        return 'latency-bound (low occ)'
    if max(sm, mem) < 40:
        return 'latency-bound'
    return 'balanced'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('report')
    args = ap.parse_args()

    header, body = parse(fetch_csv(args.report))
    if not body:
        print("No kernels found in report.", file=sys.stderr)
        sys.exit(1)

    name_c = header.index('Kernel Name')

    # Build per-row metric dict using header column index lookup
    col_idx = {col: header.index(col) for col, _, _ in METRICS if col in header}
    missing = [col for col, _, _ in METRICS if col not in header]
    if missing:
        print(f"Note: report missing {len(missing)} expected metric(s): "
              f"{', '.join(missing)}\n", file=sys.stderr)

    labels = ['kernel'] + [lbl for _, lbl, _ in METRICS] + ['bottleneck']
    table = []
    for r in body:
        kname = short_name(r[name_c])
        per_metric = {col: to_float(r[idx]) for col, idx in col_idx.items()}
        row = [kname]
        for col, _, fmt in METRICS:
            v = per_metric.get(col)
            row.append(fmt(v) if v is not None else '-')
        row.append(classify(per_metric))
        table.append(row)

    # Order by duration desc (hottest first)
    def dur_us(r):
        try:
            return float(r[1].replace('us', ''))
        except Exception:
            return 0.0
    table.sort(key=lambda r: -dur_us(r))

    widths = [max(len(str(c)) for c in [labels[i]] + [r[i] for r in table])
              for i in range(len(labels))]

    def fmt_row(r):
        return '  '.join(str(c).ljust(widths[i]) for i, c in enumerate(r))

    print(fmt_row(labels))
    print('  '.join('-' * w for w in widths))
    for r in table:
        print(fmt_row(r))


if __name__ == '__main__':
    main()
