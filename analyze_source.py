"""
Aggregate ncu source-page CSV by user-source line, ranked by stall count.

Per-section header (one section per source file):
  File Path, Function Name, then a CSV header row, then alternating
  source-line rows (with "Line No" populated) and SASS rows
  (Line No empty, Address populated).

We aggregate the SASS rows under their preceding source-line row.
"""

import csv
import sys
from collections import defaultdict


# Stall reason columns (issued + not-issued summed)
STALL_COLS = [
    'stall_lg', 'stall_long_sb', 'stall_short_sb', 'stall_wait',
    'stall_math', 'stall_barrier', 'stall_mio', 'stall_branch_resolving',
    'stall_dispatch', 'stall_drain', 'stall_membar', 'stall_no_inst',
    'stall_not_selected', 'stall_selected', 'stall_misc',
]
INTEREST_COLS = [
    '# Samples', 'Instructions Executed',
    'L1 Conflicts Shared N-Way',
    'L1 Wavefronts Shared', 'L1 Wavefronts Shared Ideal',
    'L1 Wavefronts Shared Excessive',
    'L2 Theoretical Sectors Global', 'L2 Theoretical Sectors Global Ideal',
    'L2 Theoretical Sectors Global Excessive',
]


def to_int(s):
    if s is None or s == '' or s == '-':
        return 0
    try:
        return int(s.replace(',', ''))
    except ValueError:
        try:
            return int(float(s.replace(',', '')))
        except ValueError:
            return 0


def parse_sections(path):
    """Yield (file_path, function_name, header, body_rows) per section."""
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        section = None
        rows = []
        file_path = func = header = None
        for r in reader:
            if not r:
                continue
            if r[0] == 'File Path':
                if file_path:
                    yield file_path, func, header, rows
                file_path = r[1]
                func = None
                header = None
                rows = []
                continue
            if r[0] == 'Function Name':
                func = r[1]
                continue
            if r[0] == 'Line No' and 'Address' in r:
                header = r
                continue
            if header:
                rows.append(r)
        if file_path:
            yield file_path, func, header, rows


def aggregate(file_path, header, rows, source_only_lines=False):
    """Return list of (line_no, source_text, agg_metrics_dict)."""
    col = {name: header.index(name) for name in header}

    out = []  # list of [line_no, source, metrics_dict]
    cur = None  # current source-line bucket

    for r in rows:
        if len(r) <= max(col.values()):
            continue
        line_no = r[col['Line No']].strip()
        if line_no:  # source-line row
            cur = {
                'line_no': int(line_no) if line_no.isdigit() else line_no,
                'source': r[col['Source']].strip(),
                'metrics': defaultdict(int),
            }
            out.append(cur)
            # The source-row itself carries aggregate metrics for that line.
            # Those aggregates already include all SASS instructions emitted
            # for the line, so we only consume the source row and skip SASS.
            for name in STALL_COLS:
                v = to_int(r[col[name]]) if name in col else 0
                ni = to_int(r[col.get(name + ' (Not Issued)', -1)]) \
                    if name + ' (Not Issued)' in col else 0
                cur['metrics'][name] += v + ni
            for name in INTEREST_COLS:
                if name in col:
                    cur['metrics'][name] += to_int(r[col[name]])
        # Skip SASS rows; the aggregates on the source-row above already
        # include them.

    return out


def main():
    if len(sys.argv) < 2:
        print('usage: analyze_source.py <source.csv> [--file=substring]')
        sys.exit(1)
    path = sys.argv[1]
    file_filter = None
    for a in sys.argv[2:]:
        if a.startswith('--file='):
            file_filter = a.split('=', 1)[1]

    all_lines = []
    for file_path, func, header, rows in parse_sections(path):
        if file_filter and file_filter not in file_path:
            continue
        if header is None:
            continue
        agg = aggregate(file_path, header, rows)
        for entry in agg:
            entry['file'] = file_path
            entry['func'] = func
            all_lines.append(entry)

    # Total stalls per line
    for e in all_lines:
        e['total_stalls'] = sum(e['metrics'][c] for c in STALL_COLS)
        wf = e['metrics'].get('L1 Wavefronts Shared', 0)
        wf_ideal = e['metrics'].get('L1 Wavefronts Shared Ideal', 0)
        e['shared_blowup'] = (wf / wf_ideal) if wf_ideal else 0

    grand_stalls = sum(e['total_stalls'] for e in all_lines) or 1

    print(f'\n=== TOP 15 LINES BY STALL COUNT ({len(all_lines)} lines total, '
          f'grand total {grand_stalls:,} stall samples) ===\n')
    by_stalls = sorted(all_lines, key=lambda e: -e['total_stalls'])[:15]
    for e in by_stalls:
        pct = 100 * e['total_stalls'] / grand_stalls
        m = e['metrics']
        wf = m.get('L1 Wavefronts Shared', 0)
        wfi = m.get('L1 Wavefronts Shared Ideal', 0)
        wfx = m.get('L1 Wavefronts Shared Excessive', 0)
        srcfile = e['file'].rsplit('/', 1)[-1]
        print(f'{srcfile}:{e["line_no"]:>5}  stalls={e["total_stalls"]:>10,}  '
              f'({pct:>4.1f}%)  shared_wf={wf:>12,}  ideal={wfi:>12,}  '
              f'excess={wfx:>12,}')
        src = e['source'][:120]
        print(f'        > {src}')
        # Top 3 stall reasons for this line
        top_reasons = sorted(
            ((c, e['metrics'][c]) for c in STALL_COLS if e['metrics'][c]),
            key=lambda x: -x[1])[:3]
        rs = '  '.join(f'{r}={v:,}' for r, v in top_reasons)
        print(f'          stalls: {rs}')
        print()

    print(f'\n=== TOP 10 LINES BY SHARED-MEM CONFLICT EXCESS ===\n')
    by_excess = sorted(all_lines,
                       key=lambda e: -e['metrics'].get('L1 Wavefronts Shared Excessive', 0))[:10]
    for e in by_excess:
        m = e['metrics']
        wf = m.get('L1 Wavefronts Shared', 0)
        wfi = m.get('L1 Wavefronts Shared Ideal', 0)
        wfx = m.get('L1 Wavefronts Shared Excessive', 0)
        if wfx == 0:
            continue
        ratio = wf / wfi if wfi else 0
        srcfile = e['file'].rsplit('/', 1)[-1]
        print(f'{srcfile}:{e["line_no"]:>5}  excess_wf={wfx:>12,}  '
              f'ratio={ratio:.2f}x  ({wf:,} / {wfi:,})')
        print(f'        > {e["source"][:120]}')
        print()


if __name__ == '__main__':
    main()
