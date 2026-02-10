#!/usr/bin/env python3
"""
Generate an IEEE-styled pie chart of kernel execution time distribution by layer
for the ACE2 trace, reading directly from the correlated SQLite database.

Usage:
    python3 generate_pie_chart.py                          # uses default db path
    python3 generate_pie_chart.py --db path/to/correlated.sqlite
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType for IEEE
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import sqlite3
import argparse


def extract_layer_data(db_path: str):
    """
    Extract per-layer average kernel execution time from the correlated database.
    Mirrors the logic in nsys_trace_layer_analysis.py:analyze_nvtx_ranges_with_cuda_timing().
    Excludes the first occurrence of each range (startup outlier).
    Returns list of (layer_name, avg_kernel_time_ms, first_occurrence_time).

    Optimised: pre-builds a correlation-ID -> kernel-duration lookup and
    a (start,end) interval list for the CUDA runtime table, so we don't
    issue thousands of tiny queries against a 1.6 GB database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ---- 1. Get all unique NVTX ranges ----
    print("  Querying NVTX ranges ...")
    cursor.execute("""
        WITH domains AS (
            SELECT
                ne.domainId AS id,
                ne.globalTid AS globalTid,
                COALESCE(sid.value, ne.text) AS name
            FROM NVTX_EVENTS AS ne
            LEFT JOIN StringIds AS sid ON ne.textId = sid.id
            WHERE ne.eventType = 75
            GROUP BY ne.domainId, ne.globalTid
        )
        SELECT DISTINCT
            COALESCE(d.name, '') || ':' || COALESCE(sid.value, ne.text, '') AS fullname
        FROM NVTX_EVENTS AS ne
        LEFT JOIN domains AS d
            ON ne.domainId = d.id
            AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
        LEFT JOIN StringIds AS sid ON ne.textId = sid.id
        WHERE ne.eventType = 59 OR ne.eventType = 70
        ORDER BY fullname;
    """)
    nvtx_ranges = cursor.fetchall()

    # Filter (same rules as the main script)
    filtered = [r for r in nvtx_ranges
                if "inference" not in r[0].lower()
                and not r[0].lower().startswith(":warmup")
                and r[0] != ":fcn3_profiling"
                and r[0] != ":Global convolution"
                and r[0] != ":forward_pass"]
    print(f"  {len(filtered)} ranges after filtering")

    # ---- 2. Bulk-load kernel durations keyed by correlationId ----
    print("  Loading kernel duration lookup (this may take a moment) ...")
    cursor.execute("""
        SELECT correlationId, (end - start) as duration
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE correlationId IS NOT NULL;
    """)
    kernel_dur = {}
    for corr_id, dur in cursor:
        kernel_dur[corr_id] = dur
    print(f"  Loaded {len(kernel_dur)} kernel entries")

    # ---- 3. Bulk-load CUDA runtime calls (start, end, correlationId) ----
    print("  Loading CUDA runtime calls ...")
    cursor.execute("""
        SELECT start, end, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        ORDER BY start;
    """)
    runtime_calls = cursor.fetchall()            # sorted by start
    rt_starts = [r[0] for r in runtime_calls]    # for bisect
    print(f"  Loaded {len(runtime_calls)} runtime calls")

    import bisect

    def kernel_time_in_interval(interval_start, interval_end):
        """Sum kernel durations for runtime calls within [interval_start, interval_end]."""
        lo = bisect.bisect_left(rt_starts, interval_start)
        total = 0
        for idx in range(lo, len(runtime_calls)):
            rt_start, rt_end, corr_id = runtime_calls[idx]
            if rt_start > interval_end:
                break
            if rt_end <= interval_end and corr_id is not None:
                dur = kernel_dur.get(corr_id)
                if dur is not None:
                    total += dur
        return total

    # ---- 4. Per-range: get occurrences, skip first, sum kernel time ----
    results = []
    for i, (range_name,) in enumerate(filtered):
        print(f"  [{i+1}/{len(filtered)}] {range_name} ...", end=" ", flush=True)
        cursor.execute("""
            WITH domains AS (
                SELECT
                    ne.domainId AS id,
                    ne.globalTid AS globalTid,
                    COALESCE(sid.value, ne.text) AS name
                FROM NVTX_EVENTS AS ne
                LEFT JOIN StringIds AS sid ON ne.textId = sid.id
                WHERE ne.eventType = 75
                GROUP BY ne.domainId, ne.globalTid
            )
            SELECT ne.start, ne.end
            FROM NVTX_EVENTS AS ne
            LEFT JOIN domains AS d
                ON ne.domainId = d.id
                AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
            LEFT JOIN StringIds AS sid ON ne.textId = sid.id
            WHERE (ne.eventType = 59 OR ne.eventType = 70)
              AND (COALESCE(d.name, '') || ':' || COALESCE(sid.value, ne.text, '')) = ?
            ORDER BY ne.start;
        """, (range_name,))
        range_times = cursor.fetchall()

        if not range_times or len(range_times) < 2:
            print("skipped (too few occurrences)")
            continue

        first_occurrence_start = range_times[0][0]
        # Skip first occurrence (startup outlier)
        range_times_filtered = range_times[1:]
        total_occurrences = len(range_times_filtered)

        total_kernel_time = 0
        for start_time, end_time in range_times_filtered:
            total_kernel_time += kernel_time_in_interval(start_time, end_time)

        avg_kernel_time_ms = (total_kernel_time / total_occurrences) / 1e6
        print(f"{avg_kernel_time_ms:.2f} ms")
        results.append((range_name, avg_kernel_time_ms, first_occurrence_start))

    conn.close()

    # Sort by first occurrence time
    results.sort(key=lambda x: x[2])
    return results


def generate_pie_chart(layer_data, output_dir='ace2_nvtx_400_50_layer_graphs'):
    """Generate IEEE-styled pie chart from (name, time_ms, _) tuples.
    Visually distinguishes SFNO core layers from Encoder/Decoder."""
    from matplotlib.patches import Patch
    import matplotlib.patheffects as pe

    os.makedirs(output_dir, exist_ok=True)

    # Clean layer names (strip leading ":" )
    raw_names = [item[0].lstrip(':').strip() for item in layer_data]
    raw_times = [item[1] for item in layer_data]

    # Filter out layers with 0 time (e.g. Dropout)
    filtered = [(n, t) for n, t in zip(raw_names, raw_times) if t > 0.001]
    raw_names = [x[0] for x in filtered]
    raw_times = [x[1] for x in filtered]

    # ---- Group layers ----
    # Spectral Filter Layer = RealSHT + dhconv + InverseSHT
    SPECTRAL = {'RealSHT', 'dhconv', 'InverseSHT'}
    # Normalization = Norm 0 + Norm 1
    NORM = {'Norm 0', 'Norm 1'}
    # Skip Connections = Inner Skip + Outer Skip
    SKIP = {'Inner Skip', 'Outer Skip'}

    groups = {
        'Spectral Filter': {'members': SPECTRAL, 'time': 0.0},
        'Normalization':    {'members': NORM,     'time': 0.0},
        'Skip Connections': {'members': SKIP,     'time': 0.0},
    }

    names = []
    times = []
    for n, t in zip(raw_names, raw_times):
        merged = False
        for group_name, info in groups.items():
            if n in info['members']:
                info['time'] += t
                merged = True
                break
        if not merged:
            names.append(n)
            times.append(t)

    # Collect ungrouped layers
    ungrouped = {}
    for n, t in zip(raw_names, raw_times):
        is_grouped = any(n in info['members'] for info in groups.values())
        if not is_grouped:
            ungrouped[n] = t

    # Explicit ordering: Spectral Filter (big) first, then smaller SFNO layers,
    # then Encoder/Decoder at the end.
    # This keeps Normalization near Activation/MLP/Skip, not between Spectral Filter
    # and Encoder/Decoder.
    SFNO_ORDER = ['Spectral Filter', 'MLP', 'Skip Connections', 'Normalization', 'Activation']
    NON_SFNO_ORDER = ['Encoder', 'Decoder']

    grouped_names = []
    grouped_times = []
    for name in SFNO_ORDER:
        if name in groups:
            grouped_names.append(name)
            grouped_times.append(groups[name]['time'])
        elif name in ungrouped:
            grouped_names.append(name)
            grouped_times.append(ungrouped[name])
    for name in NON_SFNO_ORDER:
        if name in ungrouped:
            grouped_names.append(name)
            grouped_times.append(ungrouped[name])

    names = grouped_names
    times = grouped_times
    total = sum(times)
    pcts  = [(t / total) * 100 for t in times]

    # ---- Classify layers ----
    NON_SFNO = {'Encoder', 'Decoder'}
    is_sfno = [n not in NON_SFNO for n in names]

    sfno_time = sum(t for t, s in zip(times, is_sfno) if s)
    non_sfno_time = sum(t for t, s in zip(times, is_sfno) if not s)
    sfno_pct = (sfno_time / total) * 100
    non_sfno_pct = (non_sfno_time / total) * 100

    # ---- IEEE-style rcParams ----
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'text.usetex': False,
    })

    # IEEE single-column width ≈ 3.5 in
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Calculate startangle so the Encoder/Decoder group is centered on the right (3 o'clock)
    # Encoder/Decoder are at the end of the list. matplotlib pie goes counter-clockwise.
    non_sfno_angle = (non_sfno_time / total) * 360
    startangle = 0 + non_sfno_angle / 2

    # ---- Assign colors per layer name ----
    # Explicit mapping so we control every slice precisely
    color_map = {
        'Spectral Filter':  '#B47CC7',  # purple
        'MLP':              '#F0B27A',  # peach/orange
        'Skip Connections': '#77BEDB',  # steel blue
        'Normalization':    '#D65F5F',  # red
        'Activation':       '#6ACC65',  # green
        'Encoder':          '#D0D0D0',  # light gray
        'Decoder':          '#D0D0D0',  # same gray (differentiated by hatch)
    }
    colors = [color_map.get(n, '#AEB6BF') for n in names]

    # No explode -- keep it clean
    explode = [0.0] * len(times)

    wedges, texts, autotexts = ax.pie(
        times,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct >= 4.0 else '',
        startangle=startangle,
        colors=colors,
        explode=explode,
        pctdistance=0.75,
        counterclock=True,
        wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'},
        textprops={'fontsize': 6, 'fontfamily': 'serif'},
    )

    # Add hatching to Encoder/Decoder wedges -- same color, different hatch to tell apart
    hatch_map = {
        'Encoder': '///',    # diagonal lines
        'Decoder': '...',    # dots
    }
    plt.rcParams['hatch.color'] = '#999999'
    plt.rcParams['hatch.linewidth'] = 0.7
    for i, wedge in enumerate(wedges):
        if not is_sfno[i]:
            wedge.set_hatch(hatch_map.get(names[i], '///'))
            # Keep edge white and same linewidth as all other slices
            wedge.set_edgecolor('white')
            wedge.set_linewidth(0.8)

    # Style percentage labels
    for i, at in enumerate(autotexts):
        at.set_fontsize(5.5)
        at.set_fontweight('bold')
        if is_sfno[i]:
            at.set_color('white')
            at.set_path_effects([pe.withStroke(linewidth=2, foreground='#00000044')])
        else:
            at.set_color('#1a1a1a')
            at.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

    ax.set_aspect('equal')

    # ---- Build grouped legend (compact -- times only, no duplicate %) ----
    legend_handles = []
    legend_labels = []

    # SFNO header
    sfno_header = Patch(facecolor='none', edgecolor='none', label='_nolegend_')
    legend_handles.append(sfno_header)
    legend_labels.append(f'$\\bf{{SFNO}}$ ({sfno_time:.2f} ms)')

    for i in range(len(names)):
        if is_sfno[i]:
            legend_handles.append(wedges[i])
            legend_labels.append(f'  {names[i]} ({times[i]:.2f} ms)')

    # Separator
    sep = Patch(facecolor='none', edgecolor='none', label='_nolegend_')
    legend_handles.append(sep)
    legend_labels.append('')

    # Encoder/Decoder header
    non_sfno_header = Patch(facecolor='#D0D0D0', edgecolor='#777777',
                            hatch='xxx', linewidth=0.5)
    legend_handles.append(non_sfno_header)
    legend_labels.append(f'$\\bf{{Enc/Dec}}$ ({non_sfno_time:.2f} ms)')

    for i in range(len(names)):
        if not is_sfno[i]:
            h = hatch_map.get(names[i], '///')
            p = Patch(facecolor='#D0D0D0', edgecolor='#777777', hatch=h, linewidth=0.5)
            legend_handles.append(p)
            legend_labels.append(f'  {names[i]} ({times[i]:.2f} ms)')

    leg = ax.legend(
        legend_handles, legend_labels,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=5.5,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        framealpha=1.0,
        handlelength=0.8,
        handleheight=0.7,
        labelspacing=0.2,
        borderpad=0.3,
        handletextpad=0.3,
    )

    plt.subplots_adjust(left=-0.38, right=0.40, top=1.18, bottom=-0.18)

    # Save PNG
    png_path = os.path.join(output_dir, 'nvtx_kernel_execution_time_pie.png')
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {png_path}")

    # Save PDF (IEEE submission quality)
    pdf_path = os.path.join(output_dir, 'nvtx_kernel_execution_time_pie.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02, format='pdf')
    print(f"Saved: {pdf_path}")

    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == '__main__':
    import json as _json

    parser = argparse.ArgumentParser(
        description='Generate IEEE pie chart for ACE2 layer execution times')
    parser.add_argument('--db', default='ace2_nvtx_400_50_correlated.sqlite',
                        help='Path to the correlated SQLite database')
    parser.add_argument('--output-dir', default='ace2_nvtx_400_50_layer_graphs',
                        help='Output directory for the charts')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-extraction even if cache exists')
    args = parser.parse_args()

    cache_path = os.path.join(args.output_dir, '.pie_chart_cache.json')

    if not args.no_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}  (use --no-cache to re-extract)")
        with open(cache_path) as f:
            layer_data = [tuple(x) for x in _json.load(f)]
    else:
        if not os.path.exists(args.db):
            print(f"Error: database not found: {args.db}")
            sys.exit(1)

        print(f"Reading data from: {args.db}")
        layer_data = extract_layer_data(args.db)

        if not layer_data:
            print("No layer data extracted. Check the database.")
            sys.exit(1)

        # Cache for fast re-runs
        os.makedirs(args.output_dir, exist_ok=True)
        with open(cache_path, 'w') as f:
            _json.dump(layer_data, f)
        print(f"Cached data to {cache_path}")

    print(f"Found {len(layer_data)} layers:")
    for name, time_ms, *_ in layer_data:
        print(f"  {name}: {time_ms:.2f} ms")

    generate_pie_chart(layer_data, output_dir=args.output_dir)
