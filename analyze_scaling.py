import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the unified summary
with open('all_traces_summary.json', 'r') as f:
    all_data = json.load(f)

# Output directory for plots
output_dir = 'scaled_model_graphs'
os.makedirs(output_dir, exist_ok=True)

# Sort traces by model size (assuming base name encodes this, e.g., sfno_1024, sfno_2048, etc.)
def extract_size(name):
    # Extracts the largest integer in the name for sorting
    import re
    nums = [int(s) for s in re.findall(r'\d+', name)]
    return nums[-1] if nums else 0

trace_names = sorted(all_data.keys(), key=extract_size)

# Extract embedded dimensions for x-axis
trace_dims = [extract_size(name) for name in trace_names]

# Print which traces have layer data
print("Checking layer data availability:")
traces_with_layers = []
for trace in trace_names:
    has_layer = 'layer' in all_data[trace]
    print(f"  {trace}: {'✓' if has_layer else '✗'} layer data")
    if has_layer:
        traces_with_layers.append(trace)

print(f"\nFound layer data in {len(traces_with_layers)} traces: {traces_with_layers}")

# The 6 metric names (as in your metric_map)
metric_names = [
    "4",  # SM Issue [Throughput %]
    "5",  # Tensor Active [Throughput %]
    "12", # Compute Warps in Flight [Throughput %]
    "15", # Unallocated Warps in Active SMs [Throughput %]
    "18", # DRAM Read Bandwidth [Throughput %]
    "19", # DRAM Write Bandwidth [Throughput %]
]
metric_labels = {
    "4": "SM Issue [Throughput %]",
    "5": "Tensor Active [Throughput %]",
    "12": "Compute Warps in Flight [Throughput %]",
    "15": "Unallocated Warps in Active SMs [Throughput %]",
    "18": "DRAM Read Bandwidth [Throughput %]",
    "19": "DRAM Write Bandwidth [Throughput %]"
}
# For filename: make a short, safe name
metric_shortnames = {
    "4": "sm_issue_throughput_pct",
    "5": "tensor_active_throughput_pct",
    "12": "compute_warps_in_flight_throughput_pct",
    "15": "unallocated_warps_in_active_sms_throughput_pct",
    "18": "dram_read_bandwidth_throughput_pct",
    "19": "dram_write_bandwidth_throughput_pct"
}

# --- 1. Top 10 Kernel Metrics Over Traces ---
# Collect top 10 kernel types for each trace
top10_kernels_per_trace = []
for trace in trace_names:
    kernel_data = all_data[trace].get('kernel', {})
    top10 = [k['kernel_type'] for k in kernel_data.get('top10_kernel_types', [])]
    top10_kernels_per_trace.append(top10)

# Check if top 10 kernels are consistent
reference_top10 = top10_kernels_per_trace[0]
for i, top10 in enumerate(top10_kernels_per_trace[1:], 1):
    if top10 != reference_top10:
        print(f"Top 10 kernels differ for trace {trace_names[i]}:")
        print(f"  Reference: {reference_top10}")
        print(f"  This trace: {top10}")
        print()

# For plotting, use the union of all top 10 kernels seen
all_top_kernels = sorted(set(k for top10 in top10_kernels_per_trace for k in top10))

# For each metric, for each kernel, plot value vs trace
for metric in metric_names:
    plt.figure(figsize=(12, 7))
    for kernel_type in all_top_kernels:
        y = []
        for trace in trace_names:
            kernel_data = all_data[trace].get('kernel', {})
            found = False
            for k in kernel_data.get('top10_kernel_types', []):
                if k['kernel_type'] == kernel_type:
                    m = k.get('metrics', {})
                    val = m.get(metric)
                    y.append(val if val is not None else np.nan)
                    found = True
                    break
            if not found:
                y.append(np.nan)
        plt.plot(trace_dims, y, marker='o', label=kernel_type)
    plt.title(f"{metric_labels[metric]} for Top Kernels Across Traces")
    plt.xlabel("Embedded Dimension")
    plt.ylabel(metric_labels[metric])
    plt.xticks(trace_dims)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save plot
    filename = f"kernels_{metric_shortnames[metric]}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# --- 2. Layer Metrics Over Traces ---
# Only plot if we have layer data
if traces_with_layers:
    # Get all layer types present across traces with layer data
    all_layer_types = set()
    for trace in traces_with_layers:
        layer_data = all_data[trace].get('layer', {})
        metrics_by_layer = layer_data.get('metrics_by_layer_type', {})
        all_layer_types.update(metrics_by_layer.keys())
    all_layer_types = sorted(all_layer_types)
    
    print(f"\nFound layer types: {all_layer_types}")
    
    # Create x-axis values for only traces with layer data
    layer_trace_dims = [extract_size(trace) for trace in traces_with_layers]
    
    for metric in metric_names:
        plt.figure(figsize=(12, 7))
        for layer_type in all_layer_types:
            y = []
            for trace in traces_with_layers:
                layer_data = all_data[trace].get('layer', {})
                metrics_by_layer = layer_data.get('metrics_by_layer_type', {})
                val = metrics_by_layer.get(layer_type, {}).get(metric)
                y.append(val if val is not None else np.nan)
            plt.plot(layer_trace_dims, y, marker='o', label=layer_type)
        plt.title(f"{metric_labels[metric]} for Layers Across Traces")
        plt.xlabel("Embedded Dimension")
        plt.ylabel(metric_labels[metric])
        plt.xticks(layer_trace_dims)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # Save plot
        filename = f"layers_{metric_shortnames[metric]}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()
else:
    print("\nSkipping layer metrics plots - no layer data found")

# --- 3. Total Runtime for Each Kernel Type Over Traces ---
plt.figure(figsize=(12, 7))
for kernel_type in all_top_kernels:
    y = []
    for trace in trace_names:
        kernel_data = all_data[trace].get('kernel', {})
        found = False
        for k in kernel_data.get('top10_kernel_types', []):
            if k['kernel_type'] == kernel_type:
                val = k.get('total_duration')
                y.append(val / 1e6 if val is not None else np.nan)  # Convert to ms
                found = True
                break
        if not found:
            y.append(np.nan)
    plt.plot(trace_dims, y, marker='o', label=kernel_type)
plt.title("Total Runtime for Top Kernels Across Traces")
plt.xlabel("Embedded Dimension")
plt.ylabel("Total Duration (ms)")
plt.xticks(trace_dims)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
filename = "kernels_total_runtime.png"
plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
plt.close()

# --- 4. Total Runtime for Each Layer Type Over Traces ---
if traces_with_layers:
    plt.figure(figsize=(12, 7))
    # Get all layer types present in runtime data
    all_runtime_layer_types = set()
    for trace in traces_with_layers:
        layer_data = all_data[trace].get('layer', {})
        layer_runtime_data = layer_data.get('layer_runtime_data', {})
        all_runtime_layer_types.update(layer_runtime_data.keys())
    all_runtime_layer_types = sorted(all_runtime_layer_types)
    
    print(f"\nFound runtime layer types: {all_runtime_layer_types}")
    
    for layer_type in all_runtime_layer_types:
        y = []
        for trace in traces_with_layers:
            layer_data = all_data[trace].get('layer', {})
            layer_runtime_data = layer_data.get('layer_runtime_data', {})
            val = layer_runtime_data.get(layer_type)
            y.append(val / 1e6 if val is not None else np.nan)  # Convert to ms
        plt.plot(layer_trace_dims, y, marker='o', label=layer_type)
    plt.title("Total Runtime for Layers Across Traces")
    plt.xlabel("Embedded Dimension")
    plt.ylabel("Total Duration (ms)")
    plt.xticks(layer_trace_dims)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filename = "layers_total_runtime.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
else:
    print("\nSkipping layer runtime plots - no layer data found")

# --- 5. Normalized Total Runtime for Each Kernel Type Over Traces (Excluding Top 3) ---
# 1. Gather total runtimes for all kernels
kernel_runtimes = {k: [] for k in all_top_kernels}
for kernel_type in all_top_kernels:
    for trace in trace_names:
        kernel_data = all_data[trace].get('kernel', {})
        found = False
        for k in kernel_data.get('top10_kernel_types', []):
            if k['kernel_type'] == kernel_type:
                val = k.get('total_duration')
                kernel_runtimes[kernel_type].append(val / 1e6 if val is not None else np.nan)  # ms
                found = True
                break
        if not found:
            kernel_runtimes[kernel_type].append(np.nan)

# 2. Identify top 3 kernels by runtime at largest embedding dimension
last_idx = -1  # index for largest embedding dimension
kernel_runtimes_at_max = [(k, kernel_runtimes[k][last_idx]) for k in all_top_kernels]
kernel_runtimes_at_max = [(k, v) for k, v in kernel_runtimes_at_max if not np.isnan(v)]
top3_kernels = [k for k, _ in sorted(kernel_runtimes_at_max, key=lambda x: x[1], reverse=True)[:3]]

# 3. Plot normalized runtimes, excluding top 3
plt.figure(figsize=(12, 7))
for kernel_type in all_top_kernels:
    if kernel_type in top3_kernels:
        continue
    y = kernel_runtimes[kernel_type]
    # Normalize by value at smallest embedding dimension (first index)
    base = y[0]
    if base is None or np.isnan(base) or base == 0:
        normed = [np.nan for _ in y]
    else:
        normed = [v / base if not np.isnan(v) and base != 0 else np.nan for v in y]
    plt.plot(trace_dims, normed, marker='o', label=kernel_type)
plt.title("Normalized Total Runtime for Kernels (Excl. Top 3) Across Traces")
plt.xlabel("Embedded Dimension")
plt.ylabel("Total Duration (normalized to 256)")
plt.xticks(trace_dims)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
filename = "kernels_total_runtime_normalized.png"
plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
plt.close()

print(f"\nAll plots saved to '{output_dir}/' directory") 