from nsys_utils import NSysAnalyzer

# Path to the SQLite database
db_path = "../A100/ACE_NSYS/ace2_nvtx_400_50.sqlite"
# db_path = "../A100/SFNO_NSYS/liqid/sfno_384.sqlite"
# db_path = "../A100/SFNO_NSYS/gigaio80/sfno_2048.sqlite"
# db_path = "../A100/ACE_NSYS/ACE2_400_50.sqlite"

# Initialize the analyzer
analyzer = NSysAnalyzer(db_path)
analyzer.connect()

# Example: Print the list of tables in the database
analyzer.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = analyzer.cursor.fetchall()
print("Tables in the database:", tables)

# Get the column names of the CUPTI_ACTIVITY_KIND_KERNEL table
analyzer.cursor.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL);")
columns = analyzer.cursor.fetchall()
column_names = [column[1] for column in columns]
print("Column names in CUPTI_ACTIVITY_KIND_KERNEL:", column_names)

print("")
print("--------------------------------")
print("")

# Fetch all kernels
all_kernels = analyzer.fetch_all_kernels()
print(f"Fetched {len(all_kernels)} total kernels")

# Group all kernels by type
kernel_types = analyzer.group_kernels_by_type(all_kernels)

# Compute total duration for each kernel type (excluding first 4 iterations as startup outliers)
sum_duration_by_type = []
for kernel_type, kernels in kernel_types.items():
    # Skip first 4 kernels of each type (startup outliers)
    kernels_filtered = kernels[4:] if len(kernels) > 4 else []
    if kernels_filtered:
        total_duration = sum(k[4] for k in kernels_filtered)
        avg_duration = total_duration / len(kernels_filtered)
        sum_duration_by_type.append((kernel_type, total_duration, avg_duration, kernels_filtered))
    else:
        print(f"Warning: Kernel type '{kernel_type}' has {len(kernels)} kernels, skipping (need >4 for analysis)")

# Sort by total duration and select top 10
sum_duration_by_type.sort(key=lambda x: x[1], reverse=True)
top10_types = sum_duration_by_type[:10]

print(f"Top 10 unique kernel types by total duration (excluding first 4 iterations):")
for i, (kernel_type, total_duration, avg_duration, kernels) in enumerate(top10_types):
    print(f"{i+1}. {kernel_type} | Total Duration: {total_duration/1e6:.2f} ms | Avg Duration: {avg_duration/1e6:.2f} ms | Calls: {len(kernels)} | Example Grid: ({kernels[0][8]},{kernels[0][9]},{kernels[0][10]}) | Block: ({kernels[0][11]},{kernels[0][12]},{kernels[0][13]})")

# Calculate total runtime of all kernels
total_all_kernels_runtime = sum(kernel[4] for kernel in all_kernels)

# Create runtime data including "Other Kernels" for pie chart
runtime_data = {kernel_type: total_duration for kernel_type, total_duration, avg_duration, kernels in top10_types}

# Calculate runtime of kernels not in top 10
top10_kernels_runtime = sum(total_duration for kernel_type, total_duration, avg_duration, kernels in top10_types)
other_kernels_runtime = total_all_kernels_runtime - top10_kernels_runtime

# Debug information
print(f"\n===== RUNTIME CALCULATION DEBUG =====")
print(f"Total runtime of all kernels: {total_all_kernels_runtime/1e6:.2f} ms")
print(f"Total runtime of top 10 kernel types: {top10_kernels_runtime/1e6:.2f} ms")
print(f"Other kernels runtime: {other_kernels_runtime/1e6:.2f} ms")
print(f"Percentage accounted for by top 10: {(top10_kernels_runtime/total_all_kernels_runtime)*100:.1f}%")
print(f"Percentage for other kernels: {(other_kernels_runtime/total_all_kernels_runtime)*100:.1f}%")

# Add "Other Kernels" to runtime data if there are kernels not in top 10
if other_kernels_runtime > 0:
    runtime_data["Other Kernels"] = other_kernels_runtime
    print(f"Added 'Other Kernels' with runtime: {other_kernels_runtime/1e6:.2f} ms")
else:
    print("No 'Other Kernels' category needed - top 10 account for 100% of runtime")

# Pass the total runtime for accurate percentage calculation
analyzer.create_runtime_visualizations(runtime_data, prefix="kernel", title_prefix="Top 10 Kernel Type (Total Duration, Excluding First 4) ", total_runtime=total_all_kernels_runtime)

print("\n===== SM ISSUE ANALYSIS FOR TOP 10 KERNEL TYPES (BY TOTAL DURATION, EXCLUDING FIRST 4) =====")
for i, (kernel_type, total_duration, avg_duration, kernels) in enumerate(top10_types):
    print(f"{i+1}. {kernel_type}")
    print(f"   Calls: {len(kernels)}")
    print(f"   Total Duration: {total_duration/1e6:.2f} ms")
    print(f"   Average Duration: {avg_duration/1e6:.2f} ms")
    print(f"   Grid: ({kernels[0][8]},{kernels[0][9]},{kernels[0][10]}), Block: ({kernels[0][11]},{kernels[0][12]},{kernels[0][13]})")

# Print both TARGET_INFO_GPU_METRICS and GPU_METRICS table columns
if analyzer.table_exists("GPU_METRICS"):
    analyzer.cursor.execute("PRAGMA table_info(GPU_METRICS);")
    gpu_metrics_columns = analyzer.cursor.fetchall()
    gpu_metrics_column_names = [column[1] for column in gpu_metrics_columns]
    print("\n===== GPU_METRICS TABLE COLUMNS =====")
    print(gpu_metrics_column_names)

if analyzer.table_exists("TARGET_INFO_GPU_METRICS"):
    analyzer.cursor.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS);")
    target_info_columns = analyzer.cursor.fetchall()
    target_info_column_names = [column[1] for column in target_info_columns]
    print("\n===== TARGET_INFO_GPU_METRICS TABLE COLUMNS =====")
    print(target_info_column_names)

# Extract relevant metrics for the top 10 kernel types
metrics_by_kernel = {}
if analyzer.table_exists("GPU_METRICS") and analyzer.table_exists("TARGET_INFO_GPU_METRICS"):
    print("\n===== METRICS FOR TOP 10 KERNEL TYPES (BY TOTAL DURATION, EXCLUDING FIRST 4) =====")
    for i, (kernel_type, total_duration, avg_duration, kernels) in enumerate(top10_types):
        print(f"\nProcessing kernel type {i}: {kernel_type}")
        # Skip first 4 kernels (startup outliers), then use middle kernels for large datasets
        kernels_no_startup = kernels[4:] if len(kernels) > 4 else kernels
        if len(kernels_no_startup) > 2000:
            start_idx = (len(kernels_no_startup) - 2000) // 2
            kernels_for_metrics = kernels_no_startup[start_idx:start_idx + 2000]
        else:
            kernels_for_metrics = kernels_no_startup
        kernel_avg_metrics = analyzer.get_metrics_by_kernel(kernels_for_metrics, batch_size=500)
        metrics_by_kernel[kernel_type] = kernel_avg_metrics

    print("\n===== METRICS BY KERNEL TYPE (TOP 10 BY TOTAL DURATION, EXCLUDING FIRST 4) =====")
    for kernel_type, metrics in metrics_by_kernel.items():
        print(f"\nKernel Type: {kernel_type}")
        if metrics:
            for metric_id, value in metrics.items():
                print(f"  {analyzer.metric_map.get(metric_id, f'Metric {metric_id}')}: {value:.2f}%")
        else:
            print("  No metrics data available")

    # Create visualizations
    if metrics_by_kernel:
        analyzer.create_visualizations(metrics_by_kernel, prefix="kernel", title_prefix="Top 10 Kernel Type (Total Duration, Excluding First 4) ")

# Close the connection when done
analyzer.disconnect()

# ---- Export summary to unified JSON for later comparison ----
import os, json
json_path = "all_traces_summary.json"
db_base = os.path.splitext(os.path.basename(db_path))[0]
# Prepare the kernel summary dict
kernel_summary = {
    'total_all_kernels_runtime': total_all_kernels_runtime,
    'top10_kernel_types': [
        {
            'kernel_type': kernel_type,
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'num_calls': len(kernels),
            'metrics': metrics_by_kernel.get(kernel_type, {})
        }
        for kernel_type, total_duration, avg_duration, kernels in top10_types
    ],
    'other_kernels_runtime': other_kernels_runtime,
    'runtime_data': runtime_data,
}
# Load or create the unified JSON
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        all_data = json.load(f)
else:
    all_data = {}
if db_base not in all_data:
    all_data[db_base] = {}
all_data[db_base]['kernel'] = kernel_summary
with open(json_path, 'w') as f:
    json.dump(all_data, f, indent=2)
print(f"Saved kernel summary for {db_base} to {json_path}")
            