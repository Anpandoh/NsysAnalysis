from nsys_analysis.nsys_utils import NSysAnalyzer

# Path to the SQLite database
db_path = "../A100/SFNO_NSYS/gigaio80/sfno_2048.sqlite"
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

# Fetch all kernels
all_kernels = analyzer.fetch_all_kernels()
# Restrict to back 3/4 of kernels to avoid the startup overhead
all_kernels = all_kernels[-(len(all_kernels) * 3 // 4):] if len(all_kernels) > 0 else all_kernels
print(f"Fetched {len(all_kernels)} total kernels (last 500)")

# Group all kernels by type
kernel_types = analyzer.group_kernels_by_type(all_kernels)

# Compute total duration for each kernel type
sum_duration_by_type = []
for kernel_type, kernels in kernel_types.items():
    total_duration = sum(k[4] for k in kernels)
    avg_duration = total_duration / len(kernels) if kernels else 0
    sum_duration_by_type.append((kernel_type, total_duration, avg_duration, kernels))

# Sort by total duration and select top 10
sum_duration_by_type.sort(key=lambda x: x[1], reverse=True)
top10_types = sum_duration_by_type[:10]

print(f"Top 10 unique kernel types by total duration:")
for i, (kernel_type, total_duration, avg_duration, kernels) in enumerate(top10_types):
    print(f"{i+1}. {kernel_type} | Total Duration: {total_duration/1e6:.2f} ms | Avg Duration: {avg_duration/1e6:.2f} ms | Calls: {len(kernels)} | Example Grid: ({kernels[0][8]},{kernels[0][9]},{kernels[0][10]}) | Block: ({kernels[0][11]},{kernels[0][12]},{kernels[0][13]})")

print(f"\n===== SM ISSUE ANALYSIS FOR TOP 10 KERNEL TYPES (BY TOTAL DURATION) =====")
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
if analyzer.table_exists("GPU_METRICS") and analyzer.table_exists("TARGET_INFO_GPU_METRICS"):
    print("\n===== METRICS FOR TOP 10 KERNEL TYPES (BY TOTAL DURATION) =====")
    metrics_by_kernel = {}
    for i, (kernel_type, total_duration, avg_duration, kernels) in enumerate(top10_types):
        print(f"\nProcessing kernel type {i}: {kernel_type}")
        # Only use the middle 750 kernels for metrics analysis
        # if len(kernels) > 750:
        #     start_idx = (len(kernels) - 750) // 2
        #     kernels_for_metrics = kernels[start_idx:start_idx + 750]
        # else:
        kernels_for_metrics = kernels
        kernel_avg_metrics = analyzer.get_metrics_for_kernels_k(kernels_for_metrics, batch_size=500, sample_fraction=1)
        metrics_by_kernel[kernel_type] = kernel_avg_metrics

    print("\n===== METRICS BY KERNEL TYPE (TOP 10 BY TOTAL DURATION) =====")
    for kernel_type, metrics in metrics_by_kernel.items():
        print(f"\nKernel Type: {kernel_type}")
        if metrics:
            for metric_id, value in metrics.items():
                print(f"  {analyzer.metric_map.get(metric_id, f'Metric {metric_id}')}: {value:.2f}%")
        else:
            print("  No metrics data available")

    # Create visualizations
    if metrics_by_kernel:
        analyzer.create_visualizations(metrics_by_kernel, prefix="kernel", title_prefix="Top 10 Kernel Type (Total Duration) ")

# Close the connection when done
analyzer.disconnect()
            