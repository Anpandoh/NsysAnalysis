import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
from typing import Dict, List, Tuple
from rocprof_utils import RocProfAnalyzer

# Main analysis script
if __name__ == "__main__":
    # Path to the CSV files
    csv_path = "../Mi100/results.csv"
    stats_path = "../Mi100/results.stats.csv"
    
    # Initialize the analyzer
    analyzer = RocProfAnalyzer(csv_path, stats_path)
    
    # Load data
    analyzer.load_data()
    
    print("===== ROCPROF KERNEL ANALYSIS =====")
    print(f"Total kernels executed: {len(analyzer.data)}")
    
    # Analyze top 10 kernel types
    top10_kernels, total_all_kernels_runtime = analyzer.analyze_top_kernels(top_n=10)
    
    # Create runtime data for visualization
    runtime_data = {kernel_type: total_duration for kernel_type, total_duration, _, _, _ in top10_kernels}
    
    # Calculate runtime of kernels not in top 10
    top10_kernels_runtime = sum(total_duration for kernel_type, total_duration, _, _, _ in top10_kernels)
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
    
    # Create visualizations
    analyzer.create_runtime_visualizations(runtime_data, prefix="kernel", 
                                         title_prefix="Top 10 Kernel Type (Total Duration) ", 
                                         total_runtime=total_all_kernels_runtime)
    
    # Analyze hardware metrics for top 10 kernel types
    metrics_by_kernel = analyzer.analyze_kernel_metrics(top10_kernels)
    
    # Create hardware metrics visualizations
    analyzer.create_metrics_visualizations(metrics_by_kernel, prefix="metrics")
    
    print("\n===== ROCPROF ANALYSIS COMPLETE =====")

