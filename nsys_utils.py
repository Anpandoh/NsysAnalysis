import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os
from typing import Dict, List, Tuple


class NSysAnalyzer:
    """Shared utilities for analyzing NSys trace databases"""
    
    def __init__(self, db_path: str):
        """Initialize the analyzer with database path"""
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.metric_map = {
            4: "SM Issue [Throughput %]",
            5: "Tensor Active [Throughput %]", 
            12: "Compute Warps in Flight [Throughput %]",
            15: "Unallocated Warps in Active SMs [Throughput %]",
            18: "DRAM Read Bandwidth [Throughput %]",
            19: "DRAM Write Bandwidth [Throughput %]"
        }
    
    def connect(self):
        """Connect to the SQLite database"""
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
    
    def disconnect(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a specific table exists"""
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        return self.cursor.fetchone() is not None
    
    def get_kernel_query(self) -> str:
        """Get the standard kernel query"""
        return """
            SELECT names.value AS kernel_name, k.correlationId, k.start, k.end, 
                   k.end - k.start AS duration, k.deviceId, k.contextId, k.streamId,
                   k.gridX, k.gridY, k.gridZ, 
                   k.blockX, k.blockY, k.blockZ,
                   k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS names ON k.demangledName = names.id
        """
    
    def fetch_all_kernels(self) -> List[Tuple]:
        """Fetch all kernels that start after the first memset kernel, ordered by start time"""
        # First, find the start time of the first memset operation from the MEMSET table
        memset_query = """
            SELECT start FROM CUPTI_ACTIVITY_KIND_MEMSET 
            ORDER BY start ASC LIMIT 1;
        """
        self.cursor.execute(memset_query)
        memset_result = self.cursor.fetchone()
        
        if memset_result is None:
            print("Warning: No memset operation found. Returning all kernels.")
            query = self.get_kernel_query() + " ORDER BY k.start ASC;"
            self.cursor.execute(query)
            return self.cursor.fetchall()
        
        memset_start = memset_result[0]  # start time is the first column
        print(f"Found memset operation starting at {memset_start}. Filtering kernels after this time.")
        
        # Fetch all kernels that start after the memset operation
        query = self.get_kernel_query() + f" WHERE k.start > {memset_start} ORDER BY k.start ASC;"
        self.cursor.execute(query)
        kernels = self.cursor.fetchall()
        print(f"Found {len(kernels)} kernels starting after memset")
        return kernels
    
    def get_metrics_by_layer(self, kernels: List[Tuple], max_kernels: int = 900) -> Dict[int, float]:
        """Get GPU metrics for a list of kernels"""
        if not kernels:
            return {}
        
        # Limit kernels to avoid SQLite expression tree limit
        if len(kernels) > max_kernels:
            start_idx = (len(kernels) - max_kernels) // 2
            kernels = kernels[start_idx:start_idx + max_kernels]
            print(f"  Processing middle {max_kernels} kernels out of {len(kernels) + max_kernels} total kernels")
        
        # Build a list of (start_time, end_time) for the kernels
        kernel_data = [(kernel[2], kernel[3]) for kernel in kernels]  # start_time, end_time
        
        if not kernel_data:
            return {}
        
        # Use efficient approach: find min and max timestamps and query within that range
        min_start = min(start for start, end in kernel_data)
        max_end = max(end for start, end in kernel_data)
        
        # Query all metrics within the time range, then filter in Python
        self.cursor.execute(f"""
            SELECT g.metricId, g.timestamp, g.value
            FROM GPU_METRICS g
            JOIN TARGET_INFO_GPU_METRICS t ON g.metricId = t.metricId
            WHERE g.typeId = t.typeId
            AND g.metricId IN {tuple(self.metric_map.keys())}
            AND g.timestamp >= ? AND g.timestamp <= ?
            ORDER BY g.timestamp
        """, [min_start, max_end])
        
        all_metrics = self.cursor.fetchall()
        
        # Filter metrics that fall within any kernel time range
        filtered_metrics = {}
        for metric_id, timestamp, value in all_metrics:
            # Check if this timestamp falls within any kernel execution
            for start_time, end_time in kernel_data:
                if start_time <= timestamp <= end_time:
                    if metric_id not in filtered_metrics:
                        filtered_metrics[metric_id] = []
                    filtered_metrics[metric_id].append(value)
                    break
        
        # Calculate averages for each metric
        layer_metrics = {}
        for metric_id, values in filtered_metrics.items():
            if values:
                layer_metrics[metric_id] = sum(values) / len(values)
        
        return layer_metrics
    
    def get_metrics_by_kernel(self, kernels: List[Tuple], batch_size: int = 100) -> Dict[int, float]:
        """Get GPU metrics for kernels using batched processing with sampling"""
        if not kernels:
            return {}
        
        # Build a list of (start_time, end_time) for the kernels
        kernel_data = [(kernel[2], kernel[3]) for kernel in kernels]
        
        # Process kernels in batches with sampling
        metrics_sum = {}
        metrics_count = {}
        
        # Calculate total number of batches and only process a fraction of them
        total_batches = (len(kernel_data) + batch_size - 1) // batch_size

        for batch_start_idx in range(total_batches):
            batch_end_idx = min(batch_start_idx + batch_size, len(kernel_data))
            batch = kernel_data[batch_start_idx:batch_end_idx]
            
            if not batch:
                continue
                
            # Construct SQL query with multiple timestamp ranges in a single query
            conditions = []
            params = []
            for start_time, end_time in batch:
                conditions.append("(g.timestamp >= ? AND g.timestamp <= ?)")
                params.extend([start_time, end_time])
            
            condition_str = " OR ".join(conditions)
            
            # Query all metrics for this batch in a single query
            self.cursor.execute(f"""
                SELECT g.metricId, AVG(g.value) as avg_value, COUNT(g.value) as count
                FROM GPU_METRICS g
                JOIN TARGET_INFO_GPU_METRICS t ON g.metricId = t.metricId
                WHERE g.typeId = t.typeId
                AND g.metricId IN {tuple(self.metric_map.keys())}
                AND ({condition_str})
                GROUP BY g.metricId
            """, params)
            
            batch_metrics = self.cursor.fetchall()
            
            # Accumulate batch results
            for metric_id, avg_value, count in batch_metrics:
                if metric_id not in metrics_sum:
                    metrics_sum[metric_id] = 0
                    metrics_count[metric_id] = 0
                metrics_sum[metric_id] += avg_value * count
                metrics_count[metric_id] += count
        
        # Calculate final averages
        kernel_avg_metrics = {}
        for metric_id in metrics_sum:
            if metrics_count[metric_id] > 0:
                kernel_avg_metrics[metric_id] = metrics_sum[metric_id] / metrics_count[metric_id]
        
        return kernel_avg_metrics
    
    def create_visualizations(self, metrics_data: Dict[str, Dict[int, float]], 
                            prefix: str = "analysis", title_prefix: str = ""):
        """Create visualizations for metrics data"""
        if not metrics_data:
            return
        
        # Create output directory based on prefix
        # Extract base name from db path using regex
        base_name = re.search(r"([^/]+?)(?:\.[^.]+)?$", self.db_path).group(1)
        output_dir = f"{base_name}_{prefix}_graphs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a dataframe for visualization
        metrics_df = pd.DataFrame(columns=['Item Type'] + list(self.metric_map.values()))
        for item_type, metrics in metrics_data.items():
            row = {'Item Type': item_type}
            for metric_id, metric_name in self.metric_map.items():
                row[metric_name] = metrics.get(metric_id, 0)
            metrics_df = metrics_df._append(row, ignore_index=True)
        
        # Create individual bar charts for each metric
        if not metrics_df.empty:
            # Get item types for x-axis
            item_types = metrics_df['Item Type']
            x = np.arange(len(item_types))
            
            # Create a separate graph for each metric
            for metric_id, metric_name in self.metric_map.items():
                # Increase figure size, especially width, to prevent x-axis label cutoff
                plt.figure(figsize=(14, 8))
                
                # Create the bar chart for this metric
                plt.bar(x, metrics_df[metric_name], width=0.6, alpha=0.7, color='skyblue')
                
                # Add labels and title
                plt.xlabel('Item Type', fontsize=12)
                plt.ylabel('Throughput %', fontsize=12)
                plt.title(f'{title_prefix}{metric_name} by Item Type', fontsize=14)
                
                # Improve x-axis labels to avoid cutoff
                x_labels = []
                for item_type in item_types:
                    if len(item_type) > 30 and 'sm80_xmma' in item_type:
                        # Shorten cudnn kernels
                        parts = item_type.split('_')
                        if len(parts) > 5:
                            x_labels.append(f"{parts[0]}_{parts[1]}_cudnn")
                    elif 'regular_fft' in item_type:
                        # Shorten FFT kernels
                        x_labels.append(item_type.split('<')[0])
                    else:
                        x_labels.append(item_type)
                
                # Adjust x-axis ticks and labels
                plt.xticks(x, x_labels, rotation=45, ha='right')
                
                # Increase bottom margin to allow space for rotated labels
                plt.subplots_adjust(bottom=0.25)
                
                # Add value labels on top of bars
                for i, v in enumerate(metrics_df[metric_name]):
                    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
                
                # Adjust layout and save
                plt.tight_layout()
                
                # Create clean filename from metric name
                filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
                filepath = os.path.join(output_dir, f'{filename}.png')
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                plt.close()
            
            print(f"\nPerformance visualizations saved in '{output_dir}/' folder:")
            for metric_id, metric_name in self.metric_map.items():
                filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
                print(f"- {filename}.png - {metric_name}")
    
    def create_runtime_visualizations(self, runtime_data: Dict[str, float], 
                                    prefix: str = "analysis", title_prefix: str = "", total_runtime: float = None):
        """Create runtime visualizations for kernel types or layers"""
        if not runtime_data:
            return
        
        # Create output directory based on prefix
        base_name = re.search(r"([^/]+?)(?:\.[^.]+)?$", self.db_path).group(1)
        output_dir = f"{base_name}_{prefix}_graphs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort data by duration (descending)
        sorted_data = sorted(runtime_data.items(), key=lambda x: x[1], reverse=True)
        item_types, durations = zip(*sorted_data)
        
        # Convert durations to milliseconds
        durations_ms = [d / 1e6 for d in durations]
        
        # Create the runtime bar chart
        plt.figure(figsize=(14, 8))
        
        # Create bar chart
        x = np.arange(len(item_types))
        bars = plt.bar(x, durations_ms, width=0.6, alpha=0.7, color='lightcoral')
        
        # Add labels and title
        plt.xlabel('Item Type', fontsize=12)
        plt.ylabel('Total Duration (ms)', fontsize=12)
        plt.title(f'{title_prefix}Total Runtime by Item Type', fontsize=14)
        
        # Improve x-axis labels to avoid cutoff
        x_labels = []
        for item_type in item_types:
            if len(item_type) > 30 and 'sm80_xmma' in item_type:
                # Shorten cudnn kernels
                parts = item_type.split('_')
                if len(parts) > 5:
                    x_labels.append(f"{parts[0]}_{parts[1]}_cudnn")
            elif 'regular_fft' in item_type:
                # Shorten FFT kernels
                x_labels.append(item_type.split('<')[0])
            elif len(item_type) > 25:
                # Shorten other long names
                x_labels.append(item_type[:25] + "...")
            else:
                x_labels.append(item_type)
        
        # Adjust x-axis ticks and labels
        plt.xticks(x, x_labels, rotation=45, ha='right')
        
        # Increase bottom margin to allow space for rotated labels
        plt.subplots_adjust(bottom=0.25)
        
        # Add value labels on top of bars
        for i, v in enumerate(durations_ms):
            plt.text(i, v + max(durations_ms) * 0.01, f'{v:.1f}ms', ha='center', fontsize=9)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the bar chart
        filepath = os.path.join(output_dir, 'total_runtime.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create pie chart for runtime percentages
        plt.figure(figsize=(12, 10))
        
        # Calculate percentages - use total_runtime if provided, otherwise sum of durations
        if total_runtime is not None:
            percentages = [(d / total_runtime) * 100 for d in durations]
        else:
            total_runtime = sum(durations)
            percentages = [(d / total_runtime) * 100 for d in durations]
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(item_types)))
        wedges, texts, autotexts = plt.pie(percentages, labels=None, autopct='%1.1f%%', 
                                          startangle=90, colors=colors, textprops={'fontsize': 10})
        
        # Add title
        plt.title(f'{title_prefix}Runtime Distribution (Percentage)', fontsize=14, pad=20)
        
        # Create legend with shortened labels
        legend_labels = []
        for item_type in item_types:
            if len(item_type) > 30 and 'sm80_xmma' in item_type:
                # Shorten cudnn kernels
                parts = item_type.split('_')
                if len(parts) > 5:
                    legend_labels.append(f"{parts[0]}_{parts[1]}_cudnn")
            elif 'regular_fft' in item_type:
                # Shorten FFT kernels
                legend_labels.append(item_type.split('<')[0])
            elif len(item_type) > 25:
                # Shorten other long names
                legend_labels.append(item_type[:25] + "...")
            else:
                legend_labels.append(item_type)
        
        # Add legend
        plt.legend(wedges, legend_labels, title="Item Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Ensure equal aspect ratio for circular pie
        plt.axis('equal')
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        
        # Save the pie chart
        pie_filepath = os.path.join(output_dir, 'runtime_percentage.png')
        plt.savefig(pie_filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"\nRuntime visualizations saved in '{output_dir}/' folder:")
        print(f"- total_runtime.png - Total Runtime by Item Type (Bar Chart)")
        print(f"- runtime_percentage.png - Runtime Distribution (Pie Chart)")
        
        # Print summary
        print(f"\n===== RUNTIME SUMMARY =====")
        print(f"Total Runtime: {total_runtime/1e6:.2f} ms")
        print(f"Number of Items: {len(runtime_data)}")
        print(f"\nTop 5 Longest Items:")
        for i, (item_type, duration) in enumerate(sorted_data[:5]):
            percentage = (duration / total_runtime) * 100
            print(f"  {i+1}. {item_type}: {duration/1e6:.2f} ms ({percentage:.1f}%)")
    
    def extract_kernel_type(self, kernel_name: str) -> str:
        """Extract kernel type from kernel name"""
        # First, check for cutlass kernel patterns and shorten them
        if "cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64" in kernel_name:
            return "cutlass_gemm_256x64"
        elif "cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128" in kernel_name:
            return "cutlass_gemm_128x128"
        
        # Check for elementwise kernels
        EK_match = re.search(r'void at::native::elementwise_kernel.*?gpu_kernel_impl_nocast<at::native::([A-Za-z_][A-Za-z0-9_]*)', kernel_name)
        if EK_match:
            return f"EK_CUDAFunctor_{EK_match.group(1)}"
            
        # Check for vectorized elementwise kernels
        VEK_match = re.search(r'vectorized_elementwise_kernel.*?([A-Za-z]+(?:Functor|Impl|cuda|CUDA)[^,<>]*)', kernel_name)
        if VEK_match:
            return f"VEK_{VEK_match.group(1)}"
        
        # Extract patterns like ampere_h16816gemm_128x128_ldg8_stages_64x3_tn
        match = re.search(r'(ampere_[^()\s]+)', kernel_name)
        if match:
            return match.group(1)
            
        # Fall back to first part of the name if pattern not found
        return kernel_name.replace('void ', '').split('<')[0].strip()
    
    def group_kernels_by_type(self, kernels: List[Tuple]) -> Dict[str, List[Tuple]]:
        """Group kernels by their extracted type"""
        kernel_types = {}
        for kernel in kernels:
            kernel_type = self.extract_kernel_type(kernel[0])
            if kernel_type not in kernel_types:
                kernel_types[kernel_type] = []
            kernel_types[kernel_type].append(kernel)
        return kernel_types 

    def create_nvtx_metrics_visualizations(self, nvtx_metrics_data):
        """
        Create GPU metrics visualizations for NVTX ranges.
        Each graph shows one metric across all NVTX ranges.
        """
        if not nvtx_metrics_data:
            print("No NVTX metrics data available for visualization")
            return
        
        # Create output directory
        # Extract clean base name from db_path
        # Use naming_db_path if it exists (for overriding directory names), otherwise use db_path
        path_for_naming = getattr(self, 'naming_db_path', self.db_path)
        base_name = os.path.splitext(os.path.basename(path_for_naming))[0]
        # Remove "_correlated" suffix if present
        if base_name.endswith("_correlated"):
            base_name = base_name[:-11]
        # Remove "_nvtx" suffix if present to get clean trace name
        if base_name.endswith("_nvtx"):
            base_name = base_name[:-5]
        output_dir = f"{base_name}_layer_graphs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get NVTX range names
        range_names = list(nvtx_metrics_data.keys())
        
        # Create a separate graph for each metric
        for metric_id, metric_name in self.metric_map.items():
            plt.figure(figsize=(16, 10))
            
            # Extract metric values for this metric across all NVTX ranges
            metric_values = []
            valid_ranges = []
            
            for range_name in range_names:
                metrics = nvtx_metrics_data[range_name].get('metrics', {})
                if metric_id in metrics:
                    metric_values.append(metrics[metric_id])
                    valid_ranges.append(range_name)
            
            if not metric_values:
                print(f"No data found for {metric_name}")
                plt.close()
                continue
            
            # Create bar chart
            x = np.arange(len(valid_ranges))
            bars = plt.bar(x, metric_values, color='skyblue', alpha=0.8)
            
            plt.xlabel('NVTX Range', fontsize=12)
            plt.ylabel('Throughput %', fontsize=12)
            plt.title(f'{metric_name} by NVTX Range', fontsize=14, pad=20)
            
            # Rotate x-axis labels for better readability
            plt.xticks(x, valid_ranges, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, v in enumerate(metric_values):
                plt.text(i, v + max(metric_values) * 0.01, f'{v:.1f}%', 
                        ha='center', va='bottom', fontsize=10)
            
            # Add grid for better readability
            plt.grid(axis='y', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f'nvtx_{filename}.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
        
        # Print summary
        print(f"\n===== NVTX METRICS VISUALIZATION SUMMARY =====")
        print(f"Graphs saved in '{output_dir}/' folder:")
        for metric_id, metric_name in self.metric_map.items():
            filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
            print(f"- nvtx_{filename}.png - {metric_name} by NVTX range")
        
        # Print statistics summary
        print(f"\n===== NVTX METRICS SUMMARY =====")
        for range_name, data in nvtx_metrics_data.items():
            print(f"\n{range_name}:")
            metrics = data.get('metrics', {})
            if metrics:
                for metric_id, value in metrics.items():
                    metric_name = self.metric_map.get(metric_id, f"Metric {metric_id}")
                    print(f"  {metric_name}: {value:.2f}%")
            else:
                print(f"  No metrics available") 