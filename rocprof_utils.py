import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from typing import Dict, List, Tuple

class RocProfAnalyzer:
    """Analyzer for rocprof CSV trace data"""
    
    def __init__(self, csv_path: str, stats_path: str = None):
        """Initialize the analyzer with CSV file paths"""
        self.csv_path = csv_path
        self.stats_path = stats_path
        self.data = None
        self.stats_data = None
        
    def load_data(self):
        """Load the CSV data"""
        print(f"Loading rocprof data from {self.csv_path}")
        # Load with low_memory=False to avoid dtype guessing issues with large files
        self.data = pd.read_csv(self.csv_path, low_memory=False)
        print(f"Loaded {len(self.data)} kernel executions")
        
        if self.stats_path and os.path.exists(self.stats_path):
            print(f"Loading stats data from {self.stats_path}")
            self.stats_data = pd.read_csv(self.stats_path)
            print(f"Loaded {len(self.stats_data)} kernel types")
        
    def extract_kernel_type(self, kernel_name: str) -> str:
        """Extract kernel type from kernel name - adapted for rocprof"""
        # Remove .kd suffix that's common in rocprof
        if kernel_name.endswith('.kd'):
            kernel_name = kernel_name[:-3]
        
        # ROCm-specific convolution kernels
        if "naive_conv_ab_nonpacked_fwd_nchw" in kernel_name:
            return "naive_conv_fwd"
        elif "gridwise_convolution_forward_implicit_gemm" in kernel_name:
            return "gridwise_conv_fwd"
        elif "gridwise_convolution_implicit_gemm" in kernel_name:
            return "gridwise_conv"
        
        # GEMM kernels (Composable Kernel library)
        if kernel_name.startswith("Cijk_"):
            # Extract key parameters from CK GEMM kernel names
            if "MT16x16x16" in kernel_name:
                return "CK_GEMM_16x16x16"
            elif "MT16x16x32" in kernel_name:
                return "CK_GEMM_16x16x32"
            elif "MT64x64x16" in kernel_name:
                return "CK_GEMM_64x64x16"
            elif "MT128x128x16" in kernel_name:
                return "CK_GEMM_128x128x16"
            elif "MT256x" in kernel_name:
                return "CK_GEMM_256x"
            else:
                return "CK_GEMM_other"
        
        # PyTorch elementwise kernels
        if "elementwise_kernel" in kernel_name:
            # Extract the operation type
            if "direct_copy_kernel" in kernel_name:
                if "c10::complex<float>" in kernel_name:
                    return "EK_copy_complex"
                else:
                    return "EK_copy"
            elif "CUDAFunctor_add" in kernel_name:
                return "EK_add"
            elif "BinaryFunctor" in kernel_name and "MulFunctor" in kernel_name:
                return "EK_mul"
            elif "BinaryFunctor" in kernel_name and "DivFunctor" in kernel_name:
                return "EK_div"
            else:
                return "EK_other"
        
        # Vectorized elementwise kernels
        if "vectorized_elementwise_kernel" in kernel_name:
            if "CUDAFunctor_add" in kernel_name:
                return "VEK_add"
            elif "GeluCUDAKernelImpl" in kernel_name:
                return "VEK_gelu"
            elif "MulFunctor" in kernel_name:
                return "VEK_mul"
            elif "FillFunctor" in kernel_name:
                if "c10::complex<float>" in kernel_name:
                    return "VEK_fill_complex"
                else:
                    return "VEK_fill"
            elif "sqrt_kernel" in kernel_name:
                return "VEK_sqrt"
            elif "pow_tensor" in kernel_name:
                return "VEK_pow"
            elif "log_kernel" in kernel_name:
                return "VEK_log"
            elif "round_kernel" in kernel_name:
                return "VEK_round"
            elif "neg_kernel" in kernel_name:
                return "VEK_neg"
            elif "AbsFunctor" in kernel_name:
                return "VEK_abs"
            elif "cos_kernel" in kernel_name:
                return "VEK_cos"
            else:
                return "VEK_other"
        
        # MIOpen convolution kernels
        if kernel_name.startswith("miopenSp3Asm"):
            return "MIOpen_conv"
        elif kernel_name.startswith("MIOpenConv"):
            return "MIOpen_conv"
        elif kernel_name.startswith("MIOpenBatchNorm"):
            return "MIOpen_batchnorm"
        
        # IGEMM kernels
        if kernel_name.startswith("igemm_fwd_gtcx"):
            if "nchw" in kernel_name:
                return "igemm_nchw"
            elif "nhwc" in kernel_name:
                return "igemm_nhwc"
            else:
                return "igemm_other"
        
        # MLIRGen kernels
        if kernel_name.startswith("mlir_gen_igemm"):
            return "mlir_igemm"
        
        # ROCm copy operations
        if "__amd_rocclr_copyBuffer" in kernel_name:
            return "rocclr_copy"
        elif "__amd_rocclr_fillBuffer" in kernel_name:
            return "rocclr_fill"
        
        # FFT kernels
        if "fft_rtc" in kernel_name:
            if "fwd" in kernel_name:
                return "fft_forward"
            elif "back" in kernel_name:
                return "fft_backward"
            else:
                return "fft_other"
        
        # Transpose kernels
        if "batched_transpose" in kernel_name:
            return "transpose"
        
        # Reduce kernels
        if "reduce_kernel" in kernel_name:
            if "sum_functor" in kernel_name:
                return "reduce_sum"
            elif "MeanOps" in kernel_name:
                return "reduce_mean"
            else:
                return "reduce_other"
        
        # Tensor scan kernels
        if "tensor_kernel_scan" in kernel_name:
            return "tensor_scan"
        
        # Cat (concatenation) kernels
        if "CatArrayBatchedCopy" in kernel_name:
            return "cat_copy"
        
        # Unrolled elementwise kernels
        if "unrolled_elementwise_kernel" in kernel_name:
            if "CUDAFunctorOnSelf_add" in kernel_name:
                return "UEK_add_self"
            elif "CUDAFunctor_add" in kernel_name:
                return "UEK_add"
            elif "MulFunctor" in kernel_name:
                return "UEK_mul"
            else:
                return "UEK_other"
        
        # Index kernels
        if "index_elementwise_kernel" in kernel_name:
            return "index_flip"
        
        # Twiddle generation kernels
        if "twiddle_gen" in kernel_name:
            return "twiddle_gen"
        
        # Fused kernels
        if kernel_name.startswith("fused_"):
            return "fused_op"
        
        # Long CK kernel names (fallback for complex ones)
        if "_ZN2ck16tensor_operation6device" in kernel_name:
            return "CK_complex"
        
        # Fall back to first meaningful part
        # Remove common prefixes and get the main identifier
        clean_name = kernel_name.replace('void ', '').replace('at::native::', '')
        return clean_name.split('<')[0].split('(')[0].strip()
    
    def group_kernels_by_type(self) -> Dict[str, pd.DataFrame]:
        """Group kernels by their extracted type"""
        if self.data is None:
            self.load_data()
        
        # Add kernel type column
        self.data['KernelType'] = self.data['KernelName'].apply(self.extract_kernel_type)
        
        # Group by kernel type
        kernel_groups = {}
        for kernel_type, group in self.data.groupby('KernelType'):
            kernel_groups[kernel_type] = group
        
        return kernel_groups
    
    def analyze_top_kernels(self, top_n: int = 10):
        """Analyze top N kernel types by total duration"""
        kernel_groups = self.group_kernels_by_type()
        
        # Calculate total duration for each kernel type
        kernel_stats = []
        for kernel_type, group in kernel_groups.items():
            total_duration = group['DurationNs'].sum()
            avg_duration = group['DurationNs'].mean()
            calls = len(group)
            kernel_stats.append((kernel_type, total_duration, avg_duration, calls, group))
        
        # Sort by total duration and select top N
        kernel_stats.sort(key=lambda x: x[1], reverse=True)
        top_kernels = kernel_stats[:top_n]
        
        # Calculate total runtime of all kernels
        total_all_kernels_runtime = self.data['DurationNs'].sum()
        
        print(f"Total runtime of all kernels: {total_all_kernels_runtime/1e6:.2f} ms")
        print(f"\nTop {top_n} kernel types by total duration:")
        
        for i, (kernel_type, total_duration, avg_duration, calls, group) in enumerate(top_kernels):
            percentage = (total_duration / total_all_kernels_runtime) * 100
            print(f"{i+1}. {kernel_type}")
            print(f"   Total Duration: {total_duration/1e6:.2f} ms ({percentage:.1f}%)")
            print(f"   Average Duration: {avg_duration/1e6:.2f} ms")
            print(f"   Calls: {calls}")
            
            # Show grid/block info for first kernel instance
            if not group.empty:
                first_kernel = group.iloc[0]
                print(f"   Grid: {first_kernel.get('grd', 'N/A')}, Workgroup: {first_kernel.get('wgr', 'N/A')}")
            print()
        
        return top_kernels, total_all_kernels_runtime
    
    def analyze_kernel_metrics(self, top_kernels):
        """Analyze hardware metrics for top kernel types"""
        if self.data is None:
            self.load_data()
        
        # Define the metrics we want to analyze
        metrics_columns = {
            'grd': 'Grid Size',
            'wgr': 'Workgroup Size', 
            'lds': 'Local Data Store (KB)',
            'scr': 'Scratch Memory (KB)',
            'arch_vgpr': 'Architecture VGPRs',
            'accum_vgpr': 'Accumulator VGPRs',
            'sgpr': 'Scalar GPRs',
            'wave_size': 'Wavefront Size'
        }
        
        metrics_by_kernel = {}
        
        print("\n===== HARDWARE METRICS FOR TOP 10 KERNEL TYPES =====")
        for i, (kernel_type, total_duration, avg_duration, calls, group) in enumerate(top_kernels):
            print(f"\n{i+1}. {kernel_type} (Calls: {calls})")
            
            kernel_metrics = {}
            for col, display_name in metrics_columns.items():
                if col in group.columns:
                    # Calculate average, min, max for each metric
                    values = group[col].astype(float)
                    avg_val = values.mean()
                    min_val = values.min()
                    max_val = values.max()
                    std_val = values.std()
                    
                    kernel_metrics[col] = {
                        'avg': avg_val,
                        'min': min_val,
                        'max': max_val,
                        'std': std_val,
                        'display_name': display_name
                    }
                    
                    if col in ['lds', 'scr']:
                        # Convert bytes to KB for memory metrics
                        print(f"   {display_name}: avg={avg_val/1024:.1f}KB, min={min_val/1024:.1f}KB, max={max_val/1024:.1f}KB")
                    else:
                        print(f"   {display_name}: avg={avg_val:.1f}, min={min_val:.1f}, max={max_val:.1f}")
            
            metrics_by_kernel[kernel_type] = kernel_metrics
        
        return metrics_by_kernel
    
    def create_runtime_visualizations(self, runtime_data: Dict[str, float], 
                                    prefix: str = "rocprof", title_prefix: str = "", 
                                    total_runtime: float = None):
        """Create runtime visualizations for kernel types"""
        if not runtime_data:
            return
        
        # Create output directory
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        output_dir = f"{base_name}_{prefix}_graphs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort data by duration (descending)
        sorted_data = sorted(runtime_data.items(), key=lambda x: x[1], reverse=True)
        kernel_types, durations = zip(*sorted_data)
        
        # Convert durations to milliseconds
        durations_ms = [d / 1e6 for d in durations]
        
        # Create the runtime bar chart
        plt.figure(figsize=(14, 8))
        
        # Create bar chart
        x = np.arange(len(kernel_types))
        bars = plt.bar(x, durations_ms, width=0.6, alpha=0.7, color='lightcoral')
        
        # Add labels and title
        plt.xlabel('Kernel Type', fontsize=12)
        plt.ylabel('Total Duration (ms)', fontsize=12)
        plt.title(f'{title_prefix}Total Runtime by Kernel Type (ROCm)', fontsize=14)
        
        # Improve x-axis labels to avoid cutoff
        x_labels = []
        for kernel_type in kernel_types:
            if len(kernel_type) > 25:
                x_labels.append(kernel_type[:25] + "...")
            else:
                x_labels.append(kernel_type)
        
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
        
        # Calculate percentages
        if total_runtime is not None:
            percentages = [(d / total_runtime) * 100 for d in durations]
        else:
            total_runtime = sum(durations)
            percentages = [(d / total_runtime) * 100 for d in durations]
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(kernel_types)))
        wedges, texts, autotexts = plt.pie(percentages, labels=None, autopct='%1.1f%%', 
                                          startangle=90, colors=colors, textprops={'fontsize': 10})
        
        # Add title
        plt.title(f'{title_prefix}Runtime Distribution (Percentage) - ROCm', fontsize=14, pad=20)
        
        # Create legend with shortened labels
        legend_labels = []
        for kernel_type in kernel_types:
            if len(kernel_type) > 25:
                legend_labels.append(kernel_type[:25] + "...")
            else:
                legend_labels.append(kernel_type)
        
        # Add legend
        plt.legend(wedges, legend_labels, title="Kernel Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Ensure equal aspect ratio for circular pie
        plt.axis('equal')
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        
        # Save the pie chart
        pie_filepath = os.path.join(output_dir, 'runtime_percentage.png')
        plt.savefig(pie_filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"\nRuntime visualizations saved in '{output_dir}/' folder:")
        print(f"- total_runtime.png - Total Runtime by Kernel Type (Bar Chart)")
        print(f"- runtime_percentage.png - Runtime Distribution (Pie Chart)")
        
        # Print summary
        print(f"\n===== RUNTIME SUMMARY =====")
        print(f"Total Runtime: {total_runtime/1e6:.2f} ms")
        print(f"Number of Kernel Types: {len(runtime_data)}")
        print(f"\nTop 5 Longest Kernel Types:")
        for i, (kernel_type, duration) in enumerate(sorted_data[:5]):
            percentage = (duration / total_runtime) * 100
            print(f"  {i+1}. {kernel_type}: {duration/1e6:.2f} ms ({percentage:.1f}%)")
    
    def create_metrics_visualizations(self, metrics_by_kernel, prefix: str = "rocprof"):
        """Create visualizations for hardware metrics"""
        if not metrics_by_kernel:
            return
        
        # Create output directory
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        output_dir = f"{base_name}_{prefix}_graphs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get kernel types
        kernel_types = list(metrics_by_kernel.keys())
        
        # Define metrics to visualize (exclude memory metrics for now due to scale differences)
        metrics_to_plot = {
            'grd': 'Grid Size',
            'wgr': 'Workgroup Size',
            'arch_vgpr': 'Architecture VGPRs',
            'accum_vgpr': 'Accumulator VGPRs', 
            'sgpr': 'Scalar GPRs',
            'wave_size': 'Wavefront Size'
        }
        
        # Create a separate graph for each metric
        for metric_col, metric_display in metrics_to_plot.items():
            plt.figure(figsize=(14, 8))
            
            # Extract metric values for this metric across all kernel types
            metric_values = []
            valid_kernels = []
            
            for kernel_type in kernel_types:
                if metric_col in metrics_by_kernel[kernel_type]:
                    metric_values.append(metrics_by_kernel[kernel_type][metric_col]['avg'])
                    valid_kernels.append(kernel_type)
            
            if not metric_values:
                print(f"No data found for {metric_display}")
                plt.close()
                continue
            
            # Create bar chart
            x = np.arange(len(valid_kernels))
            bars = plt.bar(x, metric_values, color='skyblue', alpha=0.8)
            
            plt.xlabel('Kernel Type', fontsize=12)
            plt.ylabel(metric_display, fontsize=12)
            plt.title(f'{metric_display} by Kernel Type (ROCm)', fontsize=14, pad=20)
            
            # Improve x-axis labels to avoid cutoff
            x_labels = []
            for kernel_type in valid_kernels:
                if len(kernel_type) > 15:
                    x_labels.append(kernel_type[:15] + "...")
                else:
                    x_labels.append(kernel_type)
            
            # Rotate x-axis labels for better readability
            plt.xticks(x, x_labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, v in enumerate(metric_values):
                if metric_col == 'grd' and v > 1000:
                    # Format large grid sizes
                    plt.text(i, v + max(metric_values) * 0.01, f'{v/1000:.0f}K', 
                            ha='center', va='bottom', fontsize=9)
                else:
                    plt.text(i, v + max(metric_values) * 0.01, f'{v:.0f}', 
                            ha='center', va='bottom', fontsize=9)
            
            # Add grid for better readability
            plt.grid(axis='y', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            filename = metric_display.replace(' ', '_').replace('(', '').replace(')', '').lower()
            filepath = os.path.join(output_dir, f'metric_{filename}.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
        
        # Create memory metrics visualization (separate due to different scales)
        memory_metrics = ['lds', 'scr']
        available_memory_metrics = []
        
        for metric_col in memory_metrics:
            # Check if any kernel has this metric
            has_data = any(metric_col in metrics_by_kernel[kt] for kt in kernel_types)
            if has_data:
                available_memory_metrics.append(metric_col)
        
        if available_memory_metrics:
            plt.figure(figsize=(14, 8))
            
            # Create grouped bar chart for memory metrics
            x = np.arange(len(kernel_types))
            width = 0.35
            
            for i, metric_col in enumerate(available_memory_metrics):
                metric_values = []
                for kernel_type in kernel_types:
                    if metric_col in metrics_by_kernel[kernel_type]:
                        # Convert to KB
                        value_kb = metrics_by_kernel[kernel_type][metric_col]['avg'] / 1024
                        metric_values.append(value_kb)
                    else:
                        metric_values.append(0)
                
                offset = (i - len(available_memory_metrics)/2 + 0.5) * width
                metric_display = 'LDS (KB)' if metric_col == 'lds' else 'Scratch (KB)'
                plt.bar(x + offset, metric_values, width, label=metric_display, alpha=0.8)
            
            plt.xlabel('Kernel Type', fontsize=12)
            plt.ylabel('Memory (KB)', fontsize=12)
            plt.title('Memory Usage by Kernel Type (ROCm)', fontsize=14, pad=20)
            
            # Improve x-axis labels
            x_labels = []
            for kernel_type in kernel_types:
                if len(kernel_type) > 15:
                    x_labels.append(kernel_type[:15] + "...")
                else:
                    x_labels.append(kernel_type)
            
            plt.xticks(x, x_labels, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save memory metrics plot
            filepath = os.path.join(output_dir, 'metric_memory_usage.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
        
        print(f"\nHardware metrics visualizations saved in '{output_dir}/' folder:")
        for metric_col, metric_display in metrics_to_plot.items():
            filename = metric_display.replace(' ', '_').replace('(', '').replace(')', '').lower()
            print(f"- metric_{filename}.png - {metric_display} by Kernel Type")
        if available_memory_metrics:
            print(f"- metric_memory_usage.png - Memory Usage by Kernel Type") 