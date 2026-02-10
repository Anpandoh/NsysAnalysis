#!/usr/bin/env python3
"""
Kernel Embedding Dimension Analysis Script

This script analyzes NSys traces for kernel performance across different
embedding dimensions. It generates visualizations showing how GPU metrics
and kernel performance vary with embedding dimension.

Usage:
    python kernel_embedding_analysis.py <trace_pattern>
    
Example:
    python kernel_embedding_analysis.py sfno_{}_20_a100_correlated.sqlite
"""

import os
import sys
import re
import matplotlib.pyplot as plt
from typing import Dict, List
from nsys_utils import NSysAnalyzer

# IEEE-style matplotlib configuration
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 1.2
})


class KernelEmbeddingAnalyzer:
    """Analyze kernel traces across different embedding dimensions"""
    
    def __init__(self, trace_pattern: str):
        """
        Initialize the analyzer
        
        Args:
            trace_pattern: Pattern for trace files with {} placeholder for embedding dimension
                          e.g., "sfno_{}_20_a100_correlated.sqlite"
        """
        self.trace_pattern = trace_pattern
        self.embedding_dims = [256, 384, 512, 768, 1024]
        self.metric_map = {
            4: "SM Issue [Throughput %]",
            5: "Tensor Active [Throughput %]", 
            12: "Compute Warps in Flight [Throughput %]",
            15: "Unallocated Warps in Active SMs [Throughput %]",
            18: "DRAM Read Bandwidth [Throughput %]",
            19: "DRAM Write Bandwidth [Throughput %]"
        }
        
    def _find_trace_files(self) -> Dict[int, str]:
        """
        Find trace files for each embedding dimension
        
        Returns:
            Dictionary mapping embedding dimension to trace file path
        """
        trace_files = {}
        
        for dim in self.embedding_dims:
            trace_path = self.trace_pattern.format(dim)
            if os.path.exists(trace_path):
                trace_files[dim] = trace_path
            else:
                print(f"Warning: No trace file found for embedding dimension {dim}: {trace_path}")
        
        print(f"Found trace files for embedding dimensions: {sorted(trace_files.keys())}")
        return trace_files
    
    def analyze_trace_kernels(self, trace_path: str, max_kernel_types: int = 10) -> Dict:
        """
        Analyze kernel performance for a single trace file
        
        Args:
            trace_path: Path to .sqlite trace file
            max_kernel_types: Maximum number of kernel types to analyze
            
        Returns:
            Dictionary containing kernel analysis results
        """
        analyzer = NSysAnalyzer(trace_path)
        analyzer.connect()
        
        try:
            # Check if kernel table exists
            if not analyzer.table_exists('CUPTI_ACTIVITY_KIND_KERNEL'):
                print(f"Warning: No kernel data in {os.path.basename(trace_path)}")
                return {'kernel_types': {}, 'total_runtime': 0, 'kernel_count': 0}
            
            # Fetch all kernels
            kernels = analyzer.fetch_all_kernels()
            
            if not kernels:
                print(f"Warning: No kernels found in {os.path.basename(trace_path)}")
                return {'kernel_types': {}, 'total_runtime': 0, 'kernel_count': 0}
            
            # Group kernels by type
            kernel_types = analyzer.group_kernels_by_type(kernels)
            
            # Calculate total duration for each kernel type (excluding first 4 iterations)
            kernel_analysis = {}
            total_runtime = sum(kernel[4] for kernel in kernels)
            
            sum_duration_by_type = []
            for kernel_type, type_kernels in kernel_types.items():
                # Skip first 4 kernels of each type (startup outliers)
                kernels_filtered = type_kernels[4:] if len(type_kernels) > 4 else []
                if kernels_filtered:
                    total_duration = sum(k[4] for k in kernels_filtered)
                    avg_duration = total_duration / len(kernels_filtered)
                    sum_duration_by_type.append((kernel_type, total_duration, avg_duration, kernels_filtered))
            
            # Sort by total duration and select top kernel types
            sum_duration_by_type.sort(key=lambda x: x[1], reverse=True)
            top_kernel_types = sum_duration_by_type[:max_kernel_types]
            
            # Get metrics for each top kernel type
            for kernel_type, total_duration, avg_duration, type_kernels in top_kernel_types:
                # Use middle kernels for large datasets
                kernels_no_startup = type_kernels[4:] if len(type_kernels) > 4 else type_kernels
                if len(kernels_no_startup) > 2000:
                    start_idx = (len(kernels_no_startup) - 2000) // 2
                    kernels_for_metrics = kernels_no_startup[start_idx:start_idx + 2000]
                else:
                    kernels_for_metrics = kernels_no_startup
                
                kernel_metrics = analyzer.get_metrics_by_kernel(kernels_for_metrics, batch_size=500)
                
                kernel_analysis[kernel_type] = {
                    'total_duration': total_duration,
                    'avg_duration': avg_duration,
                    'num_calls': len(kernels_filtered),
                    'metrics': kernel_metrics
                }
            
            return {
                'kernel_types': kernel_analysis,
                'total_runtime': total_runtime,
                'kernel_count': len(kernels)
            }
        
        finally:
            analyzer.disconnect()
    
    def analyze_kernels_across_embeddings(self) -> Dict[int, Dict]:
        """
        Analyze kernel performance across all embedding dimensions
        
        Returns:
            Dictionary mapping embedding dimension to kernel analysis results
        """
        print(f"\n{'='*60}")
        print("Kernel Embedding Dimension Analysis")
        print(f"{'='*60}")
        print(f"Trace pattern: {self.trace_pattern}")
        print(f"Embedding dimensions: {self.embedding_dims}")
        
        trace_files = self._find_trace_files()
        if not trace_files:
            print("No trace files found!")
            return {}
        
        results = {}
        
        for embed_dim in sorted(trace_files.keys()):
            trace_path = trace_files[embed_dim]
            print(f"\n--- Processing embedding dimension {embed_dim} ---")
            print(f"Trace file: {os.path.basename(trace_path)}")
            
            analysis = self.analyze_trace_kernels(trace_path)
            results[embed_dim] = analysis
            
            print(f"Total kernels: {analysis['kernel_count']}")
            print(f"Total runtime: {analysis['total_runtime']/1e6:.2f} ms")
            print(f"Top kernel types found: {len(analysis['kernel_types'])}")
        
        return results
    
    def create_individual_kernel_visualizations(self, kernel_type: str, results: Dict[int, Dict]):
        """
        Create visualizations for a specific kernel type across embedding dimensions
        
        Args:
            kernel_type: Name of the kernel type
            results: Dictionary mapping embedding dimension to analysis results
        """
        # Extract embedding dimensions and sort
        embed_dims = sorted(results.keys())
        
        # Create output directory for this kernel type
        safe_kernel_name = re.sub(r'[<>:"/\\|?*]', '_', kernel_type)
        output_dir = f"kernel_embedding_analysis/{safe_kernel_name[:50]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if this kernel exists across dimensions
        kernel_data = {}
        for dim in embed_dims:
            if kernel_type in results[dim]['kernel_types']:
                kernel_data[dim] = results[dim]['kernel_types'][kernel_type]
        
        if not kernel_data:
            print(f"No data found for kernel type: {kernel_type}")
            return
        
        valid_dims = sorted(kernel_data.keys())
        
        # Create IEEE-style graphs for each metric
        for metric_id, metric_name in self.metric_map.items():
            # Extract metric values for this metric
            metric_values = []
            metric_dims = []
            
            for dim in valid_dims:
                metrics = kernel_data[dim].get('metrics', {})
                if metric_id in metrics:
                    metric_values.append(metrics[metric_id])
                    metric_dims.append(dim)
            
            if not metric_values:
                print(f"No data for {metric_name} for kernel {kernel_type}")
                continue
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Create line plot with markers
            ax.plot(metric_dims, metric_values, marker='o', 
                   color='#2E86C1', linewidth=2.5, markersize=8)
            
            # Add value labels
            for dim, val in zip(metric_dims, metric_values):
                ax.text(dim, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # Add vertical lines for ACE models if they're in the metric range
            if 256 in metric_dims:
                ax.axvline(x=256, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(256, ax.get_ylim()[1] * 0.90, 'ACE', rotation=0, 
                       fontsize=11, ha='center', va='center', color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red'))
            
            if 384 in metric_dims:
                ax.axvline(x=384, color='blue', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(384, ax.get_ylim()[1] * 0.90, 'ACE2', rotation=0, 
                       fontsize=11, ha='center', va='center', color='blue', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
            
            # IEEE-style formatting
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Throughput (%)')
            ax.set_xticks(metric_dims)
            ax.set_ylim(0, 100)
            
            # Save the plot
            filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f'{filename}.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
        
        # Create runtime graphs
        self._create_kernel_runtime_graphs(kernel_type, kernel_data, output_dir, valid_dims)
        
        print(f"Visualizations saved in '{output_dir}/'")
    
    def _create_kernel_runtime_graphs(self, kernel_type: str, kernel_data: Dict, 
                                    output_dir: str, valid_dims: List[int]):
        """Create IEEE-style runtime graphs for a kernel type"""
        
        # Total duration graph
        fig, ax = plt.subplots(figsize=(8, 5))
        total_durations = [kernel_data[dim]['total_duration'] / 1e6 for dim in valid_dims]
        
        ax.plot(valid_dims, total_durations, marker='s', 
               color='#E74C3C', linewidth=2.5, markersize=8)
        
        for dim, duration in zip(valid_dims, total_durations):
            ax.text(dim, duration, f'{duration:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        # Add vertical lines for ACE models
        ax.axvline(x=256, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(256, ax.get_ylim()[1] * 0.95, 'ACE', rotation=0, 
               fontsize=11, ha='center', va='top', color='red', fontweight='bold')
        
        ax.axvline(x=384, color='blue', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(384, ax.get_ylim()[1] * 0.95, 'ACE2', rotation=0, 
               fontsize=11, ha='center', va='top', color='blue', fontweight='bold')
        
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Total Duration (ms)')
        ax.set_xticks(valid_dims)
        
        filepath = os.path.join(output_dir, 'total_duration.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # Average duration graph
        fig, ax = plt.subplots(figsize=(8, 5))
        avg_durations = [kernel_data[dim]['avg_duration'] / 1e6 for dim in valid_dims]
        
        ax.plot(valid_dims, avg_durations, marker='^', 
               color='#27AE60', linewidth=2.5, markersize=8)
        
        for dim, duration in zip(valid_dims, avg_durations):
            ax.text(dim, duration, f'{duration:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        # Add vertical lines for ACE models
        ax.axvline(x=256, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(256, ax.get_ylim()[1] * 0.95, 'ACE', rotation=0, 
               fontsize=11, ha='center', va='top', color='red', fontweight='bold')
        
        ax.axvline(x=384, color='blue', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(384, ax.get_ylim()[1] * 0.95, 'ACE2', rotation=0, 
               fontsize=11, ha='center', va='top', color='blue', fontweight='bold')
        
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Average Duration (ms)')
        ax.set_xticks(valid_dims)
        
        filepath = os.path.join(output_dir, 'average_duration.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
    
    def create_comparative_visualizations(self, all_results: Dict[int, Dict]):
        """
        Create comparative visualizations across all kernel types and embedding dimensions
        
        Args:
            all_results: Dictionary mapping embedding dimension to kernel analysis results
        """
        if not all_results:
            return
        
        output_dir = "kernel_embedding_analysis/comparative"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all kernel types that appear across dimensions
        all_kernel_types = set()
        for dim_results in all_results.values():
            all_kernel_types.update(dim_results['kernel_types'].keys())
        
        all_kernel_types = sorted(list(all_kernel_types))
        embed_dims = sorted(all_results.keys())
        
        print(f"Creating comparative visualizations for {len(all_kernel_types)} kernel types...")
        
        # Define standard colors and markers
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', 'p']
        
        # Create compact comparative graphs for each metric
        for metric_id, metric_name in self.metric_map.items():
            fig, ax = plt.subplots(figsize=(7, 4.5))
            
            # Collect all kernel data for this metric and filter significant ones
            kernel_metric_data = []
            
            for kernel_type in all_kernel_types:
                metric_values = []
                valid_dims = []
                
                for dim in embed_dims:
                    if kernel_type in all_results[dim]['kernel_types']:
                        metrics = all_results[dim]['kernel_types'][kernel_type].get('metrics', {})
                        if metric_id in metrics:
                            metric_values.append(metrics[metric_id])
                            valid_dims.append(dim)
                
                if metric_values and len(metric_values) >= 2:
                    # Only include kernels with significant throughput variation or high values
                    max_val = max(metric_values)
                    variation = max(metric_values) - min(metric_values)
                    if max_val > 20 or variation > 10:  # High throughput or significant variation
                        kernel_metric_data.append((kernel_type, valid_dims, metric_values, max_val))
            
            # Sort by maximum throughput value and take top 6
            kernel_metric_data.sort(key=lambda x: x[3], reverse=True)
            top_kernels = kernel_metric_data[:6]
            
            # Plot significant kernels
            for i, (kernel_type, valid_dims, metric_values, _) in enumerate(top_kernels):
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                display_name = self._get_kernel_display_name(kernel_type)
                
                ax.plot(valid_dims, metric_values, 
                       color=color, marker=marker, linewidth=2.0, markersize=5,
                       label=display_name)
            
            # Compact formatting
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Throughput (%)')
            
            # Shorter titles
            title_map = {
                "SM Issue [Throughput %]": "SM Issue Efficiency",
                "Tensor Active [Throughput %]": "Tensor Core Utilization", 
                "Compute Warps in Flight [Throughput %]": "Compute Warps Active",
                "Unallocated Warps in Active SMs [Throughput %]": "Unallocated Warps",
                "DRAM Read Bandwidth [Throughput %]": "DRAM Read Bandwidth",
                "DRAM Write Bandwidth [Throughput %]": "DRAM Write Bandwidth"
            }
            
            ax.set_xticks(embed_dims)
            ax.set_ylim(0, 100)
            
            # Horizontal legend at the top
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                     ncol=3, fontsize=9, frameon=True, fancybox=False, shadow=False)
            
            # Add vertical lines for ACE models (after legend positioning)
            ax.axvline(x=256, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.text(256, ax.get_ylim()[1] * 0.85, 'ACE', rotation=0, 
                   fontsize=10, ha='center', va='center', color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red'))
            
            ax.axvline(x=384, color='blue', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.text(384, ax.get_ylim()[1] * 0.85, 'ACE2', rotation=0, 
                   fontsize=10, ha='center', va='center', color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
            
            plt.tight_layout()
            
            # Save
            filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f'comparative_{filename}.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
        
        # Create comparative runtime graphs
        self._create_comparative_runtime_graphs(all_results, output_dir, all_kernel_types, embed_dims)
        
        print(f"Comparative visualizations saved in '{output_dir}/'")
    
    def _get_kernel_display_name(self, kernel_name: str) -> str:
        """Convert kernel names to shorter, more descriptive names"""
        name_map = {
            'EK_CUDAFunctor_direct_copy_kernel_cuda': 'Data Copy',
            'ampere_cgemm_64x64_nn': 'Complex GEMM',
            'cutlass_gemm_256x64': 'CUTLASS GEMM', 
            'ampere_sgemm_128x128_nn': 'SGEMM NN',
            'ampere_sgemm_128x128_tn': 'SGEMM TN',
            'VEK_GeluCUDAKernelImpl': 'GELU Activation',
            'cudnn::bn_fw_tr_1C11_kernel_NCHW': 'BatchNorm',
            'VEK_CUDAFunctor_add': 'Vector Add',
            'VEK_AUnaryFunctor': 'Unary Ops',
            'EK_CUDAFunctor_CUDAFunctor_add': 'Element Add'
        }
        
        for key, display_name in name_map.items():
            if key in kernel_name:
                return display_name
        return kernel_name[:15] + "..." if len(kernel_name) > 15 else kernel_name
    
    def _create_comparative_runtime_graphs(self, all_results: Dict[int, Dict], 
                                         output_dir: str, all_kernel_types: List[str], 
                                         embed_dims: List[int]):
        """Create IEEE-style comparative runtime graphs"""
        
        def should_group_as_others(kernel_name: str, durations_ms: List[float]) -> bool:
            """Determine if a kernel should be grouped into 'Remaining Kernels' based on duration or type"""
            max_duration = max(durations_ms) if durations_ms else 0
            # Always group Element Add with others
            if 'EK_CUDAFunctor_CUDAFunctor_add' in kernel_name:
                return True
            # Group small kernels
            return max_duration < 500  # Group kernels with max duration < 500ms
        
        # Define standard colors and markers for main kernels
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', 'p']
        
        # Total duration comparison - more compact
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        # Separate main kernels from others
        main_kernels = []
        others_data = {dim: [] for dim in embed_dims}  # Store list of durations for averaging
        
        # First pass: identify main kernels and collect others for averaging
        for kernel_type in all_kernel_types:
            durations_ms = []
            valid_dims = []
            
            for dim in embed_dims:
                if kernel_type in all_results[dim]['kernel_types']:
                    duration = all_results[dim]['kernel_types'][kernel_type]['total_duration'] / 1e6
                    durations_ms.append(duration)
                    valid_dims.append(dim)
            
            if durations_ms and len(durations_ms) >= 2:
                if should_group_as_others(kernel_type, durations_ms):
                    # Add to remaining kernels list for averaging
                    for dim, duration in zip(valid_dims, durations_ms):
                        others_data[dim].append(duration)
                else:
                    main_kernels.append((kernel_type, valid_dims, durations_ms))
        
        # Sort main kernels by peak duration
        main_kernels.sort(key=lambda x: max(x[2]), reverse=True)
        
        # Plot main kernels
        for i, (kernel_type, valid_dims, durations_ms) in enumerate(main_kernels):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            display_name = self._get_kernel_display_name(kernel_type)
            
            ax.plot(valid_dims, durations_ms, 
                   color=color, marker=marker, linewidth=2.0, markersize=5,
                   label=display_name)
        
        # Plot remaining kernel average if significant
        others_averages = []
        for dim in embed_dims:
            if others_data[dim]:  # If there are kernels in this dimension
                avg = sum(others_data[dim]) / len(others_data[dim])
                others_averages.append(avg)
            else:
                others_averages.append(0)
        
        if max(others_averages) > 10:  # Only show if average remaining kernel > 10ms
            ax.plot(embed_dims, others_averages, 
                   color='#95A5A6', marker='x', linewidth=1.5, markersize=4,
                   label='Remaining Kernel Average', linestyle='--', alpha=0.8)
        
        # Compact formatting
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Total Duration (ms)')
        ax.set_xticks(embed_dims)
        
        # Horizontal legend at the top
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                 ncol=3, fontsize=9, frameon=True, fancybox=False, shadow=False)
        
        # Add vertical lines for ACE models (after legend to avoid overlap)
        ax.axvline(x=256, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.text(256, ax.get_ylim()[1] * 0.85, 'ACE', rotation=0, 
               fontsize=10, ha='center', va='center', color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red'))
        
        ax.axvline(x=384, color='blue', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.text(384, ax.get_ylim()[1] * 0.85, 'ACE2', rotation=0, 
               fontsize=10, ha='center', va='center', color='blue', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
        
        # Tighter layout
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'comparative_total_duration.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # Average duration comparison - more compact
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        # Only plot significant kernels for average duration
        avg_main_kernels = []
        
        for kernel_type in all_kernel_types:
            avg_durations_ms = []
            valid_dims = []
            
            for dim in embed_dims:
                if kernel_type in all_results[dim]['kernel_types']:
                    avg_duration = all_results[dim]['kernel_types'][kernel_type]['avg_duration'] / 1e6
                    avg_durations_ms.append(avg_duration)
                    valid_dims.append(dim)
            
            if avg_durations_ms and len(avg_durations_ms) >= 2:
                max_avg = max(avg_durations_ms)
                if max_avg > 0.1:  # Only include kernels with avg > 0.1ms
                    avg_main_kernels.append((kernel_type, valid_dims, avg_durations_ms))
        
        # Sort by peak average duration
        avg_main_kernels.sort(key=lambda x: max(x[2]), reverse=True)
        
        # Plot main kernels
        for i, (kernel_type, valid_dims, avg_durations_ms) in enumerate(avg_main_kernels[:6]):  # Top 6 only
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            display_name = self._get_kernel_display_name(kernel_type)
            
            ax.plot(valid_dims, avg_durations_ms, 
                   color=color, marker=marker, linewidth=2.0, markersize=5,
                   label=display_name)
        
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Average Duration (ms)')
        ax.set_xticks(embed_dims)
        
        # Horizontal legend at the top
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                 ncol=3, fontsize=9, frameon=True, fancybox=False, shadow=False)
        
        # Add vertical lines for ACE models (after legend positioning)
        ax.axvline(x=256, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.text(256, ax.get_ylim()[1] * 0.85, 'ACE', rotation=0, 
               fontsize=10, ha='center', va='center', color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red'))
        
        ax.axvline(x=384, color='blue', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.text(384, ax.get_ylim()[1] * 0.85, 'ACE2', rotation=0, 
               fontsize=10, ha='center', va='center', color='blue', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'comparative_average_duration.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    
    def run_analysis(self):
        """Run the complete kernel embedding analysis"""
        # Analyze kernels across embedding dimensions
        all_results = self.analyze_kernels_across_embeddings()
        
        if not all_results:
            print("No analysis results available")
            return
        
        # Get all unique kernel types across all dimensions
        all_kernel_types = set()
        for dim_results in all_results.values():
            all_kernel_types.update(dim_results['kernel_types'].keys())
        
        print(f"Found {len(all_kernel_types)} unique kernel types across all dimensions:")
        for kernel_type in sorted(all_kernel_types):
            print(f"  - {kernel_type}")
        
        # Create individual visualizations for each kernel type
        print("Creating individual kernel visualizations...")
        for kernel_type in sorted(all_kernel_types):
            print(f"Processing kernel type: {kernel_type[:50]}...")
            self.create_individual_kernel_visualizations(kernel_type, all_results)
        
        # Create comparative visualizations
        self.create_comparative_visualizations(all_results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Kernel Embedding Analysis Complete!")
        print(f"{'='*60}")
        print("\nResults organized in 'kernel_embedding_analysis/' directory:")
        print("  - Individual kernel folders with per-kernel graphs")
        print("  - 'comparative/' folder with cross-kernel comparisons")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python kernel_embedding_analysis.py <trace_pattern>")
        print("\nExample:")
        print("  python kernel_embedding_analysis.py sfno_{}_20_a100_correlated.sqlite")
        print("  python kernel_embedding_analysis.py ace2_nvtx_400_50_correlated.sqlite")
        print("\nThe {} placeholder will be replaced with embedding dimensions: 256, 384, 512, 768, 1024")
        sys.exit(1)
    
    trace_pattern = sys.argv[1]
    
    # Validate pattern
    if '{}' not in trace_pattern:
        print("Error: Trace pattern must contain '{}' placeholder for embedding dimension")
        print(f"Got: {trace_pattern}")
        sys.exit(1)
    
    analyzer = KernelEmbeddingAnalyzer(trace_pattern)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()