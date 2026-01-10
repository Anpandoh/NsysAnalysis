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
        
        # Create a graph for each metric
        for metric_id, metric_name in self.metric_map.items():
            plt.figure(figsize=(12, 7))
            
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
                plt.close()
                continue
            
            # Create line plot with markers
            plt.plot(metric_dims, metric_values, marker='o', linewidth=2, 
                    markersize=8, color='steelblue', label=metric_name)
            
            # Add value labels
            for i, (dim, val) in enumerate(zip(metric_dims, metric_values)):
                plt.text(dim, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # Formatting
            plt.xlabel('Embedding Dimension', fontsize=12)
            plt.ylabel('Throughput %', fontsize=12)
            plt.title(f'{kernel_type} - {metric_name} vs Embedding Dimension', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_dims)
            plt.ylim(0, 100)
            
            # Save the plot
            filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f'{filename}.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
        
        # Create runtime graphs
        self._create_kernel_runtime_graphs(kernel_type, kernel_data, output_dir, valid_dims)
        
        print(f"Visualizations saved in '{output_dir}/'")
    
    def _create_kernel_runtime_graphs(self, kernel_type: str, kernel_data: Dict, 
                                    output_dir: str, valid_dims: List[int]):
        """Create runtime-related graphs for a kernel type"""
        
        # Total duration graph
        plt.figure(figsize=(12, 7))
        total_durations = [kernel_data[dim]['total_duration'] / 1e6 for dim in valid_dims]
        
        plt.plot(valid_dims, total_durations, marker='s', linewidth=2, 
                markersize=8, color='coral', label='Total Duration')
        
        for dim, duration in zip(valid_dims, total_durations):
            plt.text(dim, duration, f'{duration:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Embedding Dimension', fontsize=12)
        plt.ylabel('Total Duration (ms)', fontsize=12)
        plt.title(f'{kernel_type} - Total Duration vs Embedding Dimension', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(valid_dims)
        
        filepath = os.path.join(output_dir, 'total_duration.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Average duration graph
        plt.figure(figsize=(12, 7))
        avg_durations = [kernel_data[dim]['avg_duration'] / 1e6 for dim in valid_dims]
        
        plt.plot(valid_dims, avg_durations, marker='^', linewidth=2, 
                markersize=8, color='green', label='Average Duration')
        
        for dim, duration in zip(valid_dims, avg_durations):
            plt.text(dim, duration, f'{duration:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Embedding Dimension', fontsize=12)
        plt.ylabel('Average Duration (ms)', fontsize=12)
        plt.title(f'{kernel_type} - Average Duration vs Embedding Dimension', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(valid_dims)
        
        filepath = os.path.join(output_dir, 'average_duration.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
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
        
        # Create a comparative graph for each metric
        for metric_id, metric_name in self.metric_map.items():
            plt.figure(figsize=(14, 8))
            
            for kernel_type in all_kernel_types:
                metric_values = []
                valid_dims = []
                
                for dim in embed_dims:
                    if kernel_type in all_results[dim]['kernel_types']:
                        metrics = all_results[dim]['kernel_types'][kernel_type].get('metrics', {})
                        if metric_id in metrics:
                            metric_values.append(metrics[metric_id])
                            valid_dims.append(dim)
                
                if metric_values and len(metric_values) >= 2:  # Only plot if we have multiple points
                    plt.plot(valid_dims, metric_values, marker='o', linewidth=2, 
                            markersize=6, label=kernel_type[:30] + ("..." if len(kernel_type) > 30 else ""))
            
            # Formatting
            plt.xlabel('Embedding Dimension', fontsize=12)
            plt.ylabel('Throughput %', fontsize=12)
            plt.title(f'All Kernels - {metric_name} vs Embedding Dimension', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(embed_dims)
            plt.ylim(0, 100)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save
            filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f'comparative_{filename}.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
        
        # Create comparative runtime graphs
        self._create_comparative_runtime_graphs(all_results, output_dir, all_kernel_types, embed_dims)
        
        print(f"Comparative visualizations saved in '{output_dir}/'")
    
    def _create_comparative_runtime_graphs(self, all_results: Dict[int, Dict], 
                                         output_dir: str, all_kernel_types: List[str], 
                                         embed_dims: List[int]):
        """Create comparative runtime graphs"""
        
        # Total duration comparison
        plt.figure(figsize=(14, 8))
        
        for kernel_type in all_kernel_types:
            durations_ms = []
            valid_dims = []
            
            for dim in embed_dims:
                if kernel_type in all_results[dim]['kernel_types']:
                    duration = all_results[dim]['kernel_types'][kernel_type]['total_duration'] / 1e6
                    durations_ms.append(duration)
                    valid_dims.append(dim)
            
            if durations_ms and len(durations_ms) >= 2:
                plt.plot(valid_dims, durations_ms, marker='s', linewidth=2, 
                        markersize=6, label=kernel_type[:30] + ("..." if len(kernel_type) > 30 else ""))
        
        plt.xlabel('Embedding Dimension', fontsize=12)
        plt.ylabel('Total Duration (ms)', fontsize=12)
        plt.title('All Kernels - Total Duration vs Embedding Dimension', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(embed_dims)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        filepath = os.path.join(output_dir, 'comparative_total_duration.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Average duration comparison
        plt.figure(figsize=(14, 8))
        
        for kernel_type in all_kernel_types:
            avg_durations_ms = []
            valid_dims = []
            
            for dim in embed_dims:
                if kernel_type in all_results[dim]['kernel_types']:
                    avg_duration = all_results[dim]['kernel_types'][kernel_type]['avg_duration'] / 1e6
                    avg_durations_ms.append(avg_duration)
                    valid_dims.append(dim)
            
            if avg_durations_ms and len(avg_durations_ms) >= 2:
                plt.plot(valid_dims, avg_durations_ms, marker='^', linewidth=2, 
                        markersize=6, label=kernel_type[:30] + ("..." if len(kernel_type) > 30 else ""))
        
        plt.xlabel('Embedding Dimension', fontsize=12)
        plt.ylabel('Average Duration (ms)', fontsize=12)
        plt.title('All Kernels - Average Duration vs Embedding Dimension', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(embed_dims)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        filepath = os.path.join(output_dir, 'comparative_average_duration.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
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