#!/usr/bin/env python3
"""
Layer Embedding Dimension Analysis Script

This script analyzes NSys traces for individual model layers across different
embedding dimensions. It generates visualizations showing how GPU metrics
vary with embedding dimension for each layer.

Usage:
    python layer_embedding_analysis.py <trace_directory>
    
Example:
    python layer_embedding_analysis.py /path/to/40GB_layer_seperated/
"""

import os
import sys
import re
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from nsys_utils import NSysAnalyzer


class LayerEmbeddingAnalyzer:
    """Analyze layer traces across different embedding dimensions"""
    
    def __init__(self, base_directory: str):
        """
        Initialize the analyzer
        
        Args:
            base_directory: Path to directory containing layer subdirectories
        """
        self.base_directory = base_directory
        self.layers = self._discover_layers()
        self.embedding_dims = [256, 384, 512, 768, 1024]
        self.metric_map = {
            4: "SM Issue [Throughput %]",
            5: "Tensor Active [Throughput %]", 
            12: "Compute Warps in Flight [Throughput %]",
            15: "Unallocated Warps in Active SMs [Throughput %]",
            18: "DRAM Read Bandwidth [Throughput %]",
            19: "DRAM Write Bandwidth [Throughput %]"
        }
        
    def _discover_layers(self) -> List[str]:
        """Discover all layer subdirectories"""
        layers = []
        if not os.path.exists(self.base_directory):
            print(f"Error: Directory {self.base_directory} does not exist")
            return layers
            
        for item in os.listdir(self.base_directory):
            layer_path = os.path.join(self.base_directory, item)
            if os.path.isdir(layer_path):
                layers.append(item)
        
        layers.sort()
        print(f"Discovered {len(layers)} layers: {', '.join(layers)}")
        return layers
    
    def _find_trace_file(self, layer: str, embed_dim: int) -> str:
        """
        Find the .sqlite trace file for a given layer and embedding dimension
        
        Args:
            layer: Layer name
            embed_dim: Embedding dimension
            
        Returns:
            Path to the .sqlite file, or None if not found
        """
        layer_dir = os.path.join(self.base_directory, layer)
        pattern = f"profile_{layer}_embed{embed_dim}.sqlite"
        
        trace_path = os.path.join(layer_dir, pattern)
        if os.path.exists(trace_path):
            return trace_path
        
        # Try to find any matching file
        for file in os.listdir(layer_dir):
            if file.endswith('.sqlite') and f'embed{embed_dim}' in file:
                return os.path.join(layer_dir, file)
        
        return None
    
    def analyze_layer_trace(self, trace_path: str) -> Dict:
        """
        Analyze a single trace file and extract metrics and runtime
        
        Args:
            trace_path: Path to .sqlite trace file
            
        Returns:
            Dictionary containing metrics and runtime information
        """
        analyzer = NSysAnalyzer(trace_path)
        analyzer.connect()
        
        try:
            # Check if kernel table exists
            if not analyzer.table_exists('CUPTI_ACTIVITY_KIND_KERNEL'):
                print(f"  Warning: No kernel data in {os.path.basename(trace_path)}")
                return {'metrics': {}, 'runtime': 0, 'kernel_count': 0}
            
            # Fetch all kernels
            kernels = analyzer.fetch_all_kernels()
            
            if not kernels:
                print(f"  Warning: No kernels found in {os.path.basename(trace_path)}")
                return {'metrics': {}, 'runtime': 0, 'kernel_count': 0}
            
            # Get metrics for all kernels
            metrics = analyzer.get_metrics_by_kernel(kernels)
            
            # Calculate total runtime
            total_runtime = sum(kernel[4] for kernel in kernels)  # kernel[4] is duration
            
            return {
                'metrics': metrics,
                'runtime': total_runtime,
                'kernel_count': len(kernels)
            }
        
        finally:
            analyzer.disconnect()
    
    def analyze_layer_across_embeddings(self, layer: str) -> Dict[int, Dict]:
        """
        Analyze a layer across all embedding dimensions
        
        Args:
            layer: Layer name
            
        Returns:
            Dictionary mapping embedding dimension to analysis results
        """
        print(f"\nAnalyzing layer: {layer}")
        results = {}
        
        for embed_dim in self.embedding_dims:
            trace_path = self._find_trace_file(layer, embed_dim)
            
            if trace_path is None:
                print(f"  Warning: No trace file found for {layer} with embedding {embed_dim}")
                continue
            
            print(f"  Processing embedding dimension {embed_dim}...")
            analysis = self.analyze_layer_trace(trace_path)
            results[embed_dim] = analysis
            
            print(f"    Kernels: {analysis['kernel_count']}, Runtime: {analysis['runtime']/1e6:.2f} ms")
        
        return results
    
    def create_layer_visualizations(self, layer: str, results: Dict[int, Dict]):
        """
        Create visualizations for a layer showing metrics vs embedding dimension
        
        Args:
            layer: Layer name
            results: Dictionary mapping embedding dimension to analysis results
        """
        if not results:
            print(f"No data to visualize for layer {layer}")
            return
        
        # Create output directory
        output_dir = f"layer_embedding_analysis/{layer}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract embedding dimensions and sort
        embed_dims = sorted(results.keys())
        
        # Create a graph for each metric
        for metric_id, metric_name in self.metric_map.items():
            plt.figure(figsize=(12, 7))
            
            # Extract metric values for this metric
            metric_values = []
            valid_dims = []
            
            for dim in embed_dims:
                metrics = results[dim].get('metrics', {})
                if metric_id in metrics:
                    metric_values.append(metrics[metric_id])
                    valid_dims.append(dim)
            
            if not metric_values:
                print(f"  No data for {metric_name}")
                plt.close()
                continue
            
            # Create line plot with markers
            plt.plot(valid_dims, metric_values, marker='o', linewidth=2, 
                    markersize=8, color='steelblue', label=metric_name)
            
            # Add value labels
            for i, (dim, val) in enumerate(zip(valid_dims, metric_values)):
                plt.text(dim, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # Formatting
            plt.xlabel('Embedding Dimension', fontsize=12)
            plt.ylabel('Throughput %', fontsize=12)
            plt.title(f'{layer.upper()} - {metric_name} vs Embedding Dimension', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(embed_dims)
            plt.ylim(0, 100)  # Set y-axis range for percentage metrics
            
            # Save the plot
            filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f'{filename}.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
        
        # Create runtime graph
        plt.figure(figsize=(12, 7))
        
        runtimes_ms = [results[dim]['runtime'] / 1e6 for dim in embed_dims]
        
        plt.plot(embed_dims, runtimes_ms, marker='s', linewidth=2, 
                markersize=8, color='coral', label='Runtime')
        
        # Add value labels
        for dim, runtime in zip(embed_dims, runtimes_ms):
            plt.text(dim, runtime, f'{runtime:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        # Formatting
        plt.xlabel('Embedding Dimension', fontsize=12)
        plt.ylabel('Runtime (ms)', fontsize=12)
        plt.title(f'{layer.upper()} - Runtime vs Embedding Dimension', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(embed_dims)
        
        # Save the plot
        filepath = os.path.join(output_dir, 'runtime.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create kernel count graph
        plt.figure(figsize=(12, 7))
        
        kernel_counts = [results[dim]['kernel_count'] for dim in embed_dims]
        
        plt.plot(embed_dims, kernel_counts, marker='^', linewidth=2, 
                markersize=8, color='green', label='Kernel Count')
        
        # Add value labels
        for dim, count in zip(embed_dims, kernel_counts):
            plt.text(dim, count, f'{count}', ha='center', va='bottom', fontsize=10)
        
        # Formatting
        plt.xlabel('Embedding Dimension', fontsize=12)
        plt.ylabel('Number of Kernels', fontsize=12)
        plt.title(f'{layer.upper()} - Kernel Count vs Embedding Dimension', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(embed_dims)
        
        # Save the plot
        filepath = os.path.join(output_dir, 'kernel_count.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Visualizations saved in '{output_dir}/'")
    
    def create_comparative_visualizations(self, all_results: Dict[str, Dict[int, Dict]]):
        """
        Create comparative visualizations across all layers
        
        Args:
            all_results: Nested dictionary of layer -> embedding_dim -> results
        """
        if not all_results:
            return
        
        output_dir = "layer_embedding_analysis/comparative"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get sorted lists
        layers = sorted(all_results.keys())
        embed_dims = sorted(self.embedding_dims)
        
        # Create a comparative graph for each metric
        for metric_id, metric_name in self.metric_map.items():
            plt.figure(figsize=(14, 8))
            
            for layer in layers:
                metric_values = []
                valid_dims = []
                
                for dim in embed_dims:
                    if dim in all_results[layer]:
                        metrics = all_results[layer][dim].get('metrics', {})
                        if metric_id in metrics:
                            metric_values.append(metrics[metric_id])
                            valid_dims.append(dim)
                
                if metric_values:
                    plt.plot(valid_dims, metric_values, marker='o', linewidth=2, 
                            markersize=6, label=layer.upper())
            
            # Formatting
            plt.xlabel('Embedding Dimension', fontsize=12)
            plt.ylabel('Throughput %', fontsize=12)
            plt.title(f'All Layers - {metric_name} vs Embedding Dimension', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(embed_dims)
            plt.ylim(0, 100)  # Set y-axis range for percentage metrics
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save
            filename = metric_name.replace('[', '').replace(']', '').replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f'comparative_{filename}.png')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
        
        # Create comparative runtime graph
        plt.figure(figsize=(14, 8))
        
        for layer in layers:
            runtimes_ms = []
            valid_dims = []
            
            for dim in embed_dims:
                if dim in all_results[layer]:
                    runtime = all_results[layer][dim]['runtime'] / 1e6
                    runtimes_ms.append(runtime)
                    valid_dims.append(dim)
            
            if runtimes_ms:
                plt.plot(valid_dims, runtimes_ms, marker='s', linewidth=2, 
                        markersize=6, label=layer.upper())
        
        # Formatting
        plt.xlabel('Embedding Dimension', fontsize=12)
        plt.ylabel('Runtime (ms)', fontsize=12)
        plt.title('All Layers - Runtime vs Embedding Dimension', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(embed_dims)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save
        filepath = os.path.join(output_dir, 'comparative_runtime.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"\nComparative visualizations saved in '{output_dir}/'")
    
    def run_analysis(self):
        """Run the complete analysis for all layers"""
        if not self.layers:
            print("No layers found to analyze")
            return
        
        print(f"\n{'='*60}")
        print(f"Layer Embedding Dimension Analysis")
        print(f"{'='*60}")
        print(f"Base directory: {self.base_directory}")
        print(f"Embedding dimensions: {self.embedding_dims}")
        print(f"Layers to analyze: {len(self.layers)}")
        
        all_results = {}
        
        for layer in self.layers:
            results = self.analyze_layer_across_embeddings(layer)
            all_results[layer] = results
            
            if results:
                self.create_layer_visualizations(layer, results)
        
        # Create comparative visualizations
        self.create_comparative_visualizations(all_results)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Analysis Complete!")
        print(f"{'='*60}")
        print(f"\nResults organized in 'layer_embedding_analysis/' directory:")
        print(f"  - Individual layer folders with per-layer graphs")
        print(f"  - 'comparative/' folder with cross-layer comparisons")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python layer_embedding_analysis.py <trace_directory>")
        print("\nExample:")
        print("  python layer_embedding_analysis.py /path/to/40GB_layer_seperated/")
        sys.exit(1)
    
    trace_directory = sys.argv[1]
    
    if not os.path.exists(trace_directory):
        print(f"Error: Directory '{trace_directory}' does not exist")
        sys.exit(1)
    
    analyzer = LayerEmbeddingAnalyzer(trace_directory)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

