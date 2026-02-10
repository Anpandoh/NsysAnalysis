#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot

import matplotlib.pyplot as plt
import os
import numpy as np
from nsys_utils import NSysAnalyzer

# IEEE formatting
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class LayerStackedAnalyzer:
    def __init__(self, trace_pattern: str):
        self.trace_pattern = trace_pattern
        self.embedding_dims = [256, 384, 512, 768, 1024]

    def _find_trace_files(self):
        trace_files = {}
        for dim in self.embedding_dims:
            trace_file = self.trace_pattern.format(dim)
            if os.path.exists(trace_file):
                trace_files[dim] = trace_file
        return trace_files

    def analyze_and_visualize(self):
        print("=" * 60)
        print("NVTX Layer Analysis - Stacked Percentage View")
        print("=" * 60)
        
        trace_files = self._find_trace_files()
        print(f"Found trace files: {list(trace_files.keys())}")
        
        results = {}
        
        # Analyze each dimension
        for embed_dim in sorted(trace_files.keys()):
            trace_file = trace_files[embed_dim]
            print(f"\\n--- Processing dimension {embed_dim} ---")
            
            layer_analysis = self._analyze_single_trace(trace_file)
            results[embed_dim] = layer_analysis
        
        # Create visualizations
        self._create_stacked_runtime_plot(results)
        self._create_line_plot(results)
        
        return results

    def _analyze_single_trace(self, trace_file):
        analyzer = NSysAnalyzer(trace_file)
        analyzer.connect()
        
        # Get NVTX layers
        nvtx_query = """
            SELECT DISTINCT COALESCE(sid.value, ne.text, '') AS layer_name
            FROM NVTX_EVENTS AS ne
            LEFT JOIN StringIds AS sid ON ne.textId = sid.id
            WHERE (ne.eventType = 59 OR ne.eventType = 70)
            AND COALESCE(sid.value, ne.text, '') IN 
                ('dhconv', 'RealSHT', 'InverseSHT', 'act_layer', 'decoder', 
                 'encoder', 'inner_skip', 'mlp', 'norm0', 'norm1', 'outer_skip')
            ORDER BY layer_name;
        """
        
        analyzer.cursor.execute(nvtx_query)
        layer_names = [row[0] for row in analyzer.cursor.fetchall()]
        
        # Track used correlation IDs to prevent double-counting
        used_correlation_ids = set()
        layer_results = {}
        
        for layer_name in layer_names:
            # Get NVTX time ranges for this layer
            timing_query = """
                SELECT ne.start, ne.end FROM NVTX_EVENTS AS ne
                LEFT JOIN StringIds AS sid ON ne.textId = sid.id
                WHERE (ne.eventType = 59 OR ne.eventType = 70)
                AND COALESCE(sid.value, ne.text, '') = ?
                ORDER BY ne.start;
            """
            
            analyzer.cursor.execute(timing_query, (layer_name,))
            time_ranges = analyzer.cursor.fetchall()
            
            if not time_ranges:
                continue
                
            # Skip first occurrence, collect GPU kernel times
            filtered_ranges = time_ranges[1:] if len(time_ranges) > 1 else time_ranges
            total_gpu_time = 0
            
            for start_time, end_time in filtered_ranges:
                # Find API calls within NVTX range
                api_query = """
                    SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
                    WHERE start >= ? AND end <= ? AND correlationId IS NOT NULL
                    AND (name = 'cudaLaunchKernel' OR name = 'cudaLaunchKernel_v7000' OR name = 'cuLaunchKernel')
                """
                
                analyzer.cursor.execute(api_query, (start_time, end_time))
                correlation_ids = [row[0] for row in analyzer.cursor.fetchall()]
                
                # Get GPU kernel times (avoid double counting)
                for corr_id in correlation_ids:
                    if corr_id not in used_correlation_ids:
                        kernel_query = """
                            SELECT (end - start) as kernel_duration
                            FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId = ?
                        """
                        
                        analyzer.cursor.execute(kernel_query, (corr_id,))
                        kernel_result = analyzer.cursor.fetchone()
                        if kernel_result:
                            total_gpu_time += kernel_result[0]
                            used_correlation_ids.add(corr_id)
            
            if total_gpu_time > 0:
                layer_results[layer_name] = total_gpu_time
        
        # Combine norm layers
        if 'norm0' in layer_results and 'norm1' in layer_results:
            layer_results['norm'] = (layer_results['norm0'] + layer_results['norm1']) / 2
            del layer_results['norm0']
            del layer_results['norm1']
        
        analyzer.disconnect()
        return layer_results

    def _create_stacked_runtime_plot(self, results):
        """Create stacked area chart with actual runtime values (not percentages)"""
        
        main_layer_names = ['dhconv', 'RealSHT', 'InverseSHT', 'mlp']
        embed_dims = sorted(results.keys())
        
        # Prepare runtime data for each layer (in milliseconds)
        layer_runtimes = {}
        for layer in main_layer_names + ['other']:
            layer_runtimes[layer] = []
        
        for dim in embed_dims:
            # Get main layer runtimes
            for layer in main_layer_names:
                if layer in results[dim]:
                    runtime_ms = results[dim][layer] / 1e6  # Convert to ms
                    layer_runtimes[layer].append(runtime_ms)
                else:
                    layer_runtimes[layer].append(0)
            
            # Calculate "other" layers combined runtime
            other_total = sum(time for layer_name, time in results[dim].items() 
                            if layer_name not in main_layer_names)
            other_runtime_ms = other_total / 1e6  # Convert to ms
            layer_runtimes['other'].append(other_runtime_ms)
        
        # Create stacked area plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        # Match kernel analysis color scheme
        layer_colors = {
            'dhconv': '#ff7f0e',      # Orange
            'RealSHT': '#1f77b4',     # Blue
            'InverseSHT': '#2ca02c',  # Green  
            'mlp': '#d62728',         # Red
            'other': '#7f7f7f'        # Grey
        }
        
        # Create the stacked areas - build from bottom up
        bottom = np.zeros(len(embed_dims))
        
        all_layers = main_layer_names + ['other']
        for layer in all_layers:
            if any(layer_runtimes[layer]):  # Only plot if there's data
                color = layer_colors.get(layer, layer_colors['other'])
                
                # Clean up display name
                display_name = layer
                if display_name.lower() == 'dhconv':
                    display_name = 'DHConv'
                elif display_name.lower() == 'realsht':
                    display_name = 'RealSHT'
                elif display_name.lower() == 'inversesht':
                    display_name = 'InverseSHT'
                elif display_name.lower() == 'mlp':
                    display_name = 'MLP'
                elif display_name == 'other':
                    display_name = 'Other Layers'
                
                # Stack this layer on top of previous layers
                layer_values = np.array(layer_runtimes[layer])
                ax.fill_between(embed_dims, bottom, bottom + layer_values,
                              color=color, alpha=0.8, label=display_name)
                
                bottom += layer_values
        
        # IEEE formatting
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Runtime (ms)')
        ax.set_xticks(embed_dims)
        ax.grid(True, alpha=0.3)
        
        # Add ACE/ACE2 lines
        ymax = ax.get_ylim()[1]
        ax.axvline(x=256, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=384, color='blue', linestyle='--', alpha=0.7)
        
        # Add labels with background
        ax.text(256, ymax * 0.85, 'ACE', ha='center', va='center', color='red',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red'))
        ax.text(384, ymax * 0.85, 'ACE2', ha='center', va='center', color='blue', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
        
        # Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, 
                 shadow=False, fancybox=False)
        
        # Save plot
        os.makedirs('layer_embedding_analysis/comparative', exist_ok=True)
        output_path = 'layer_embedding_analysis/comparative/comparative_runtime_stacked.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"\\nStacked runtime plot saved: {output_path}")
        
        # Print total runtime at each dimension (top of the stack)
        print(f"\\nTotal runtime at each dimension:")
        for i, dim in enumerate(embed_dims):
            total_runtime = sum(layer_runtimes[layer][i] for layer in all_layers)
            print(f"  {dim}d: {total_runtime:.1f} ms")
        
        # Print breakdown for largest dimension
        max_dim = max(embed_dims)
        max_idx = embed_dims.index(max_dim)
        total_at_max = sum(layer_runtimes[layer][max_idx] for layer in all_layers)
        print(f"\\nRuntime breakdown at {max_dim} dimensions (total: {total_at_max:.1f} ms):")
        for layer in all_layers:
            if layer_runtimes[layer][max_idx] > 0:
                runtime = layer_runtimes[layer][max_idx]
                pct = (runtime / total_at_max) * 100
                display_name = layer
                if display_name.lower() == 'dhconv':
                    display_name = 'DHConv'
                elif display_name.lower() == 'realsht':
                    display_name = 'RealSHT'
                elif display_name.lower() == 'inversesht':
                    display_name = 'InverseSHT'
                elif display_name.lower() == 'mlp':
                    display_name = 'MLP'
                elif display_name == 'other':
                    display_name = 'Other Layers'
                print(f"  {display_name}: {runtime:.1f} ms ({pct:.1f}%)")

    def _create_line_plot(self, results):
        """Create regular line plot for comparison"""
        
        main_layer_names = ['dhconv', 'RealSHT', 'InverseSHT', 'mlp']
        embed_dims = sorted(results.keys())
        
        # Separate main layers from others
        main_layers = {}
        other_layers = {}
        
        for dim in embed_dims:
            main_layers[dim] = {}
            other_layers[dim] = {}
            
            for layer, time in results[dim].items():
                if layer in main_layer_names:
                    main_layers[dim][layer] = time
                else:
                    other_layers[dim][layer] = time
        
        # Calculate "Other Layer Average"
        other_layer_avg = {}
        for dim in embed_dims:
            if other_layers[dim]:
                avg_time = sum(other_layers[dim].values()) / len(other_layers[dim])
                other_layer_avg[dim] = avg_time / 1e6  # Convert to ms
            else:
                other_layer_avg[dim] = 0
        
        # Create plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        layer_colors = {
            'dhconv': '#ff7f0e',      # Orange
            'RealSHT': '#1f77b4',     # Blue
            'InverseSHT': '#2ca02c',  # Green
            'mlp': '#d62728',         # Red
            'other': '#7f7f7f'        # Grey
        }
        
        markers = ['o', 's', '^', 'v', 'D']
        plot_index = 0
        
        # Plot main layers
        for layer in main_layer_names:
            layer_times = []
            valid_dims = []
            
            for dim in embed_dims:
                if layer in main_layers[dim]:
                    runtime_ms = main_layers[dim][layer] / 1e6
                    layer_times.append(runtime_ms)
                    valid_dims.append(dim)
            
            if layer_times:
                color = layer_colors.get(layer, layer_colors['other'])
                marker = markers[plot_index % len(markers)]
                
                display_name = layer
                if display_name.lower() == 'dhconv':
                    display_name = 'DHConv'
                elif display_name.lower() == 'realsht':
                    display_name = 'RealSHT'
                elif display_name.lower() == 'inversesht':
                    display_name = 'InverseSHT'
                elif display_name.lower() == 'mlp':
                    display_name = 'MLP'
                
                ax.plot(valid_dims, layer_times, color=color, marker=marker, 
                       linewidth=2.0, markersize=6, label=display_name)
                plot_index += 1
        
        # Plot "Other Layer Average" 
        if any(other_layer_avg.values()):
            other_times = [other_layer_avg[dim] for dim in embed_dims]
            ax.plot(embed_dims, other_times, color=layer_colors['other'], 
                   marker='x', linestyle='--', linewidth=2.0, markersize=6, 
                   label='Other Layer Average')
        
        # IEEE formatting
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Runtime (ms)')
        ax.set_xticks(embed_dims)
        ax.grid(True, alpha=0.3)
        
        # Add ACE/ACE2 lines
        ymax = ax.get_ylim()[1]
        ax.axvline(x=256, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=384, color='blue', linestyle='--', alpha=0.7)
        
        # Add labels with background
        ax.text(256, ymax * 0.85, 'ACE', ha='center', va='center', color='red',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red'))
        ax.text(384, ymax * 0.85, 'ACE2', ha='center', va='center', color='blue', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
        
        # Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, 
                 shadow=False, fancybox=False)
        
        # Save plot
        output_path = 'layer_embedding_analysis/comparative/comparative_runtime_updated.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Updated line plot saved: {output_path}")

def main():
    if len(os.sys.argv) != 2:
        print("Usage: python3 layer_stacked_analysis.py <trace_pattern>")
        print("Example: python3 layer_stacked_analysis.py 'sfno_{}_20_a100_correlated.sqlite'")
        return
    
    trace_pattern = os.sys.argv[1]
    
    analyzer = LayerStackedAnalyzer(trace_pattern)
    results = analyzer.analyze_and_visualize()
    
    if results:
        print("\\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
    else:
        print("No results found!")

if __name__ == "__main__":
    main()