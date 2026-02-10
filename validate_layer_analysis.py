#!/usr/bin/env python3

import os
from nsys_utils import NSysAnalyzer

def validate_layer_analysis():
    """
    Compare NVTX-based layer analysis with kernel-names approach for sanity check
    """
    
    trace_file = "sfno_1024_20_a100_correlated.sqlite"
    
    print("=" * 80)
    print("LAYER ANALYSIS VALIDATION")
    print("=" * 80)
    print(f"Testing with: {trace_file}")
    
    analyzer = NSysAnalyzer(trace_file)
    analyzer.connect()
    
    # 1. NVTX-based approach (our main method)
    print("\\n--- Method 1: NVTX-based Layer Attribution ---")
    nvtx_results = analyze_nvtx_layers(analyzer)
    
    # 2. Kernel names approach (sanity check)
    print("\\n--- Method 2: Kernel Names Attribution ---")
    kernel_results = analyze_kernel_names(analyzer)
    
    # 3. Compare results
    print("\\n--- COMPARISON ---")
    compare_results(nvtx_results, kernel_results)
    
    analyzer.disconnect()

def analyze_nvtx_layers(analyzer):
    """NVTX-based approach (same as our fixed script)"""
    
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
    
    used_correlation_ids = set()
    layer_results = {}
    
    for layer_name in layer_names:
        # Get NVTX time ranges
        timing_query = """
            SELECT ne.start, ne.end FROM NVTX_EVENTS AS ne
            LEFT JOIN StringIds AS sid ON ne.textId = sid.id
            WHERE (ne.eventType = 59 OR ne.eventType = 70)
            AND COALESCE(sid.value, ne.text, '') = ?
            ORDER BY ne.start;
        """
        
        analyzer.cursor.execute(timing_query, (layer_name,))
        time_ranges = analyzer.cursor.fetchall()
        
        # Skip first occurrence, collect GPU kernel times
        filtered_ranges = time_ranges[1:] if len(time_ranges) > 1 else time_ranges
        total_gpu_time = 0
        total_kernel_count = 0
        
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
                        total_kernel_count += 1
                        used_correlation_ids.add(corr_id)
        
        if total_gpu_time > 0:
            layer_results[layer_name] = total_gpu_time / 1e6  # Convert to ms
            print(f"  {layer_name}: {total_gpu_time/1e6:.2f}ms ({total_kernel_count} kernels)")
    
    # Combine norm layers
    if 'norm0' in layer_results and 'norm1' in layer_results:
        layer_results['norm'] = (layer_results['norm0'] + layer_results['norm1']) / 2
        del layer_results['norm0']
        del layer_results['norm1']
    
    total_nvtx_time = sum(layer_results.values())
    print(f"NVTX Total: {total_nvtx_time:.2f}ms using {len(used_correlation_ids)} correlation IDs")
    
    return layer_results

def analyze_kernel_names(analyzer):
    """Kernel names approach for comparison"""
    
    # Get all kernels
    kernels = analyzer.fetch_all_kernels()
    
    layer_kernels = {}
    
    for kernel in kernels:
        kernel_name = kernel[0].lower()  # kernel name at index 0
        kernel_duration = kernel[4] / 1e6  # duration in ms
        
        # Map kernel names to layers
        layer_type = 'other'
        
        # DHConv - GEMM/Conv operations
        if any(x in kernel_name for x in ['cgemm', 'gemm', 'sgemm', 'hgemm']):
            layer_type = 'dhconv'
        # SHT operations
        elif any(x in kernel_name for x in ['fft', 'sht']):
            # Try to distinguish Real vs Inverse SHT
            if 'inverse' in kernel_name or 'ifft' in kernel_name:
                layer_type = 'InverseSHT'
            else:
                layer_type = 'RealSHT'
        # Normalization
        elif any(x in kernel_name for x in ['norm', 'batch_norm', 'layer_norm']):
            layer_type = 'norm'
        # Activation layers
        elif any(x in kernel_name for x in ['relu', 'gelu', 'act', 'sigmoid', 'tanh']):
            layer_type = 'act_layer'
        # Element-wise operations (likely skip connections)
        elif any(x in kernel_name for x in ['elementwise', 'add', 'sum']):
            layer_type = 'inner_skip'  
        # MLP operations
        elif any(x in kernel_name for x in ['linear', 'dense', 'matmul', 'mm']):
            layer_type = 'mlp'
        # Memory/copy operations
        elif any(x in kernel_name for x in ['copy', 'memcpy', 'memset']):
            layer_type = 'other'
        
        if layer_type not in layer_kernels:
            layer_kernels[layer_type] = 0
        layer_kernels[layer_type] += kernel_duration
    
    # Print results
    total_kernel_time = sum(layer_kernels.values())
    for layer, duration in sorted(layer_kernels.items(), key=lambda x: x[1], reverse=True):
        if duration > 10:  # Only show significant layers
            print(f"  {layer}: {duration:.2f}ms")
    
    print(f"Kernel Names Total: {total_kernel_time:.2f}ms")
    
    return layer_kernels

def compare_results(nvtx_results, kernel_results):
    """Compare the two approaches"""
    
    print("\\nDirect Comparison (NVTX vs Kernel Names):")
    print("-" * 50)
    
    # Get common layers
    nvtx_layers = set(nvtx_results.keys())
    kernel_layers = set(kernel_results.keys())
    common_layers = nvtx_layers.intersection(kernel_layers)
    
    total_nvtx = sum(nvtx_results.values())
    total_kernel = sum(kernel_results.values())
    
    print(f"{'Layer':<15} {'NVTX (ms)':<12} {'Kernel (ms)':<12} {'Ratio':<8} {'Diff %':<8}")
    print("-" * 60)
    
    for layer in sorted(common_layers):
        nvtx_time = nvtx_results[layer]
        kernel_time = kernel_results.get(layer, 0)
        
        if kernel_time > 0:
            ratio = nvtx_time / kernel_time
            diff_pct = ((nvtx_time - kernel_time) / kernel_time) * 100
        else:
            ratio = float('inf')
            diff_pct = float('inf')
        
        print(f"{layer:<15} {nvtx_time:<12.2f} {kernel_time:<12.2f} {ratio:<8.2f} {diff_pct:<8.1f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_nvtx:<12.2f} {total_kernel:<12.2f} {total_nvtx/total_kernel:<8.2f} {((total_nvtx-total_kernel)/total_kernel)*100:<8.1f}%")
    
    # Analysis
    print(f"\\nAnalysis:")
    print(f"- NVTX approach accounts for {total_nvtx:.0f}ms of GPU time")
    print(f"- Kernel names approach accounts for {total_kernel:.0f}ms of GPU time") 
    
    if abs(total_nvtx - total_kernel) / total_kernel < 0.1:
        print("✅ Results are consistent (within 10%)")
    else:
        print("⚠️  Significant difference - needs investigation")
    
    # Check which approach shows DHConv as dominant
    nvtx_dhconv = nvtx_results.get('dhconv', 0)
    kernel_dhconv = kernel_results.get('dhconv', 0)
    
    print(f"\\nDHConv Analysis:")
    print(f"- NVTX DHConv: {nvtx_dhconv:.2f}ms")
    print(f"- Kernel Names DHConv: {kernel_dhconv:.2f}ms")
    
    nvtx_max = max(nvtx_results.values()) if nvtx_results else 0
    kernel_max = max(kernel_results.values()) if kernel_results else 0
    
    nvtx_dominant = max(nvtx_results, key=nvtx_results.get) if nvtx_results else None
    kernel_dominant = max(kernel_results, key=kernel_results.get) if kernel_results else None
    
    print(f"- NVTX dominant layer: {nvtx_dominant} ({nvtx_max:.2f}ms)")
    print(f"- Kernel Names dominant layer: {kernel_dominant} ({kernel_max:.2f}ms)")
    
    if nvtx_dominant == kernel_dominant:
        print("✅ Both approaches agree on dominant layer")
    else:
        print("⚠️  Different dominant layers identified")

if __name__ == "__main__":
    validate_layer_analysis()