from nsys_utils import NSysAnalyzer

# Path to the SQLite database
db_path = "../A100/SFNO_NSYS/gigaio80/sfno_1024.sqlite"
# db_path = "../A100/ACE_NSYS/ACE2_400_50.sqlite"
# Initialize the analyzer
analyzer = NSysAnalyzer(db_path)
analyzer.connect()

# Fetch all kernels once for efficient grouping
print("Fetching all kernels for analysis after Memset...")
all_kernels = analyzer.fetch_all_kernels()
print(f"Fetched {len(all_kernels)} total kernels ordered by start time")

# Create a mapping of start times to kernel indices for quick lookup
kernel_by_start = {kernel[2]: (i, kernel) for i, kernel in enumerate(all_kernels)}

# Find NCHW kernels
nchw_kernels = analyzer.fetch_kernels_by_pattern("NCHW")

# Group NCHW kernels with surrounding kernels as "norm layers"
norm_layers = []
for i, nchw_kernel in enumerate(nchw_kernels):
    nchw_start_time = nchw_kernel[2]
    
    # Check if this kernel exists in our filtered kernel list
    if nchw_start_time not in kernel_by_start:
        print(f"Warning: NCHW kernel with start time {nchw_start_time} not found in filtered kernel list. Skipping.")
        continue
    
    nchw_idx = kernel_by_start[nchw_start_time][0]
    
    # Get 1 kernel before (if available)
    kernel_before = None
    if nchw_idx > 0:
        kernel_before = all_kernels[nchw_idx - 1]
    
    # Create norm layer structure
    norm_layer = {
        'norm_layer_id': i + 1,
        'kernel_before': kernel_before,
        'nchw_kernel': nchw_kernel,
        'total_duration': (kernel_before[4] if kernel_before else 0) + nchw_kernel[4],
        'kernels': [k for k in [kernel_before, nchw_kernel] if k is not None]
    }
    norm_layers.append(norm_layer)

analyzer.print_summary("Norm Layer", norm_layers)


# Find fft_r2c kernels (we cant simply do the next 10 kernels after the norm layer because there are 2 norm layers within the SFNO block)
fft_r2c_kernels = analyzer.fetch_kernels_by_pattern("fft_r2c")

# Group fft_r2c kernels with 10 kernels after as "RealSHT_layer"
realsht_layers = []
for i, fft_r2c_kernel in enumerate(fft_r2c_kernels):
    fft_r2c_start_time = fft_r2c_kernel[2]
    
    # Check if this kernel exists in our filtered kernel list
    if fft_r2c_start_time not in kernel_by_start:
        print(f"Warning: fft_r2c kernel with start time {fft_r2c_start_time} not found in filtered kernel list. Skipping.")
        continue
    
    fft_r2c_idx = kernel_by_start[fft_r2c_start_time][0]
    
    # Get 10 kernels after fft_r2c (if available)
    kernels_after = []
    for j in range(fft_r2c_idx + 1, min(fft_r2c_idx + 11, len(all_kernels))):
        kernels_after.append(all_kernels[j])
    
    # Create RealSHT layer structure (excluding fft_r2c kernel itself)
    realsht_layer = {
        'realsht_layer_id': i + 1,
        'fft_r2c_kernel': fft_r2c_kernel,
        'kernels_after': kernels_after,
        'total_duration': sum(k[4] for k in kernels_after),  # Only count kernels after, not fft_r2c itself
        'kernels': kernels_after
    }
    realsht_layers.append(realsht_layer)

analyzer.print_summary("RealSHT Layer", realsht_layers)



# Group the next 3 kernels after each RealSHT layer as "DHconv_layer"
dhconv_layers = []
for i, realsht_layer in enumerate(realsht_layers):
    # Find the last kernel of the RealSHT layer
    if realsht_layer['kernels_after']:
        last_realsht_kernel = realsht_layer['kernels_after'][-1]
        last_realsht_start_time = last_realsht_kernel[2]
        
        # Check if this kernel exists in our filtered kernel list
        if last_realsht_start_time not in kernel_by_start:
            print(f"Warning: Last RealSHT kernel with start time {last_realsht_start_time} not found in filtered kernel list. Skipping DHconv layer.")
            continue
        
        last_realsht_idx = kernel_by_start[last_realsht_start_time][0]
        
        # Get 3 kernels after the RealSHT layer (if available)
        kernels_after_realsht = []
        for j in range(last_realsht_idx + 1, min(last_realsht_idx + 4, len(all_kernels))):
            kernels_after_realsht.append(all_kernels[j])
        
        # Create DHconv layer structure
        dhconv_layer = {
            'dhconv_layer_id': i + 1,
            'realsht_layer_id': realsht_layer['realsht_layer_id'],
            'kernels_after_realsht': kernels_after_realsht,
            'total_duration': sum(k[4] for k in kernels_after_realsht),
            'kernels': kernels_after_realsht
        }
        dhconv_layers.append(dhconv_layer)

analyzer.print_summary("DHconv Layer", dhconv_layers)



# Group the next 8 kernels after each DHconv layer as "InverseSHT"
inversesht_layers = []
for i, dhconv_layer in enumerate(dhconv_layers):
    # Find the last kernel of the DHconv layer
    if dhconv_layer['kernels_after_realsht']:
        last_dhconv_kernel = dhconv_layer['kernels_after_realsht'][-1]
        last_dhconv_start_time = last_dhconv_kernel[2]
        
        # Check if this kernel exists in our filtered kernel list
        if last_dhconv_start_time not in kernel_by_start:
            print(f"Warning: Last DHconv kernel with start time {last_dhconv_start_time} not found in filtered kernel list. Skipping InverseSHT layer.")
            continue
        
        last_dhconv_idx = kernel_by_start[last_dhconv_start_time][0]
        
        # Get 8 kernels after the DHconv layer (if available)
        kernels_after_dhconv = []
        for j in range(last_dhconv_idx + 1, min(last_dhconv_idx + 9, len(all_kernels))):
            kernels_after_dhconv.append(all_kernels[j])
        
        # Create InverseSHT layer structure
        inversesht_layer = {
            'inversesht_layer_id': i + 1,
            'dhconv_layer_id': dhconv_layer['dhconv_layer_id'],
            'realsht_layer_id': dhconv_layer['realsht_layer_id'],
            'kernels_after_dhconv': kernels_after_dhconv,
            'total_duration': sum(k[4] for k in kernels_after_dhconv),
            'kernels': kernels_after_dhconv
        }
        inversesht_layers.append(inversesht_layer)

analyzer.print_summary("InverseSHT Layer", inversesht_layers)


# Group the next 3 kernels after each InverseSHT layer as "Inner_Skip_Layers"
inner_skip_layers = []
for i, inversesht_layer in enumerate(inversesht_layers):
    # Find the last kernel of the InverseSHT layer
    if inversesht_layer['kernels_after_dhconv']:
        last_inversesht_kernel = inversesht_layer['kernels_after_dhconv'][-1]
        last_inversesht_start_time = last_inversesht_kernel[2]
        
        # Check if this kernel exists in our filtered kernel list
        if last_inversesht_start_time not in kernel_by_start:
            print(f"Warning: Last InverseSHT kernel with start time {last_inversesht_start_time} not found in filtered kernel list. Skipping Inner Skip layer.")
            continue
        
        last_inversesht_idx = kernel_by_start[last_inversesht_start_time][0]
        
        # Get 3 kernels after the InverseSHT layer (if available)
        kernels_after_inversesht = []
        for j in range(last_inversesht_idx + 1, min(last_inversesht_idx + 4, len(all_kernels))):
            kernels_after_inversesht.append(all_kernels[j])
        
        # Create Inner Skip layer structure
        inner_skip_layer = {
            'inner_skip_layer_id': i + 1,
            'inversesht_layer_id': inversesht_layer['inversesht_layer_id'],
            'dhconv_layer_id': inversesht_layer['dhconv_layer_id'],
            'realsht_layer_id': inversesht_layer['realsht_layer_id'],
            'kernels_after_inversesht': kernels_after_inversesht,
            'total_duration': sum(k[4] for k in kernels_after_inversesht),
            'kernels': kernels_after_inversesht
        }
        inner_skip_layers.append(inner_skip_layer)

analyzer.print_summary("Inner Skip Layer", inner_skip_layers)


# Group the next kernel after each Inner Skip layer as "Activation_Layers"
activation_layers = []
for i, inner_skip_layer in enumerate(inner_skip_layers):
    # Find the last kernel of the Inner Skip layer
    if inner_skip_layer['kernels_after_inversesht']:
        last_inner_skip_kernel = inner_skip_layer['kernels_after_inversesht'][-1]
        last_inner_skip_start_time = last_inner_skip_kernel[2]
        
        # Check if this kernel exists in our filtered kernel list
        if last_inner_skip_start_time not in kernel_by_start:
            print(f"Warning: Last Inner Skip kernel with start time {last_inner_skip_start_time} not found in filtered kernel list. Skipping Activation layer.")
            continue
        
        last_inner_skip_idx = kernel_by_start[last_inner_skip_start_time][0]
        
        # Get 1 kernel after the Inner Skip layer (if available)
        kernel_after_inner_skip = None
        if last_inner_skip_idx + 1 < len(all_kernels):
            kernel_after_inner_skip = all_kernels[last_inner_skip_idx + 1]
        
        # Create Activation layer structure
        activation_layer = {
            'activation_layer_id': i + 1,
            'inner_skip_layer_id': inner_skip_layer['inner_skip_layer_id'],
            'inversesht_layer_id': inner_skip_layer['inversesht_layer_id'],
            'dhconv_layer_id': inner_skip_layer['dhconv_layer_id'],
            'realsht_layer_id': inner_skip_layer['realsht_layer_id'],
            'kernel_after_inner_skip': kernel_after_inner_skip,
            'total_duration': kernel_after_inner_skip[4] if kernel_after_inner_skip else 0,
            'kernels': [kernel_after_inner_skip] if kernel_after_inner_skip else []
        }
        activation_layers.append(activation_layer)

analyzer.print_summary("Activation Layer", activation_layers)


# Skip 2 kernels after each Activation layer, then group the next 5 kernels as "MLP_layer"
mlp_layers = []
for i, activation_layer in enumerate(activation_layers):
    # Find the last kernel of the Activation layer
    if activation_layer['kernel_after_inner_skip']:
        last_activation_kernel = activation_layer['kernel_after_inner_skip']
        last_activation_start_time = last_activation_kernel[2]
        
        # Check if this kernel exists in our filtered kernel list
        if last_activation_start_time not in kernel_by_start:
            print(f"Warning: Last Activation kernel with start time {last_activation_start_time} not found in filtered kernel list. Skipping MLP layer.")
            continue
        
        last_activation_idx = kernel_by_start[last_activation_start_time][0]
        
        # Skip 2 kernels after the Activation layer, then get the next 5 kernels
        skip_start_idx = last_activation_idx + 1
        skip_end_idx = skip_start_idx + 2  # Skip 2 kernels
        mlp_start_idx = skip_end_idx
        mlp_end_idx = min(mlp_start_idx + 5, len(all_kernels))  # Get next 5 kernels
        
        # Get 5 kernels after skipping 2 (if available)
        kernels_for_mlp = []
        for j in range(mlp_start_idx, mlp_end_idx):
            kernels_for_mlp.append(all_kernels[j])
        
        # Create MLP layer structure
        mlp_layer = {
            'mlp_layer_id': i + 1,
            'activation_layer_id': activation_layer['activation_layer_id'],
            'inner_skip_layer_id': activation_layer['inner_skip_layer_id'],
            'inversesht_layer_id': activation_layer['inversesht_layer_id'],
            'dhconv_layer_id': activation_layer['dhconv_layer_id'],
            'realsht_layer_id': activation_layer['realsht_layer_id'],
            'kernels_for_mlp': kernels_for_mlp,
            'total_duration': sum(k[4] for k in kernels_for_mlp),
            'kernels': kernels_for_mlp
        }
        mlp_layers.append(mlp_layer)

analyzer.print_summary("MLP Layer", mlp_layers)


# Group the next kernel after each MLP layer as "Outer_skip_layer"
outer_skip_layers = []
for i, mlp_layer in enumerate(mlp_layers):
    # Find the last kernel of the MLP layer
    if mlp_layer['kernels_for_mlp']:
        last_mlp_kernel = mlp_layer['kernels_for_mlp'][-1]
        last_mlp_start_time = last_mlp_kernel[2]
        
        # Check if this kernel exists in our filtered kernel list
        if last_mlp_start_time not in kernel_by_start:
            print(f"Warning: Last MLP kernel with start time {last_mlp_start_time} not found in filtered kernel list. Skipping Outer Skip layer.")
            continue
        
        last_mlp_idx = kernel_by_start[last_mlp_start_time][0]
        
        # Get 1 kernel after the MLP layer (if available)
        kernel_after_mlp = None
        if last_mlp_idx + 1 < len(all_kernels):
            kernel_after_mlp = all_kernels[last_mlp_idx + 1]
        
        # Create Outer Skip layer structure
        outer_skip_layer = {
            'outer_skip_layer_id': i + 1,
            'mlp_layer_id': mlp_layer['mlp_layer_id'],
            'activation_layer_id': mlp_layer['activation_layer_id'],
            'inner_skip_layer_id': mlp_layer['inner_skip_layer_id'],
            'inversesht_layer_id': mlp_layer['inversesht_layer_id'],
            'dhconv_layer_id': mlp_layer['dhconv_layer_id'],
            'realsht_layer_id': mlp_layer['realsht_layer_id'],
            'kernel_after_mlp': kernel_after_mlp,
            'total_duration': kernel_after_mlp[4] if kernel_after_mlp else 0,
            'kernels': [kernel_after_mlp] if kernel_after_mlp else []
        }
        outer_skip_layers.append(outer_skip_layer)

analyzer.print_summary("Outer Skip Layer", outer_skip_layers)

# Create runtime visualization for all layer types
print("\n===== RUNTIME VISUALIZATION FOR LAYERS =====")
layer_runtime_data = {}
if norm_layers:
    layer_runtime_data["Norm Layer"] = sum(layer['total_duration'] for layer in norm_layers)
if realsht_layers:
    layer_runtime_data["RealSHT Layer"] = sum(layer['total_duration'] for layer in realsht_layers)
if dhconv_layers:
    layer_runtime_data["DHconv Layer"] = sum(layer['total_duration'] for layer in dhconv_layers)
if inversesht_layers:
    layer_runtime_data["InverseSHT Layer"] = sum(layer['total_duration'] for layer in inversesht_layers)
if inner_skip_layers:
    layer_runtime_data["Inner Skip Layer"] = sum(layer['total_duration'] for layer in inner_skip_layers)
if activation_layers:
    layer_runtime_data["Activation Layer"] = sum(layer['total_duration'] for layer in activation_layers)
if mlp_layers:
    layer_runtime_data["MLP Layer"] = sum(layer['total_duration'] for layer in mlp_layers)
if outer_skip_layers:
    layer_runtime_data["Outer Skip Layer"] = sum(layer['total_duration'] for layer in outer_skip_layers)

analyzer.create_runtime_visualizations(layer_runtime_data, prefix="layer", title_prefix="Layer ")

#--------------------------------Generate Plots--------------------------------#
# Now analyze GPU metrics for all layers
metrics_by_layer_type = {}
if analyzer.table_exists("GPU_METRICS") and analyzer.table_exists("TARGET_INFO_GPU_METRICS"):
    print("\n===== GPU METRICS ANALYSIS FOR LAYERS =====")
    
    # Collect all layer types for analysis
    all_layer_types = [
        ("Norm Layer", norm_layers),
        ("RealSHT Layer", realsht_layers),
        ("DHconv Layer", dhconv_layers),
        ("InverseSHT Layer", inversesht_layers),
        ("Inner Skip Layer", inner_skip_layers),
        ("Activation Layer", activation_layers),
        ("MLP Layer", mlp_layers),
        ("Outer Skip Layer", outer_skip_layers)
    ]
    
    # Get metrics for each layer type
    for layer_type_name, layers in all_layer_types:
        if not layers:
            continue
            
        print(f"\nProcessing {layer_type_name}...")
        
        # Collect all kernels from all layers of this type
        all_kernels_for_type = []
        for layer in layers:
            if 'kernels' in layer and layer['kernels']:
                all_kernels_for_type.extend(layer['kernels'])
        
        if not all_kernels_for_type:
            print(f"  No kernels found for {layer_type_name}")
            continue
        
        print(f"  Processing {len(all_kernels_for_type)} total kernels for {layer_type_name}")
        
        # Get metrics for this layer type
        layer_type_metrics = analyzer.get_metrics_by_layer(all_kernels_for_type)
        metrics_by_layer_type[layer_type_name] = layer_type_metrics
        print(f"  Processed {len(layer_type_metrics)} metrics for {layer_type_name}")
    
    # Create visualizations and print summary
    if metrics_by_layer_type:
        analyzer.create_visualizations(metrics_by_layer_type, prefix="layer")
        analyzer.print_metrics_summary(metrics_by_layer_type, "METRICS SUMMARY BY LAYER TYPE")

# Close the database connection
analyzer.disconnect()

# ---- Export summary to unified JSON for later comparison ----
import os, json
json_path = "all_traces_summary.json"
db_base = os.path.splitext(os.path.basename(db_path))[0]
# Prepare the layer summary dict
layer_summary = {
    'layer_runtime_data': layer_runtime_data,
    'metrics_by_layer_type': {k: {str(m): float(v) for m, v in metrics.items()} for k, metrics in metrics_by_layer_type.items()}
}
# Load or create the unified JSON
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        all_data = json.load(f)
else:
    all_data = {}
if db_base not in all_data:
    all_data[db_base] = {}
all_data[db_base]['layer'] = layer_summary
with open(json_path, 'w') as f:
    json.dump(all_data, f, indent=2)
print(f"Saved layer summary for {db_base} to {json_path}")
