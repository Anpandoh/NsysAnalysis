### Structure:
- nsys_trace_layer_analysis.py: Converts sqlite to graphs seperated by their NVTX ranges/markers (Only the leaf marks (hardcoded to omit non leaf))
- nsys_trace_kernel_analysis.py: Converts sqlite to graphs by individual kernel (I havent checked whether it is ok for FCN3 but should work correctly, I think some of the kernel names might in the long form however)
- nsys_utils.py: Has some general utils for breaking down the the sqlite. IT ALSO CONTAINS THE CODE TO CREATE THE INDIVIDUAL GRAPHS FOR KERNEL AND LAYER
- scaling_analysis.py: RESPONSIBLE FOR ALL THE SCALING GRAPHS


### How to use:
1. Convert Nsys traces from Gdrive to sqlite (can be done through cli or gui export)
2. Run nsys_trace_layer_analysis and/or nsys_trace_layer_analysis to create visualizations for each trace, this will also save them to a file named all_traces_summary.json
3. Run scaling_analysis which takes in the all_traces_summary.json and generates the the scaling analysis graphs


### Where to find code to modify graphs:
- Individual graphs: in nsys_utils under create visualizations and create runtime visualizations
- Scaled graphs: in scaling analysis line 64 for kernels graphs, line 115 for layer graphs, line 178 for runtime graphs

