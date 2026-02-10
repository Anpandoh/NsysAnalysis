from nsys_utils import NSysAnalyzer
import sqlite3
import os
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np
# Use absolute path to avoid working directory issues
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "../A100/ACE_NSYS/ace2_nvtx_400_50.sqlite")


def correlate_cuda_kernels_with_api(original_db_path: str, output_db_path: str = None):
    """
    Correlate CUDA kernel launches with CUDA API kernel launches by creating a new database
    with additional columns and correlated data.
    
    Args:
        original_db_path: Path to the original NSys SQLite database
        output_db_path: Path for the new correlated database (optional)
    """
    if output_db_path is None:
        # Create output path based on original path
        base_name = os.path.splitext(os.path.basename(original_db_path))[0]
        output_db_path = f"{base_name}_correlated.sqlite"
    
    # Check if correlated database already exists
    if os.path.exists(output_db_path):
        print(f"Correlated database already exists: {output_db_path}")
        print("Skipping correlation step and using existing database.")
        return output_db_path
    
    print(f"Creating correlated database: {output_db_path}")
    
    # Copy the original database to preserve it
    shutil.copy2(original_db_path, output_db_path)
    
    # Connect to the new database
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    
    try:
        # First, check what tables are available
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"Available tables: {tables}")
        
        # Check if database is empty
        if not tables:
            print("Warning: Database is completely empty (no tables found).")
            print("This may be a corrupted or invalid NSys trace file.")
            print("Skipping correlation step and using original database.")
            conn.close()
            # Remove the empty correlated database
            os.remove(output_db_path)
            return original_db_path
        
        # Check if CUPTI_ACTIVITY_KIND_RUNTIME table exists
        if 'CUPTI_ACTIVITY_KIND_RUNTIME' not in tables:
            print("Warning: CUPTI_ACTIVITY_KIND_RUNTIME table not found in database.")
            print("This trace may not have CUDA API runtime data.")
            print("Skipping correlation step and using original database.")
            conn.close()
            # Remove the empty correlated database since no correlation was performed
            os.remove(output_db_path)
            return original_db_path
        
        # Add new columns to CUPTI_ACTIVITY_KIND_RUNTIME table
        print("Adding correlation columns...")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN name TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN kernelName TEXT;")
        
        # Update kernelName by correlating with CUPTI_ACTIVITY_KIND_KERNEL
        print("Correlating kernel names...")
        cursor.execute("""
            UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET kernelName =
                (SELECT value FROM StringIds
                JOIN CUPTI_ACTIVITY_KIND_KERNEL AS cuda_gpu
                    ON cuda_gpu.shortName = StringIds.id
                    AND CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = cuda_gpu.correlationId);
        """)
        
        # Update name from StringIds
        print("Updating API names...")
        cursor.execute("""
            UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET name =
                (SELECT value FROM StringIds WHERE nameId = StringIds.id);
        """)
        
        # Commit the changes
        conn.commit()
        
        # Query the results to show correlation
        print("\n===== CUDA API to Kernel Correlation Results =====")
        
        # Get total count of correlated entries
        cursor.execute("""
            SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME 
            WHERE kernelName IS NOT NULL
        """)
        correlated_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        total_count = cursor.fetchone()[0]
        
        print(f"Total CUDA API calls: {total_count}")
        print(f"Correlated with kernel launches: {correlated_count}")
        print(f"Correlation rate: {(correlated_count/total_count)*100:.1f}%")
        
        # Show top 10 longest CUDA API ranges that resulted in kernel execution
        print("\nTop 10 longest CUDA API ranges that resulted in kernel execution:")
        cursor.execute("""
            SELECT name, kernelName, start, end, (end - start) as duration 
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE kernelName IS NOT NULL 
            ORDER BY duration DESC 
            LIMIT 10;
        """)
        
        results = cursor.fetchall()
        for i, (api_name, kernel_name, start, end, duration) in enumerate(results, 1):
            print(f"{i:2d}. API: {api_name or 'Unknown'}")
            print(f"    Kernel: {kernel_name or 'Unknown'}")
            print(f"    Duration: {duration/1e6:.2f} ms")
            print(f"    Time: {start} - {end}")
            print()
        
        # Show some statistics about the correlation
        print("===== Correlation Statistics =====")
        
        # Most common API calls that launch kernels
        cursor.execute("""
            SELECT name, COUNT(*) as count, AVG(end - start) as avg_duration
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE kernelName IS NOT NULL AND name IS NOT NULL
            GROUP BY name
            ORDER BY count DESC
            LIMIT 10;
        """)
        
        print("\nMost common CUDA API calls that launch kernels:")
        api_stats = cursor.fetchall()
        for api_name, count, avg_duration in api_stats:
            print(f"  {api_name}: {count} calls, avg {avg_duration/1e6:.2f} ms")
        
        # Most common kernel types launched
        cursor.execute("""
            SELECT kernelName, COUNT(*) as count, AVG(end - start) as avg_duration
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE kernelName IS NOT NULL
            GROUP BY kernelName
            ORDER BY count DESC
            LIMIT 10;
        """)
        
        print("\nMost common kernel types launched:")
        kernel_stats = cursor.fetchall()
        for kernel_name, count, avg_duration in kernel_stats:
            print(f"  {kernel_name}: {count} launches, avg {avg_duration/1e6:.2f} ms")
        
        print(f"\nCorrelated database saved to: {output_db_path}")
        return output_db_path
        
    except Exception as e:
        print(f"Error during correlation: {e}")
        conn.rollback()
        # Clean up the failed correlated database
        try:
            os.remove(output_db_path)
        except:
            pass
        raise
    finally:
        conn.close()


def analyze_nvtx_ranges_with_cuda_timing(correlated_db_path: str):
    """
    Analyze NVTX ranges and their associated CUDA API calls and kernel timing.
    
    Args:
        correlated_db_path: Path to the correlated database
    """
    print(f"\n===== Analyzing NVTX Ranges with CUDA Timing =====")
    
    # Connect to the correlated database
    conn = sqlite3.connect(correlated_db_path)
    cursor = conn.cursor()
    
    try:
        # Check if NVTX_EVENTS table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS';")
        if not cursor.fetchone():
            print("Warning: NVTX_EVENTS table not found in database.")
            print("This trace may not have NVTX data.")
            print("Skipping NVTX analysis.")
            return
        
        # First, get all unique NVTX ranges
        print("Finding NVTX ranges...")
        cursor.execute("""
            WITH domains AS (
                SELECT
                    ne.domainId AS id,
                    ne.globalTid AS globalTid,
                    COALESCE(sid.value, ne.text) AS name
                FROM
                    NVTX_EVENTS AS ne
                LEFT JOIN
                    StringIds AS sid ON ne.textId = sid.id
                WHERE
                    ne.eventType = 75  -- NVTX_DOMAIN_CREATE
                GROUP BY ne.domainId, ne.globalTid
            )
            SELECT DISTINCT
                COALESCE(d.name, '') || ':' || COALESCE(sid.value, ne.text, '') AS fullname
            FROM
                NVTX_EVENTS AS ne
            LEFT JOIN
                domains AS d
                ON ne.domainId = d.id
                AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
            LEFT JOIN
                StringIds AS sid ON ne.textId = sid.id
            WHERE
                ne.eventType = 59  -- NVTX_PUSHPOP_RANGE
                OR ne.eventType = 70  -- NVTXT_PUSHPOP_RANGE
            ORDER BY fullname;
        """)
        
        nvtx_ranges = cursor.fetchall()
        
        if not nvtx_ranges:
            print("No NVTX ranges found in database.")
            return
        
        # Filter out "inference", "warmup", "fcn3_profiling", and "Global convolution" ranges
        filtered_ranges = [range_info for range_info in nvtx_ranges 
                          if "inference" not in range_info[0].lower() 
                          and not range_info[0].lower().startswith(":warmup")
                          and range_info[0] != ":fcn3_profiling"
                          and range_info[0] != ":Global convolution"
                          and range_info[0] != ":forward_pass"]
        
        print(f"\nFound {len(filtered_ranges)} unique NVTX ranges (excluding 'inference', ':warmup*', ':fcn3_profiling', ':Global convolution', and ':forward_pass'):")
        for range_name in filtered_ranges:
            print(f"- {range_name[0]}")
        
        # Store data for visualization
        nvtx_data = {}
        
        # Now analyze each NVTX range with aggregated statistics
        print(f"\n===== Aggregated Analysis by NVTX Range =====")
        
        for range_info in filtered_ranges:
            range_name = range_info[0]
            print(f"\n--- NVTX Range: {range_name} ---")
            
            # Get all occurrences of this NVTX range
            cursor.execute("""
                WITH domains AS (
                    SELECT
                        ne.domainId AS id,
                        ne.globalTid AS globalTid,
                        COALESCE(sid.value, ne.text) AS name
                    FROM
                        NVTX_EVENTS AS ne
                    LEFT JOIN
                        StringIds AS sid ON ne.textId = sid.id
                    WHERE
                        ne.eventType = 75  -- NVTX_DOMAIN_CREATE
                    GROUP BY ne.domainId, ne.globalTid
                )
                SELECT 
                    ne.start,
                    ne.end,
                    ne.end - ne.start as duration
                FROM
                    NVTX_EVENTS AS ne
                LEFT JOIN
                    domains AS d
                    ON ne.domainId = d.id
                    AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
                LEFT JOIN
                    StringIds AS sid ON ne.textId = sid.id
                WHERE
                    (ne.eventType = 59 OR ne.eventType = 70)  -- NVTX_PUSHPOP_RANGE
                    AND (COALESCE(d.name, '') || ':' || COALESCE(sid.value, ne.text, '')) = ?
                ORDER BY ne.start;
            """, (range_name,))
            
            range_times = cursor.fetchall()
            
            if not range_times:
                print("  No timing information found for this range")
                continue
            
            # Aggregate statistics across all occurrences (excluding first occurrence)
            range_times_filtered = range_times[1:]  # Skip first occurrence (startup outlier)
            total_occurrences = len(range_times_filtered)
            if total_occurrences == 0:
                print("  No occurrences remaining after excluding first occurrence")
                continue
            total_nvtx_time = sum(duration for _, _, duration in range_times_filtered)
            avg_nvtx_time = total_nvtx_time / total_occurrences
            
            print(f"  Occurrences: {total_occurrences}")
            print(f"  Average Duration: {avg_nvtx_time/1e6:.2f} ms")
            print(f"  Total Duration: {total_nvtx_time/1e6:.2f} ms")
            
            # Check if CUPTI_ACTIVITY_KIND_RUNTIME table exists for correlation
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_RUNTIME';")
            if not cursor.fetchone():
                print("  No CUDA API runtime data available for correlation")
                continue
            
            # Aggregate all CUDA API calls across all occurrences (excluding first occurrence)
            all_cuda_calls = []
            for start_time, end_time, _ in range_times[1:]:  # Skip first occurrence (startup outlier)
                cursor.execute("""
                    SELECT 
                        name,
                        kernelName,
                        start,
                        end,
                        (end - start) as api_duration,
                        correlationId
                    FROM CUPTI_ACTIVITY_KIND_RUNTIME
                    WHERE start >= ? AND end <= ?
                    ORDER BY start;
                """, (start_time, end_time))
                
                cuda_calls = cursor.fetchall()
                all_cuda_calls.extend(cuda_calls)
            
            if not all_cuda_calls:
                print("  No CUDA API calls found in this range")
                continue
            
            # Get actual kernel execution times using correlation IDs from CUDA API calls
            kernel_execution_times = {}
            correlation_ids_used = set()
            
            for api_name, kernel_name, api_start, api_end, api_duration, corr_id in all_cuda_calls:
                if corr_id is not None:
                    # Get the actual kernel execution time using the correlation ID
                    cursor.execute("""
                        SELECT 
                            k.start,
                            k.end,
                            (k.end - k.start) as kernel_duration,
                            COALESCE(sid.value, 'Unknown') as kernel_name
                        FROM CUPTI_ACTIVITY_KIND_KERNEL k
                        LEFT JOIN StringIds sid ON k.shortName = sid.id
                        WHERE k.correlationId = ?
                    """, (corr_id,))
                    
                    kernel_execution = cursor.fetchone()
                    if kernel_execution:
                        k_start, k_end, k_duration, k_name = kernel_execution
                        correlation_ids_used.add(corr_id)
                        
                        if k_name not in kernel_execution_times:
                            kernel_execution_times[k_name] = {'count': 0, 'total_time': 0}
                        kernel_execution_times[k_name]['count'] += 1
                        kernel_execution_times[k_name]['total_time'] += k_duration
            
            # Aggregate API and kernel statistics
            api_stats = {}
            total_api_time = 0
            
            for api_name, kernel_name, api_start, api_end, api_duration, corr_id in all_cuda_calls:
                # API statistics
                if api_name not in api_stats:
                    api_stats[api_name] = {'count': 0, 'total_time': 0}
                api_stats[api_name]['count'] += 1
                api_stats[api_name]['total_time'] += api_duration
                total_api_time += api_duration
            
            # Calculate total kernel execution time
            total_kernel_time = sum(stats['total_time'] for stats in kernel_execution_times.values())
            
            # Show aggregated API call summary
            print(f"  API Call Summary:")
            for api_name, stats in sorted(api_stats.items(), key=lambda x: x[1]['total_time'], reverse=True):
                avg_time = stats['total_time'] / stats['count']
                print(f"    {api_name}: {stats['count']} calls, {stats['total_time']/1e6:.2f} ms total, {avg_time/1e6:.2f} ms avg")
            
            # Show aggregated kernel execution summary (using actual kernel times from correlation)
            if kernel_execution_times:
                print(f"  Kernel Execution Summary (from correlation IDs):")
                for kernel_name, stats in sorted(kernel_execution_times.items(), key=lambda x: x[1]['total_time'], reverse=True):
                    avg_time = stats['total_time'] / stats['count']
                    print(f"    {kernel_name}: {stats['count']} executions, {stats['total_time']/1e6:.2f} ms total, {avg_time/1e6:.2f} ms avg")
            
            print(f"  Correlation IDs used: {len(correlation_ids_used)}")
            
            # Show aggregated timing breakdown
            avg_api_time = total_api_time / total_occurrences
            avg_kernel_time = total_kernel_time / total_occurrences
            avg_overhead = avg_nvtx_time - avg_api_time
            
            print(f"  Average Timing Breakdown:")
            print(f"    Average NVTX range time: {avg_nvtx_time/1e6:.2f} ms")
            print(f"    Average CUDA API time: {avg_api_time/1e6:.2f} ms ({(avg_api_time/avg_nvtx_time)*100:.1f}%)")
            print(f"    Average kernel execution time: {avg_kernel_time/1e6:.2f} ms ({(avg_kernel_time/avg_nvtx_time)*100:.1f}%)")
            print(f"    Average overhead: {avg_overhead/1e6:.2f} ms ({(avg_overhead/avg_nvtx_time)*100:.1f}%)")
            
            # Store data for visualization
            nvtx_data[range_name] = {
                'avg_nvtx_time': avg_nvtx_time,
                'avg_api_time': avg_api_time,
                'avg_kernel_time': avg_kernel_time,
                'avg_overhead': avg_overhead,
                'total_occurrences': total_occurrences,
                'total_nvtx_time': total_nvtx_time
            }
        
        # Create runtime visualization
        create_nvtx_runtime_visualization(nvtx_data, db_path)
        
        # Create IEEE-styled pie chart
        create_nvtx_pie_chart(nvtx_data, db_path)
        
        # Overall summary across all ranges
        print(f"\n===== Overall Summary =====")
        
        # Get total time spent in all NVTX ranges
        cursor.execute("""
            WITH domains AS (
                SELECT
                    ne.domainId AS id,
                    ne.globalTid AS globalTid,
                    COALESCE(sid.value, ne.text) AS name
                FROM
                    NVTX_EVENTS AS ne
                LEFT JOIN
                    StringIds AS sid ON ne.textId = sid.id
                WHERE
                    ne.eventType = 75  -- NVTX_DOMAIN_CREATE
                GROUP BY ne.domainId, ne.globalTid
            )
            SELECT 
                SUM(ne.end - ne.start) as total_nvtx_time
            FROM
                NVTX_EVENTS AS ne
            LEFT JOIN
                domains AS d
                ON ne.domainId = d.id
                AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
            LEFT JOIN
                StringIds AS sid ON ne.textId = sid.id
            WHERE
                ne.eventType = 59 OR ne.eventType = 70;  -- NVTX_PUSHPOP_RANGE
        """)
        
        total_nvtx_time = cursor.fetchone()[0] or 0
        
        # Get total CUDA API time
        cursor.execute("""
            SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_RUNTIME;
        """)
        
        total_cuda_time = cursor.fetchone()[0] or 0
        
        # Get total kernel execution time
        cursor.execute("""
            SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL;
        """)
        
        total_kernel_time = cursor.fetchone()[0] or 0
        
        print(f"Total NVTX range time: {total_nvtx_time/1e6:.2f} ms")
        print(f"Total CUDA API time: {total_cuda_time/1e6:.2f} ms")
        print(f"Total kernel execution time: {total_kernel_time/1e6:.2f} ms")
        print(f"CUDA API coverage: {(total_cuda_time/total_nvtx_time)*100:.1f}%" if total_nvtx_time > 0 else "No NVTX ranges found")
        print(f"Kernel execution coverage: {(total_kernel_time/total_nvtx_time)*100:.1f}%" if total_nvtx_time > 0 else "No NVTX ranges found")
        
    except Exception as e:
        print(f"Error during NVTX analysis: {e}")
        raise
    finally:
        conn.close()


def create_nvtx_runtime_visualization(nvtx_data: dict, db_path: str):
    """
    Create runtime visualization for NVTX ranges showing kernel execution times.
    
    Args:
        nvtx_data: Dictionary containing NVTX range analysis data
        db_path: Path to the database for naming output files
    """
    if not nvtx_data:
        print("No NVTX data available for visualization")
        return
    
    # Create output directory using same logic as metrics graphs
    # Check if we should use original path (will be passed from main script)
    base_name = os.path.splitext(os.path.basename(db_path))[0]
    # Remove "_nvtx" suffix if present to get clean trace name
    if base_name.endswith("_nvtx"):
        base_name = base_name[:-5]
    output_dir = f"{base_name}_layer_graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    range_names = list(nvtx_data.keys())
    avg_kernel_times = [nvtx_data[name]['avg_kernel_time']/1e6 for name in range_names]  # Convert to ms
    
    # Get first occurrence time for each NVTX range
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    first_occurrence_times = {}
    for range_name in range_names:
        cursor.execute("""
            WITH domains AS (
                SELECT
                    ne.domainId AS id,
                    ne.globalTid AS globalTid,
                    COALESCE(sid.value, ne.text) AS name
                FROM
                    NVTX_EVENTS AS ne
                LEFT JOIN
                    StringIds AS sid ON ne.textId = sid.id
                WHERE
                    ne.eventType = 75  -- NVTX_DOMAIN_CREATE
                GROUP BY ne.domainId, ne.globalTid
            )
            SELECT MIN(ne.start) as first_start
            FROM
                NVTX_EVENTS AS ne
            LEFT JOIN
                domains AS d
                ON ne.domainId = d.id
                AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
            LEFT JOIN
                StringIds AS sid ON ne.textId = sid.id
            WHERE
                (ne.eventType = 59 OR ne.eventType = 70)  -- NVTX_PUSHPOP_RANGE
                AND (COALESCE(d.name, '') || ':' || COALESCE(sid.value, ne.text, '')) = ?
        """, (range_name,))
        
        result = cursor.fetchone()
        if result and result[0] is not None:
            first_occurrence_times[range_name] = result[0]
    
    conn.close()
    
    # Create kernel execution time bar chart sorted by first occurrence
    plt.figure(figsize=(16, 10))
    
    # Sort by first occurrence time
    sorted_by_occurrence = []
    for range_name in range_names:
        if range_name in first_occurrence_times:
            kernel_time = nvtx_data[range_name]['avg_kernel_time']/1e6
            sorted_by_occurrence.append((range_name, kernel_time, first_occurrence_times[range_name]))
    
    # Sort by first occurrence time
    sorted_by_occurrence.sort(key=lambda x: x[2])
    occurrence_names, occurrence_kernel_times, _ = zip(*sorted_by_occurrence)
    
    x = np.arange(len(occurrence_names))
    width = 0.8
    bars = plt.bar(x, occurrence_kernel_times, width, color='skyblue', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('NVTX Range', fontsize=12)
    plt.ylabel('Average Kernel Execution Time (ms)', fontsize=12)
    plt.title('Average Kernel Execution Time by NVTX Range (Excluding First Occurrence)\n(Ordered by First Occurrence)', fontsize=14, pad=20)
    
    # Rotate x-axis labels
    plt.xticks(x, occurrence_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(occurrence_kernel_times):
        plt.text(i, v + max(occurrence_kernel_times) * 0.01, f'{v:.2f}ms', ha='center', va='bottom', fontsize=9)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    kernel_filepath = os.path.join(output_dir, 'nvtx_kernel_execution_time.png')
    plt.savefig(kernel_filepath, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nRuntime visualizations saved in '{output_dir}/' folder:")
    print(f"- nvtx_kernel_execution_time.png - Kernel execution time by NVTX range (excluding first occurrence, ordered by first occurrence)")
    
    # Print summary statistics
    print(f"\n===== NVTX RUNTIME SUMMARY (Excluding First Occurrences) =====")
    total_kernel_time = sum(avg_kernel_times)
    
    print(f"Total kernel execution time: {total_kernel_time:.2f} ms")
    
    print(f"\nNVTX ranges ordered by first occurrence (data excludes first occurrence):")
    for i, (name, kernel_time, _) in enumerate(sorted_by_occurrence, 1):
        percentage = kernel_time / total_kernel_time * 100 if total_kernel_time > 0 else 0
        print(f"  {i}. {name}: {kernel_time:.2f} ms ({percentage:.1f}%)")


def create_nvtx_pie_chart(nvtx_data: dict, db_path: str):
    """
    Create an IEEE-styled pie chart showing kernel execution time distribution
    across NVTX ranges (layers).
    
    Args:
        nvtx_data: Dictionary containing NVTX range analysis data
        db_path: Path to the database for naming output files
    """
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts for IEEE
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if not nvtx_data:
        print("No NVTX data available for pie chart")
        return
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(db_path))[0]
    if base_name.endswith("_nvtx"):
        base_name = base_name[:-5]
    output_dir = f"{base_name}_layer_graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first occurrence time for ordering (same logic as bar chart)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    range_names = list(nvtx_data.keys())
    first_occurrence_times = {}
    for range_name in range_names:
        cursor.execute("""
            WITH domains AS (
                SELECT
                    ne.domainId AS id,
                    ne.globalTid AS globalTid,
                    COALESCE(sid.value, ne.text) AS name
                FROM
                    NVTX_EVENTS AS ne
                LEFT JOIN
                    StringIds AS sid ON ne.textId = sid.id
                WHERE
                    ne.eventType = 75
                GROUP BY ne.domainId, ne.globalTid
            )
            SELECT MIN(ne.start) as first_start
            FROM
                NVTX_EVENTS AS ne
            LEFT JOIN
                domains AS d
                ON ne.domainId = d.id
                AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
            LEFT JOIN
                StringIds AS sid ON ne.textId = sid.id
            WHERE
                (ne.eventType = 59 OR ne.eventType = 70)
                AND (COALESCE(d.name, '') || ':' || COALESCE(sid.value, ne.text, '')) = ?
        """, (range_name,))
        result = cursor.fetchone()
        if result and result[0] is not None:
            first_occurrence_times[range_name] = result[0]
    conn.close()
    
    # Sort by first occurrence time (same order as bar chart)
    sorted_data = []
    for range_name in range_names:
        if range_name in first_occurrence_times:
            kernel_time = nvtx_data[range_name]['avg_kernel_time'] / 1e6  # Convert to ms
            sorted_data.append((range_name, kernel_time, first_occurrence_times[range_name]))
    sorted_data.sort(key=lambda x: x[2])
    
    names = [item[0] for item in sorted_data]
    times = [item[1] for item in sorted_data]
    total_time = sum(times)
    percentages = [(t / total_time) * 100 for t in times]
    
    # Clean up layer names: strip leading colon and whitespace
    clean_names = [n.lstrip(':').strip() for n in names]
    
    # --- IEEE-style rcParams ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'text.usetex': False,
    })
    
    # IEEE single-column width is ~3.5 in, double-column ~7.16 in
    fig, ax = plt.subplots(figsize=(7.16, 4.5))
    
    # Color palette: muted, professional, distinguishable in grayscale
    # Using a curated set rather than a colormap for better control
    ieee_colors = [
        '#4878CF',  # blue
        '#6ACC65',  # green
        '#D65F5F',  # red
        '#B47CC7',  # purple
        '#C4AD66',  # olive
        '#77BEDB',  # light blue
        '#D4A373',  # tan
        '#92C5DE',  # sky blue
        '#F0B27A',  # peach
        '#82C785',  # sage
        '#AEB6BF',  # gray
        '#F1948A',  # salmon
    ]
    colors = ieee_colors[:len(names)]
    
    # Explode the largest slice slightly for emphasis
    max_idx = times.index(max(times))
    explode = [0.03 if i == max_idx else 0.0 for i in range(len(times))]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        times,
        labels=None,  # We'll use a legend instead for cleanliness
        autopct=lambda pct: f'{pct:.1f}%' if pct >= 3.0 else '',
        startangle=90,
        colors=colors,
        explode=explode,
        pctdistance=0.75,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
        textprops={'fontsize': 8, 'fontfamily': 'serif'},
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(7.5)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_aspect('equal')
    
    # Build legend labels: "Layer Name (X.XX ms, Y.Y%)"
    legend_labels = [
        f'{clean_names[i]} ({times[i]:.2f} ms, {percentages[i]:.1f}%)'
        for i in range(len(clean_names))
    ]
    
    ax.legend(
        wedges, legend_labels,
        title='Layer',
        title_fontproperties={'weight': 'bold', 'size': 9},
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        framealpha=1.0,
    )
    
    ax.set_title(
        'Kernel Execution Time Distribution by Layer',
        fontsize=11, fontweight='bold', fontfamily='serif',
        pad=12
    )
    
    # Add total time annotation
    ax.annotate(
        f'Total: {total_time:.2f} ms',
        xy=(0.5, -0.02), xycoords='axes fraction',
        ha='center', fontsize=8, fontstyle='italic', fontfamily='serif',
    )
    
    plt.tight_layout()
    
    pie_filepath = os.path.join(output_dir, 'nvtx_kernel_execution_time_pie.png')
    plt.savefig(pie_filepath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Also save as PDF for IEEE submission quality
    pdf_filepath = os.path.join(output_dir, 'nvtx_kernel_execution_time_pie.pdf')
    fig2, ax2 = plt.subplots(figsize=(7.16, 4.5))
    
    wedges2, texts2, autotexts2 = ax2.pie(
        times,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct >= 3.0 else '',
        startangle=90,
        colors=colors,
        explode=explode,
        pctdistance=0.75,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
        textprops={'fontsize': 8, 'fontfamily': 'serif'},
    )
    for autotext in autotexts2:
        autotext.set_fontsize(7.5)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax2.set_aspect('equal')
    ax2.legend(
        wedges2, legend_labels,
        title='Layer',
        title_fontproperties={'weight': 'bold', 'size': 9},
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        framealpha=1.0,
    )
    ax2.set_title(
        'Kernel Execution Time Distribution by Layer',
        fontsize=11, fontweight='bold', fontfamily='serif',
        pad=12
    )
    ax2.annotate(
        f'Total: {total_time:.2f} ms',
        xy=(0.5, -0.02), xycoords='axes fraction',
        ha='center', fontsize=8, fontstyle='italic', fontfamily='serif',
    )
    plt.tight_layout()
    fig2.savefig(pdf_filepath, bbox_inches='tight', format='pdf')
    plt.close()
    
    # Reset rcParams to defaults so we don't affect other plots
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"- nvtx_kernel_execution_time_pie.png - Kernel execution time pie chart (IEEE style)")
    print(f"- nvtx_kernel_execution_time_pie.pdf - Kernel execution time pie chart (PDF, IEEE submission quality)")


def get_metrics_by_nvtx_range(correlated_db_path: str, analyzer: NSysAnalyzer):
    """
    Get GPU metrics for every kernel within each NVTX range.
    
    Args:
        correlated_db_path: Path to the correlated database
        analyzer: NSysAnalyzer instance with metric_map
    
    Returns:
        Dictionary mapping NVTX range names to their kernel metrics
    """
    print(f"\n===== Getting Metrics by NVTX Range =====")
    
    # Connect to the correlated database
    conn = sqlite3.connect(correlated_db_path)
    cursor = conn.cursor()
    
    try:
        # Check if NVTX_EVENTS table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS';")
        if not cursor.fetchone():
            print("Warning: NVTX_EVENTS table not found in database.")
            print("This trace may not have NVTX data.")
            print("Skipping NVTX metrics analysis.")
            return {}
        
        # First, get all unique NVTX ranges
        print("Finding NVTX ranges...")
        cursor.execute("""
            WITH domains AS (
                SELECT
                    ne.domainId AS id,
                    ne.globalTid AS globalTid,
                    COALESCE(sid.value, ne.text) AS name
                FROM
                    NVTX_EVENTS AS ne
                LEFT JOIN
                    StringIds AS sid ON ne.textId = sid.id
                WHERE
                    ne.eventType = 75  -- NVTX_DOMAIN_CREATE
                GROUP BY ne.domainId, ne.globalTid
            )
            SELECT DISTINCT
                COALESCE(d.name, '') || ':' || COALESCE(sid.value, ne.text, '') AS fullname
            FROM
                NVTX_EVENTS AS ne
            LEFT JOIN
                domains AS d
                ON ne.domainId = d.id
                AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
            LEFT JOIN
                StringIds AS sid ON ne.textId = sid.id
            WHERE
                ne.eventType = 59  -- NVTX_PUSHPOP_RANGE
                OR ne.eventType = 70  -- NVTXT_PUSHPOP_RANGE
            ORDER BY fullname;
        """)
        
        nvtx_ranges = cursor.fetchall()
        
        if not nvtx_ranges:
            print("No NVTX ranges found in database.")
            return {}
        
        # Filter out "inference", "warmup", "fcn3_profiling", and "Global convolution" ranges
        filtered_ranges = [range_info for range_info in nvtx_ranges 
                          if not range_info[0].lower().startswith(":inference") 
                          and not range_info[0].lower().startswith(":warmup")
                          and range_info[0] != ":fcn3_profiling"
                          and range_info[0] != ":Global convolution"
                          and range_info[0] != ":forward_pass"]
        
        print(f"\nFound {len(filtered_ranges)} unique NVTX ranges (excluding 'inference', ':warmup*', ':fcn3_profiling', ':Global convolution', and ':forward_pass'):")
        for range_name in filtered_ranges:
            print(f"- {range_name[0]}")
        
        # Store metrics data for each NVTX range
        nvtx_metrics_data = {}
        
        # Process each NVTX range
        for range_info in filtered_ranges:
            range_name = range_info[0]
            print(f"\n--- Processing NVTX Range: {range_name} ---")
            
            # Get all occurrences of this NVTX range
            cursor.execute("""
                WITH domains AS (
                    SELECT
                        ne.domainId AS id,
                        ne.globalTid AS globalTid,
                        COALESCE(sid.value, ne.text) AS name
                    FROM
                        NVTX_EVENTS AS ne
                    LEFT JOIN
                        StringIds AS sid ON ne.textId = sid.id
                    WHERE
                        ne.eventType = 75  -- NVTX_DOMAIN_CREATE
                    GROUP BY ne.domainId, ne.globalTid
                )
                SELECT 
                    ne.start,
                    ne.end,
                    ne.end - ne.start as duration
                FROM
                    NVTX_EVENTS AS ne
                LEFT JOIN
                    domains AS d
                    ON ne.domainId = d.id
                    AND (ne.globalTid & 0x0000FFFFFF000000) = (d.globalTid & 0x0000FFFFFF000000)
                LEFT JOIN
                    StringIds AS sid ON ne.textId = sid.id
                WHERE
                    (ne.eventType = 59 OR ne.eventType = 70)  -- NVTX_PUSHPOP_RANGE
                    AND (COALESCE(d.name, '') || ':' || COALESCE(sid.value, ne.text, '')) = ?
                ORDER BY ne.start;
            """, (range_name,))
            
            range_times = cursor.fetchall()
            
            if not range_times:
                print("  No timing information found for this range")
                continue
            
            # Collect all kernels that execute within this NVTX range using correlation IDs (excluding first occurrence)
            all_kernels_in_range = []
            correlation_ids_used = set()
            
            range_times_filtered = range_times[1:]  # Skip first occurrence (startup outlier)
            if not range_times_filtered:
                print("  No occurrences remaining after excluding first occurrence")
                continue
            
            for start_time, end_time, _ in range_times_filtered:
                # Check if CUPTI_ACTIVITY_KIND_RUNTIME table exists for correlation
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_RUNTIME';")
                if not cursor.fetchone():
                    print("  No CUDA API runtime data available for correlation")
                    break
                
                # Get CUDA API calls within this NVTX range occurrence
                cursor.execute("""
                    SELECT 
                        name,
                        kernelName,
                        start,
                        end,
                        (end - start) as api_duration,
                        correlationId
                    FROM CUPTI_ACTIVITY_KIND_RUNTIME
                    WHERE start >= ? AND end <= ?
                    ORDER BY start;
                """, (start_time, end_time))
                
                cuda_calls = cursor.fetchall()
                
                # For each CUDA API call, get the corresponding kernel using correlation ID
                for api_name, kernel_name, api_start, api_end, api_duration, corr_id in cuda_calls:
                    if corr_id is not None:
                        # Get the actual kernel execution using the correlation ID
                        cursor.execute("""
                            SELECT 
                                COALESCE(sid.value, 'Unknown') as kernel_name,
                                k.correlationId,
                                k.start,
                                k.end,
                                k.end - k.start AS duration,
                                k.deviceId,
                                k.contextId,
                                k.streamId,
                                k.gridX,
                                k.gridY,
                                k.gridZ,
                                k.blockX,
                                k.blockY,
                                k.blockZ,
                                k.registersPerThread,
                                k.staticSharedMemory,
                                k.dynamicSharedMemory
                            FROM CUPTI_ACTIVITY_KIND_KERNEL k
                            LEFT JOIN StringIds sid ON k.shortName = sid.id
                            WHERE k.correlationId = ?
                        """, (corr_id,))
                        
                        kernel_execution = cursor.fetchone()
                        if kernel_execution:
                            all_kernels_in_range.append(kernel_execution)
                            correlation_ids_used.add(corr_id)
            
            if not all_kernels_in_range:
                print("  No kernels found in this range")
                continue
            
            print(f"  Found {len(all_kernels_in_range)} total kernel executions across filtered occurrences")
            print(f"  Correlation IDs used: {len(correlation_ids_used)}")
            
            # Calculate metrics for all kernels in this NVTX range
            # Use the existing get_metrics_by_layer method from NSysAnalyzer
            range_metrics = analyzer.get_metrics_by_layer(all_kernels_in_range)
            
            # Store the metrics data (using filtered range times)
            nvtx_metrics_data[range_name] = {
                'metrics': range_metrics,
                'kernel_count': len(all_kernels_in_range),
                'range_occurrences': len(range_times_filtered),
                'total_range_time': sum(duration for _, _, duration in range_times_filtered)
            }
            
            # Print summary for this range
            print(f"  Kernel count: {len(all_kernels_in_range)}")
            print(f"  Range occurrences: {len(range_times_filtered)} (excluding first occurrence)")
            print(f"  Total range time: {sum(duration for _, _, duration in range_times_filtered)/1e6:.2f} ms")
            
            if range_metrics:
                print(f"  Metrics found:")
                for metric_id, value in range_metrics.items():
                    metric_name = analyzer.metric_map.get(metric_id, f"Metric {metric_id}")
                    print(f"    {metric_name}: {value:.2f}%")
            else:
                print(f"  No metrics found for this range")
        
        # Print overall summary
        print(f"\n===== OVERALL METRICS SUMMARY =====")
        print(f"Total NVTX ranges processed: {len(nvtx_metrics_data)}")
        
        total_kernels = sum(data['kernel_count'] for data in nvtx_metrics_data.values())
        print(f"Total kernels across all ranges: {total_kernels}")
        
        # Show metrics summary for each range
        for range_name, data in nvtx_metrics_data.items():
            print(f"\n{range_name}:")
            print(f"  Kernels: {data['kernel_count']}")
            print(f"  Occurrences: {data['range_occurrences']}")
            print(f"  Total time: {data['total_range_time']/1e6:.2f} ms")
            
            if data['metrics']:
                for metric_id, value in data['metrics'].items():
                    metric_name = analyzer.metric_map.get(metric_id, f"Metric {metric_id}")
                    print(f"  {metric_name}: {value:.2f}%")
            else:
                print(f"  No metrics available")
        
        return nvtx_metrics_data
        
    except Exception as e:
        print(f"Error during NVTX metrics analysis: {e}")
        raise
    finally:
        conn.close()


# --- Aggregate by layer type and save to JSON for analyze_scaling.py ---
def aggregate_layer_metrics(nvtx_metrics):
    layer_runtime_data = {}
    metrics_by_layer_type = {}
    weighted_metrics_sum = {}  # Sum of (metric_value * duration)
    total_duration_by_layer = {}  # Total duration for each layer type
    
    # Assume NVTX range names are of the form 'LayerType:...' or just 'LayerType'
    for range_name, data in nvtx_metrics.items():
        # Extract layer type (before colon, or whole name)
        if ':' in range_name:
            # Split on colon and take the part after the domain
            parts = range_name.split(':', 1)
            if len(parts) > 1:
                layer_type = parts[1].strip()
            else:
                layer_type = parts[0].strip()
        else:
            layer_type = range_name.strip()
        
        # If layer_type is still empty, use the full range_name
        if not layer_type:
            layer_type = range_name
        
        # Get the total range time for this NVTX range
        range_duration = data.get('total_range_time', 0)
        
        # Aggregate total time
        layer_runtime_data.setdefault(layer_type, 0)
        layer_runtime_data[layer_type] += range_duration
        
        # Aggregate metrics with proper weighting
        if 'metrics' in data and data['metrics'] and range_duration > 0:
            if layer_type not in weighted_metrics_sum:
                weighted_metrics_sum[layer_type] = {}
                total_duration_by_layer[layer_type] = 0
            
            total_duration_by_layer[layer_type] += range_duration
            
            for metric_id, value in data['metrics'].items():
                weighted_metrics_sum[layer_type].setdefault(metric_id, 0)
                # Weight the metric value by the duration of this range
                weighted_metrics_sum[layer_type][metric_id] += value * range_duration
    
    # Calculate weighted averages for metrics
    for layer_type, metrics_sum in weighted_metrics_sum.items():
        if layer_type not in metrics_by_layer_type:
            metrics_by_layer_type[layer_type] = {}
        
        total_duration = total_duration_by_layer[layer_type]
        if total_duration > 0:
            for metric_id, weighted_sum in metrics_sum.items():
                # Calculate weighted average: sum(metric_value * duration) / total_duration
                metrics_by_layer_type[layer_type][metric_id] = weighted_sum / total_duration
    
    return layer_runtime_data, metrics_by_layer_type



# Initialize the analyzer
analyzer = NSysAnalyzer(db_path)
analyzer.connect()

# Perform the correlation analysis
correlated_db_path = correlate_cuda_kernels_with_api(db_path)

# Check if correlation was successful or if we're using the original database
if correlated_db_path == db_path:
    print(f"Using original database: {db_path}")
    # Reinitialize analyzer with original database
    analyzer.disconnect()
    analyzer = NSysAnalyzer(db_path)
    analyzer.connect()
else:
    # Reinitialize analyzer with correlated database
    analyzer.disconnect()
    analyzer = NSysAnalyzer(correlated_db_path)
    analyzer.connect()
    # Override the path used for naming to use the original clean path
    analyzer.naming_db_path = db_path

# Analyze NVTX ranges with CUDA timing (only if we have correlated data)
if correlated_db_path != db_path:
    analyze_nvtx_ranges_with_cuda_timing(correlated_db_path)
else:
    print("Skipping NVTX CUDA timing analysis (no correlation data available)")

# Get metrics for every kernel within each NVTX range
nvtx_metrics = get_metrics_by_nvtx_range(correlated_db_path, analyzer)

# Only proceed with NVTX analysis if we have data
if nvtx_metrics:
    # --- Aggregate by layer type and save to JSON for analyze_scaling.py ---
    layer_runtime_data, metrics_by_layer_type = aggregate_layer_metrics(nvtx_metrics)
    
    # Use original db_path instead of correlated_db_path to get clean trace name
    trace_id = os.path.splitext(os.path.basename(db_path))[0]
    # Remove "_nvtx" suffix to match existing JSON entries
    if trace_id.endswith("_nvtx"):
        trace_id = trace_id[:-5]
    
    # Merge into all_traces_summary.json directly
    unified_json_path = "all_traces_summary.json"
    if os.path.exists(unified_json_path):
        with open(unified_json_path, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {}
    if trace_id not in all_data:
        all_data[trace_id] = {}
    if 'layer' not in all_data[trace_id]:
        all_data[trace_id]['layer'] = {}
    all_data[trace_id]['layer']['layer_runtime_data'] = layer_runtime_data
    all_data[trace_id]['layer']['metrics_by_layer_type'] = metrics_by_layer_type
    with open(unified_json_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved layer summary for {trace_id} to {unified_json_path}")
    
    # Create NVTX metrics visualizations
    print(f"\n===== CREATING NVTX METRICS VISUALIZATIONS =====")
    analyzer.create_nvtx_metrics_visualizations(nvtx_metrics)
else:
    print("No NVTX metrics data available. Skipping NVTX analysis and visualizations.")

# You can now use the correlated database for further analysis
print(f"\nYou can now analyze the correlated data using: {correlated_db_path}")


