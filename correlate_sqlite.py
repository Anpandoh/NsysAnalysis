#!/usr/bin/env python3
"""
Correlate CUDA kernel launches with CUDA API in Nsight Systems SQLite traces.
Creates a new database with added columns (name, kernelName) on CUPTI_ACTIVITY_KIND_RUNTIME
by joining on correlationId to CUPTI_ACTIVITY_KIND_KERNEL.

Usage:
  python3 correlate_sqlite.py <input.sqlite> [output.sqlite]
  python3 correlate_sqlite.py *.sqlite   # correlate each file (output: <base>_correlated.sqlite)
"""

import sqlite3
import os
import shutil
import sys


def correlate_cuda_kernels_with_api(original_db_path: str, output_db_path: str = None):
    """
    Correlate CUDA kernel launches with CUDA API kernel launches by creating a new database
    with additional columns and correlated data.

    Args:
        original_db_path: Path to the original NSys SQLite database
        output_db_path: Path for the new correlated database (optional)

    Returns:
        Path to the correlated database (or original if correlation was skipped)
    """
    if output_db_path is None:
        base_name = os.path.splitext(os.path.basename(original_db_path))[0]
        output_db_path = f"{base_name}_correlated.sqlite"

    if not os.path.exists(original_db_path):
        print(f"Error: Input file not found: {original_db_path}")
        return None

    # Check if correlated database already exists
    if os.path.exists(output_db_path):
        print(f"Correlated database already exists: {output_db_path}")
        print("Skipping correlation step and using existing database.")
        return output_db_path

    print(f"Creating correlated database: {output_db_path}")

    # Copy the original database to preserve it
    shutil.copy2(original_db_path, output_db_path)

    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            print("Warning: Database is completely empty (no tables found).")
            conn.close()
            os.remove(output_db_path)
            return original_db_path

        if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
            print("Warning: CUPTI_ACTIVITY_KIND_RUNTIME table not found.")
            conn.close()
            os.remove(output_db_path)
            return original_db_path

        # Check if columns already exist (e.g. re-correlating)
        cursor.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME);")
        columns = [row[1] for row in cursor.fetchall()]
        if "kernelName" in columns:
            print("Database already has correlation columns. Skipping ALTER TABLE.")
        else:
            print("Adding correlation columns...")
            cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN name TEXT;")
            cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN kernelName TEXT;")

            print("Correlating kernel names...")
            cursor.execute("""
                UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET kernelName =
                    (SELECT value FROM StringIds
                    JOIN CUPTI_ACTIVITY_KIND_KERNEL AS cuda_gpu
                        ON cuda_gpu.shortName = StringIds.id
                        AND CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = cuda_gpu.correlationId);
            """)

            print("Updating API names...")
            cursor.execute("""
                UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET name =
                    (SELECT value FROM StringIds WHERE nameId = StringIds.id);
            """)

        conn.commit()

        # Summary
        cursor.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE kernelName IS NOT NULL"
        )
        correlated_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        total_count = cursor.fetchone()[0]

        print(f"\nTotal CUDA API calls: {total_count}")
        print(f"Correlated with kernel launches: {correlated_count}")
        if total_count > 0:
            print(f"Correlation rate: {(correlated_count / total_count) * 100:.1f}%")
        print(f"Correlated database saved to: {output_db_path}")
        return output_db_path

    except Exception as e:
        print(f"Error during correlation: {e}")
        conn.rollback()
        try:
            os.remove(output_db_path)
        except OSError:
            pass
        raise
    finally:
        conn.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 correlate_sqlite.py <input.sqlite> [output.sqlite]")
        print("       python3 correlate_sqlite.py file1.sqlite file2.sqlite ...")
        print("\nWith one file: writes <base>_correlated.sqlite unless output path is given.")
        print("With two args: input and output path.")
        print("With 3+ files: correlates each; output is <base>_correlated.sqlite per file.")
        sys.exit(1)

    args = [p for p in sys.argv[1:] if not p.startswith("-")]
    if not args:
        print("No input files given.")
        sys.exit(1)

    # Two args: input + output. One arg: input only. 3+: multiple inputs.
    if len(args) == 2:
        result = correlate_cuda_kernels_with_api(args[0], args[1])
        if result is None:
            sys.exit(1)
        return
    if len(args) == 1:
        result = correlate_cuda_kernels_with_api(args[0], None)
        if result is None:
            sys.exit(1)
        return

    # Multiple files: each -> <base>_correlated.sqlite
    for input_path in args:
        print("\n" + "=" * 60)
        print(f"Input: {input_path}")
        print("=" * 60)
        result = correlate_cuda_kernels_with_api(input_path)
        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
