#!/usr/bin/env python3
"""
Compare event counts between datasets with and without Ucav/Uci filtering
"""
import pickle
from pathlib import Path
import glob

def count_events_in_directory(cooked_dir):
    """Count total events in all batch files"""
    batch_dir = Path(cooked_dir) / 'batches'
    batch_files = sorted(glob.glob(str(batch_dir / 'batch_*.pkl')))

    total_events = 0
    batch_info = []

    for bf in batch_files:
        try:
            with open(bf, 'rb') as f:
                bd = pickle.load(f)
                n_events = bd.get('n_events', 0)
                total_events += n_events
                batch_info.append((Path(bf).name, n_events))
        except Exception as e:
            print(f"  Error reading {bf}: {e}")

    return total_events, batch_info, len(batch_files)

# Directories to compare
dir_with_filter = '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'
dir_without_filter = '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_no_ucav_uci_filtering'

print("="*80)
print("COMPARING DATASETS WITH AND WITHOUT UCAV/UCI FILTERING")
print("="*80)

# Count events with filtering
print(f"\n1. Dataset WITH Ucav/Uci filtering:")
print(f"   Directory: {dir_with_filter}")
if Path(dir_with_filter).exists():
    total_with, batches_with, n_batches_with = count_events_in_directory(dir_with_filter)
    print(f"   Number of batches: {n_batches_with}")
    print(f"   Total events: {total_with}")
else:
    print(f"   Directory does not exist!")
    total_with = 0

# Count events without filtering
print(f"\n2. Dataset WITHOUT Ucav/Uci filtering:")
print(f"   Directory: {dir_without_filter}")
if Path(dir_without_filter).exists():
    total_without, batches_without, n_batches_without = count_events_in_directory(dir_without_filter)
    print(f"   Number of batches: {n_batches_without}")
    print(f"   Total events: {total_without}")
else:
    print(f"   Directory does not exist!")
    total_without = 0

# Calculate rejection
print(f"\n{'='*80}")
print("FILTERING IMPACT ANALYSIS")
print(f"{'='*80}")

if total_without > 0:
    events_rejected = total_without - total_with
    rejection_rate = 100 * events_rejected / total_without

    print(f"Events WITHOUT Ucav/Uci filter: {total_without}")
    print(f"Events WITH Ucav/Uci filter:    {total_with}")
    print(f"Events rejected by filter:       {events_rejected}")
    print(f"Rejection rate:                  {rejection_rate:.1f}%")
    print(f"Pass rate:                       {100-rejection_rate:.1f}%")
else:
    print("Cannot calculate - no data without filtering")

# Check timestamps from processing summaries
print(f"\n{'='*80}")
print("TIMESTAMPS FROM PROCESSING SUMMARIES")
print(f"{'='*80}")

for label, directory in [("WITH filter", dir_with_filter),
                          ("WITHOUT filter", dir_without_filter)]:
    summary_file = Path(directory) / 'processing_summary.txt'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            for line in f:
                if 'Timestamp:' in line:
                    print(f"{label}: {line.strip()}")
                    break
    else:
        print(f"{label}: No summary file found")
