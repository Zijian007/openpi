import os
import sys
import re
from pathlib import Path
from collections import defaultdict

def calculate_accuracy(folder_path):
    """
    Calculate the success rate of samples in a folder.
    Files ending with 'success' are counted as successful,
    files ending with 'failure' are counted as failures.
    
    Args:
        folder_path: Path to the folder containing the files
    
    Returns:
        tuple: (success_count, failure_count, total_count, accuracy)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    success_count = 0
    failure_count = 0
    
    # Iterate through all files in the folder
    for file in folder.iterdir():
        if file.is_file():
            file_name = file.stem  # Get filename without extension
            
            if file_name.endswith('_success'):
                success_count += 1
            elif file_name.endswith('_failure'):
                failure_count += 1
    
    total_count = success_count + failure_count
    
    if total_count == 0:
        print("No files with '_success' or '_failure' suffix found.")
        return 0, 0, 0, 0.0
    
    accuracy = (success_count / total_count) * 100
    
    return success_count, failure_count, total_count, accuracy


def calculate_accuracy_by_task(folder_path):
    """
    Calculate the success rate for each task separately.
    Files are expected to have names like: task00_ep000_..._success.mp4 or task00_ep000_..._failure.mp4
    
    Args:
        folder_path: Path to the folder containing the files
    
    Returns:
        dict: Dictionary with task numbers as keys and (success, failure, total, accuracy) as values
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    task_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
    
    # Iterate through all files in the folder
    for file in folder.iterdir():
        if file.is_file():
            file_name = file.stem  # Get filename without extension
            
            # Extract task number using regex (e.g., task00, task01, etc.)
            match = re.match(r'task(\d+)_', file_name)
            if match:
                task_num = match.group(1)  # Get the task number as string (e.g., "00", "01")
                
                if file_name.endswith('_success'):
                    task_stats[task_num]['success'] += 1
                elif file_name.endswith('_failure'):
                    task_stats[task_num]['failure'] += 1
    
    # Calculate accuracy for each task
    results = {}
    for task_num in sorted(task_stats.keys()):
        success = task_stats[task_num]['success']
        failure = task_stats[task_num]['failure']
        total = success + failure
        accuracy = (success / total * 100) if total > 0 else 0.0
        results[task_num] = (success, failure, total, accuracy)
    
    return results


def print_accuracy_report(folder_path, by_task=True):
    """
    Print accuracy report for a given folder.
    
    Args:
        folder_path: Path to the folder containing the files
        by_task: If True, also print per-task statistics
    """
    try:
        # Overall accuracy
        success, failure, total, accuracy = calculate_accuracy(folder_path)
        
        print(f"Folder: {folder_path}")
        print(f"{'='*80}")
        print(f"OVERALL STATISTICS:")
        print(f"  Success count: {success}")
        print(f"  Failure count: {failure}")
        print(f"  Total samples: {total}")
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        print()
        
        # Per-task accuracy
        if by_task:
            task_results = calculate_accuracy_by_task(folder_path)
            
            if task_results:
                print(f"PER-TASK STATISTICS:")
                print(f"{'-'*80}")
                print(f"{'Task':<8} {'Success':<10} {'Failure':<10} {'Total':<10} {'Accuracy':<10}")
                print(f"{'-'*80}")
                
                for task_num in sorted(task_results.keys()):
                    task_success, task_failure, task_total, task_accuracy = task_results[task_num]
                    print(f"task{task_num:<4} {task_success:<10} {task_failure:<10} {task_total:<10} {task_accuracy:>6.2f}%")
                
                print(f"{'-'*80}")
            else:
                print("No task-specific files found (files should start with 'taskXX_')")
        
        print()
        return success, failure, total, accuracy, task_results if by_task else None
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    # 默认文件夹路径 - 可以在这里修改
    DEFAULT_FOLDER = "/hdd/zijianwang/openpi/experiments/videos/20251101_221331_libero_spatial_displacement"
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) >= 2:
        folder_path = sys.argv[1]
    else:
        # 否则使用默认路径
        folder_path = DEFAULT_FOLDER
        print(f"Using default folder: {folder_path}")
        print(f"(You can override by: python computy_acc.py <folder_path>)")
        print()
    
    print_accuracy_report(folder_path)


if __name__ == "__main__":
    main()

