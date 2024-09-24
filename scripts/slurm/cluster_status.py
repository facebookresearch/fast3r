import subprocess
import re
import sys
import argparse
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------- USAGE GUIDE ----------------------
# This script fetches and displays the GPU, CPU, and memory usage 
# per user for each Quality of Service (QOS) associated with a given account. 
# It shows the resources used in active jobs for each QOS and the 
# total allowable resources for each QOS.
# 
# Usage:
# python script.py --account <account_name>
# 
# Example:
# python script.py --account cortex
# ---------------------------------------------------------

# Initialize the console
console = Console()

# Setup argparse to handle account as command-line argument (no QOS argument)
parser = argparse.ArgumentParser(description='Fetch resource usage from SLURM.')
parser.add_argument('--account', type=str, required=True, help='Account name for SLURM query')

args = parser.parse_args()
account_name = args.account

# Function to extract available QOS for the account using the pipe-separated format
def get_qos_for_account(account):
    sacctmgr_qos_command = (
        f"sacctmgr show assoc format=Account,User,QOS where Account={account} -P"
    )
    
    # Run the command and capture stdout and stderr
    result_qos = subprocess.run(sacctmgr_qos_command, shell=True, capture_output=True, text=True)
    
    output = result_qos.stdout.strip()
    
    # Extract the QOS column (3rd column) from the pipe-separated output
    qos_set = set()  # Use a set to avoid duplicates
    for line in output.splitlines()[1:]:  # Skip the header
        fields = line.split("|")
        if len(fields) == 3 and fields[2].strip():  # Ensure we have 3 fields and QOS is not empty
            # QOS can have multiple values, split them by commas
            qos_values = fields[2].split(",")
            for qos in qos_values:
                qos_set.add(qos.strip())  # Strip any whitespace
    
    # Ensure that "lowest" QOS comes first, if present
    qos_list = sorted(qos_set, key=lambda x: (x != 'lowest', x))
    return qos_list

# Fetch QOS values associated with the account
qos_list = get_qos_for_account(account_name)

# Check if we found any QOS values
if not qos_list:
    console.print(f"[red]No QOS found for account {account_name}[/red]")
    sys.exit(1)

# Function to extract CPU, GPU, and memory usage from the ReqTRES or AllocTRES column
def extract_tres_usage(tres_str):
    cpu_usage = gpu_usage = memory_usage = 0
    if 'cpu=' in tres_str:
        cpu_match = re.search(r'cpu=([0-9]+)', tres_str)
        cpu_usage = int(cpu_match.group(1)) if cpu_match else 0
    if 'gres/gpu=' in tres_str:
        gpu_match = re.search(r'gres/gpu=([0-9]+)', tres_str)
        gpu_usage = int(gpu_match.group(1)) if gpu_match else 0
    if 'mem=' in tres_str:
        mem_match = re.search(r'mem=([0-9]+)([A-Za-z]+)', tres_str)
        if mem_match:
            mem_value, mem_unit = int(mem_match.group(1)), mem_match.group(2)
            # Convert memory to GB for consistent reporting
            memory_usage = mem_value if mem_unit == 'G' else mem_value / 1024 if mem_unit == 'M' else mem_value * 1024
    return cpu_usage, memory_usage, gpu_usage

# Loop through each QOS for the account
for qos_name in qos_list:
    console.print(f"\n[bold green]Fetching data for QOS: {qos_name}[/bold green]\n")
    
    # Construct the sacctmgr command to get total allowable resources for the QOS
    sacctmgr_command = [
        "sacctmgr", 
        "show", 
        "qos", 
        qos_name, 
        "format=GrpTRES%50", 
        "-P"
    ]
    
    # Fetch total allowable resources for the QOS
    result_qos_resources = subprocess.run(sacctmgr_command, capture_output=True, text=True)
    qos_output = result_qos_resources.stdout.strip()
    
    # Extract the CPU, GPU, and memory limits from GrpTRES
    cpu_limit = gpu_limit = memory_limit = "N/A"
    cpu_match = re.search(r'cpu=([0-9]+)', qos_output)
    gpu_match = re.search(r'gres/gpu=([0-9]+)', qos_output)
    mem_match = re.search(r'mem=([0-9]+)([A-Za-z]+)', qos_output)

    if cpu_match:
        cpu_limit = int(cpu_match.group(1))
    if gpu_match:
        gpu_limit = int(gpu_match.group(1))
    if mem_match:
        mem_value, mem_unit = int(mem_match.group(1)), mem_match.group(2)
        # Convert memory to GB
        memory_limit = mem_value if mem_unit == 'G' else mem_value / 1024 if mem_unit == 'M' else mem_value * 1024
    
    # Fetch job information for the given account and QOS
    sacct_command = [
        "sacct", 
        "-a", 
        "--qos=" + qos_name, 
        "--account=" + account_name,
        "--format=JobID,User%20,Partition,JobName,State,ReqTRES%60,AllocTRES%60", 
        "-P"
    ]
    result_jobs = subprocess.run(sacct_command, capture_output=True, text=True)
    job_output = result_jobs.stdout.strip()
    
    # Split the output into lines
    job_lines = job_output.splitlines()

    # Split the first row as the header
    header = job_lines[0].split("|")

    # Parse the output rows into a list of dictionaries
    job_data = []
    for line in job_lines[1:]:
        fields = line.split("|")
        if len(fields) == len(header):
            job_data.append(dict(zip(header, fields)))

    # Collect CPU, GPU, and memory usage per user for RUNNING jobs only, using ReqTRES for memory
    cpu_usage_per_user = {}
    mem_usage_per_user = {}
    gpu_usage_per_user = {}
    grand_total_cpu_usage = 0
    grand_total_mem_usage = 0
    grand_total_gpu_usage = 0

    for job in job_data:
        job_id = job['JobID']
        user = job['User'].strip()
        state = job['State'].strip()

        # Only consider master jobs (JobID without dots), non-empty users, and RUNNING jobs
        if '.' not in job_id and user and state == 'RUNNING':
            # Use ReqTRES for memory usage since AllocTRES doesn't show it
            cpu_usage, mem_usage, gpu_usage = extract_tres_usage(job['ReqTRES'])

            if user not in cpu_usage_per_user:
                cpu_usage_per_user[user] = 0
                mem_usage_per_user[user] = 0
                gpu_usage_per_user[user] = 0

            cpu_usage_per_user[user] += cpu_usage
            mem_usage_per_user[user] += int(mem_usage)  # rounding memory usage to integer
            gpu_usage_per_user[user] += gpu_usage

            grand_total_cpu_usage += cpu_usage
            grand_total_mem_usage += int(mem_usage)
            grand_total_gpu_usage += gpu_usage

    # Create a table using rich (without a divider)
    table = Table(title=f"Resource Usage Summary per User (QOS: {qos_name})", box=box.SQUARE)

    # Add columns to the table (GPU moved after user)
    table.add_column("User", justify="left", style="cyan", no_wrap=True)
    table.add_column("Total GPU Usage", justify="right", style="magenta")
    table.add_column("Total CPU Usage", justify="right", style="magenta")
    table.add_column("Total Memory Usage (GB)", justify="right", style="magenta")

    # Add rows for each user with commas for readability
    for user in cpu_usage_per_user:
        table.add_row(
            user, 
            f"{gpu_usage_per_user[user]:,}", 
            f"{cpu_usage_per_user[user]:,}", 
            f"{mem_usage_per_user[user]:,}"
        )

    # Add rows for the grand total
    table.add_row("Grand Total", f"{grand_total_gpu_usage:,}", f"{grand_total_cpu_usage:,}", f"{grand_total_mem_usage:,}", style="bold")

    # Format the total allowable resources and add to the table
    formatted_cpu_limit = f"{cpu_limit:,}" if cpu_limit != "N/A" else cpu_limit
    formatted_gpu_limit = f"{gpu_limit:,}" if gpu_limit != "N/A" else gpu_limit
    formatted_mem_limit = f"{int(memory_limit):,}" if memory_limit != "N/A" else memory_limit

    # Add a row for the total allowable resources for the specific QOS
    table.add_row(f"Total Allowable Resources (QOS: {qos_name})", formatted_gpu_limit, formatted_cpu_limit, formatted_mem_limit, style="bold cyan")

    # Display the table for the current QOS
    console.print(table)
