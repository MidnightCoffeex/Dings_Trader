"""
Multi-Instance Manager for dings-trader.
Manages multiple paper trading instances using PM2.
"""
import os
import subprocess
import argparse
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(PROJECT_ROOT, "ml")
PYTHON_INTERPRETER = os.path.join(PROJECT_ROOT, "TraderHimSelf", "venv", "bin", "python")

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")
        return None

def start_instance(model_id, symbol="BTCUSDT", interval=60, warmup=1500, package_id=None):
    instance_name = f"dt-loop-{model_id}"
    cmd = (
        f"pm2 start {os.path.join(ML_DIR, 'paper_inference_loop.py')} "
        f"--name {instance_name} "
        f"--interpreter {PYTHON_INTERPRETER} "
        f"-- "
        f"--model-id {model_id} "
        f"--symbol {symbol} "
        f"--interval {interval} "
        f"--warmup-candles {warmup} "
        f"--create-account"
    )
    print(f"Starting instance: {instance_name}...")
    
    # Pre-create account with mapping if provided
    if package_id:
        try:
            import requests
            requests.post("http://127.0.0.1:8000/paper/account/create", json={
                "model_id": model_id,
                "model_package_id": package_id
            }, timeout=5)
        except Exception as e:
            print(f"Warning: Could not pre-create account via API: {e}")

    return run_command(cmd)

def stop_instance(model_id):
    instance_name = f"dt-loop-{model_id}"
    print(f"Stopping instance: {instance_name}...")
    return run_command(f"pm2 stop {instance_name}")

def delete_instance(model_id):
    instance_name = f"dt-loop-{model_id}"
    print(f"Deleting instance: {instance_name}...")
    return run_command(f"pm2 delete {instance_name}")

def list_instances():
    stdout = run_command("pm2 jlist")
    if not stdout:
        return []
    
    apps = json.loads(stdout)
    instances = []
    for app in apps:
        if app['name'].startswith("dt-loop-"):
            instances.append({
                "name": app['name'],
                "status": app['pm2_env']['status'],
                "cpu": app['monit']['cpu'],
                "memory": app['monit']['memory'],
                "uptime": app['pm2_env']['pm_uptime']
            })
    return instances

def main():
    parser = argparse.ArgumentParser(description="Multi-Instance Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start
    start_parser = subparsers.add_parser("start", help="Start an instance")
    start_parser.add_argument("--model-id", required=True, help="Model ID")
    start_parser.add_argument("--package-id", help="Model Package ID")
    start_parser.add_argument("--symbol", default="BTCUSDT", help="Symbol")
    start_parser.add_argument("--interval", type=int, default=60, help="Interval")
    start_parser.add_argument("--warmup", type=int, default=1500, help="Warmup candles")

    # Stop
    stop_parser = subparsers.add_parser("stop", help="Stop an instance")
    stop_parser.add_argument("--model-id", required=True, help="Model ID")

    # Delete
    delete_parser = subparsers.add_parser("delete", help="Delete an instance")
    delete_parser.add_argument("--model-id", required=True, help="Model ID")

    # Status
    subparsers.add_parser("status", help="List instances status")

    args = parser.parse_args()

    if args.command == "start":
        start_instance(args.model_id, args.symbol, args.interval, args.warmup, args.package_id)
    elif args.command == "stop":
        stop_instance(args.model_id)
    elif args.command == "delete":
        delete_instance(args.model_id)
    elif args.command == "status":
        instances = list_instances()
        if not instances:
            print("No active instances found.")
        else:
            print(f"{'Name':<25} {'Status':<10} {'CPU':<10} {'Memory':<10}")
            print("-" * 60)
            for inst in instances:
                print(f"{inst['name']:<25} {inst['status']:<10} {inst['cpu']:<10} {inst['memory']/1024/1024:<10.1f} MB")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
