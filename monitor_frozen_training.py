import psutil
import argparse
import os
import time
import wandb


def monitor_last_log(log_file):
    while True:
        log_file_modified_time = os.path.getmtime(log_file)
        current_time = time.time()
        wandb.log({"log_file_modified_time": log_file_modified_time, "current_time": current_time})

        if current_time - log_file_modified_time > 60:
            print('The training is frozen. Rebooting the system...')
            os.system("sudo reboot")
        else:

            print(f"Last log update is within 1 minutes ({current_time - log_file_modified_time} s).")
            print(f"System uptime: {time.time() - psutil.boot_time()} seconds")

        time.sleep(0.5)  # Check every 30 seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor the last log file')
    parser.add_argument('--log_file', type=str, default='log.txt', help='The log file to monitor')
    parser.add_argument('--wandb_project', type=str, default='Log Monitor', help='The wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='The wandb run name')

    args = parser.parse_args()
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    monitor_last_log(args.log_file)