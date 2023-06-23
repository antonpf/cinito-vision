#!/bin/bash

IP_ADDRESS="172.19.12.1"  # Replace with the IP address you want to ping
MAX_FAILED_COUNT=5        # Maximum consecutive ping failures before restarting

ping -c 1 $IP_ADDRESS >/dev/null  # Send a single ICMP echo request
if [ $? -ne 0 ]; then
    # IP address is not reachable
    failed_count_file="/tmp/network_restart_failed_count"
    if [ ! -f "$failed_count_file" ]; then
        echo 1 > "$failed_count_file"
    else
        failed_count=$(<"$failed_count_file")
        ((failed_count++))
        echo $failed_count > "$failed_count_file"
        if [ $failed_count -ge $MAX_FAILED_COUNT ]; then
            echo "Restarting NetworkManager..."
            systemctl restart NetworkManager  # Restart NetworkManager
            echo 0 > "$failed_count_file"
        fi
    fi
else
    # IP address is reachable, reset failed_count
    failed_count_file="/tmp/network_restart_failed_count"
    if [ -f "$failed_count_file" ]; then
        rm "$failed_count_file"
    fi
fi
