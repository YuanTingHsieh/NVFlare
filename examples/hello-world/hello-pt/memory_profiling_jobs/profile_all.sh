#!/bin/bash
# Profile all 3 jobs and compare memory usage

set -euo pipefail

INTERVAL=0.2  # Sample every 200ms

echo "========================================"
echo "Memory Profiling: 3-Way Comparison"
echo "========================================"
echo "Job 1: FedAvg (standard)"
echo "Job 2: Scatter and Gather"
echo "Job 3: FedAvg (memory-efficient)"
echo ""
echo "Clients: 1"
echo "Rounds: 3"
echo "========================================"
echo

# Function to recursively get all descendant PIDs (avoids duplicates)
get_descendants() {
    local parent=$1
    pgrep -P "$parent" 2>/dev/null | while read -r child; do
        echo "$child"
        get_descendants "$child"
    done
}

# Function to profile a single job
profile_job() {
    local JOB_NUM=$1
    local JOB_NAME=$2
    local JOB_SCRIPT=$3
    
    echo ">>> Profiling $JOB_NAME..."
    echo
    
    # Clean previous result file for this job
    RESULT_FILE="results/results_job${JOB_NUM}.dat"
    rm -f "$RESULT_FILE"
    
    # Start job in background
    python3 "$JOB_SCRIPT" &
    JOB_PID=$!
    
    echo "Job PID: $JOB_PID"
    
    # Monitor memory
    PEAK_RSS=0
    PEAK_SERVER=0
    PEAK_CLIENT=0
    PEAK_TIME=0
    START_TIME=$(date +%s)
    
    echo "Monitoring memory (Server vs Client breakdown)..."
    echo "Time(s)  Total(MB)  Server  Client  Children" | tee -a results/summary.txt
    echo "-------  ---------  ------  ------  --------" | tee -a results/summary.txt
    
    DEBUG_PRINTED=""
    
    while kill -0 $JOB_PID 2>/dev/null; do
        ELAPSED=$(($(date +%s) - START_TIME))
        
        # Get memory of main process and ALL descendants (entire process tree)
        TOTAL_RSS=0
        SERVER_RSS=0
        CLIENT_RSS=0
        NUM_CHILDREN=0
        
        # Get all PIDs in the process tree (recursively) and deduplicate
        ALL_PIDS="$JOB_PID $(get_descendants $JOB_PID)"
        ALL_PIDS=$(echo "$ALL_PIDS" | tr ' ' '\n' | sort -u | tr '\n' ' ')
        
        for PID in $ALL_PIDS; do
            if [ -d "/proc/$PID" ]; then
                RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ' || echo 0)
                if [ ! -z "$RSS" ] && [ "$RSS" != "0" ]; then
                    TOTAL_RSS=$((TOTAL_RSS + RSS))
                    
                    # Check process command to categorize as server or client
                    # Server: Main process + any non-client-worker subprocess
                    # Client: subprocess with "simulator_worker" AND "--client"
                    CMDLINE=$(cat /proc/$PID/cmdline 2>/dev/null | tr '\0' ' ')
                    
                    # Categorize: Must have BOTH "simulator_worker" AND "--client" to be a client process
                    IS_CLIENT=""
                    if echo "$CMDLINE" | grep -q "simulator_worker" && echo "$CMDLINE" | grep -q "\--client"; then
                        CLIENT_RSS=$((CLIENT_RSS + RSS))
                        IS_CLIENT="CLIENT"
                    else
                        # Everything else is server (main process + server threads)
                        SERVER_RSS=$((SERVER_RSS + RSS))
                        IS_CLIENT="SERVER"
                    fi
                    
                    # Debug: Print first detection (can be removed later)
                    if [ -z "$DEBUG_PRINTED" ] && [ $ELAPSED -gt 10 ]; then
                        RSS_MB_DBG=$((RSS / 1024))
                        PARENT_PID=$(ps -o ppid= -p $PID 2>/dev/null | tr -d ' ')
                        echo "[DEBUG] PID $PID (parent: $PARENT_PID): ${RSS_MB_DBG}MB [$IS_CLIENT] - CMD: ${CMDLINE:0:100}" 1>&2
                    fi
                    
                    if [ "$PID" != "$JOB_PID" ]; then
                        NUM_CHILDREN=$((NUM_CHILDREN + 1))
                    fi
                fi
            fi
        done
        
        # Mark debug as printed after first time at 10+ seconds
        if [ $ELAPSED -gt 10 ] && [ -z "$DEBUG_PRINTED" ]; then
            DEBUG_PRINTED=1
        fi
        
        # Convert KB to MB
        RSS_MB=$((TOTAL_RSS / 1024))
        SERVER_MB=$((SERVER_RSS / 1024))
        CLIENT_MB=$((CLIENT_RSS / 1024))
        
        # Track peak
        if [ $RSS_MB -gt $PEAK_RSS ]; then
            PEAK_RSS=$RSS_MB
            PEAK_SERVER=$SERVER_MB
            PEAK_CLIENT=$CLIENT_MB
            PEAK_TIME=$ELAPSED
        fi
        
        # Output
        printf "\r%7d  %7d  (S:%5d C:%5d) Children:%2d" $ELAPSED $RSS_MB $SERVER_MB $CLIENT_MB $NUM_CHILDREN
        
        # Log to file (time, total, server, client)
        echo "$ELAPSED $RSS_MB $SERVER_MB $CLIENT_MB" >> "$RESULT_FILE"
        
        sleep $INTERVAL
    done
    
    wait $JOB_PID
    
    echo
    echo
    echo "Peak memory: ${PEAK_RSS} MB at ${PEAK_TIME}s" | tee -a results/summary.txt
    echo "  - Server: ${PEAK_SERVER} MB" | tee -a results/summary.txt
    echo "  - Client: ${PEAK_CLIENT} MB" | tee -a results/summary.txt
    echo "Results saved to: $RESULT_FILE" | tee -a results/summary.txt
    echo | tee -a results/summary.txt
    
    # Return peak memory
    echo $PEAK_RSS
}

# Create results directory and clean previous run
mkdir -p results
rm -f results/summary.txt
rm -f results/results_job*.dat
rm -f results/memory_comparison.png

# Profile each job
echo "Job 1: FedAvg (Standard)" | tee results/summary.txt
echo "========================" | tee -a results/summary.txt
PEAK1=$(profile_job 1 "FedAvg (Standard)" "job1_fedavg_standard.py")

sleep 2

echo "Job 2: Scatter and Gather" | tee -a results/summary.txt
echo "=========================" | tee -a results/summary.txt
PEAK2=$(profile_job 2 "Scatter and Gather" "job2_scatter_gather.py")

sleep 2

echo "Job 3: FedAvg (Memory-Efficient)" | tee -a results/summary.txt
echo "================================" | tee -a results/summary.txt
PEAK3=$(profile_job 3 "FedAvg (Memory-Efficient)" "job3_fedavg_memory_efficient.py")

# # Final comparison
# echo "========================================"  | tee -a results/summary.txt
# echo "FINAL COMPARISON" | tee -a results/summary.txt
# echo "========================================"  | tee -a results/summary.txt
# echo "Job 1 (FedAvg Standard):        ${PEAK1} MB" | tee -a results/summary.txt
# echo "Job 2 (Scatter and Gather):     ${PEAK2} MB" | tee -a results/summary.txt
# echo "Job 3 (FedAvg Memory-Efficient): ${PEAK3} MB" | tee -a results/summary.txt
# echo | tee -a results/summary.txt

# # Calculate savings
# SAVING_VS_STANDARD=$((PEAK1 - PEAK3))
# SAVING_VS_SAG=$((PEAK2 - PEAK3))

# if [ $SAVING_VS_STANDARD -gt 0 ]; then
#     PERCENT=$(awk "BEGIN {printf \"%.1f\", ($SAVING_VS_STANDARD / $PEAK1) * 100}")
#     echo "Savings vs FedAvg Standard:      ${SAVING_VS_STANDARD} MB (${PERCENT}%)" | tee -a results/summary.txt
# fi

# if [ $SAVING_VS_SAG -gt 0 ]; then
#     PERCENT=$(awk "BEGIN {printf \"%.1f\", ($SAVING_VS_SAG / $PEAK2) * 100}")
#     echo "Savings vs Scatter and Gather:   ${SAVING_VS_SAG} MB (${PERCENT}%)" | tee -a results/summary.txt
# fi

# echo "========================================" | tee -a results/summary.txt
# echo | tee -a results/summary.txt
# echo "Full results saved to: results/summary.txt"

# # Generate plot if gnuplot is available
# if command -v gnuplot &> /dev/null; then
#     echo "Generating plot..."
     
#     gnuplot <<EOF
# set terminal png size 1400,800
# set output 'results/memory_comparison_3way.png'
# set title 'Memory Usage Comparison: FedAvg vs Scatter and Gather (600MB model, 3 clients)'
# set xlabel 'Time (seconds)'
# set ylabel 'Memory (MB)'
# set grid
# set key left top
 
# plot 'results/results_job1.dat' using 1:2 with lines lw 2 title 'Job 1: FedAvg (Standard)' linecolor rgb 'red', \
#      'results/results_job2.dat' using 1:2 with lines lw 2 title 'Job 2: Scatter and Gather' linecolor rgb 'blue', \
#      'results/results_job3.dat' using 1:2 with lines lw 2 title 'Job 3: FedAvg (Memory-Efficient)' linecolor rgb 'green'
# EOF
     
#     echo "Plot saved to: results/memory_comparison_3way.png"
# fi

# echo
# echo "Done!"
