#!/bin/bash

# Batch download MG datasets from HuggingFace (parallel)
# Usage: ./download_mg_datasets.sh [N]
#   N: number of parallel processes (default: 4)

LOCAL_DIR="/data/sunyi/robocasa/v0.1/single_stage_mg"
LOG_DIR="/home/sunyi/robocasa/download_logs"
N=${1:-4}

mkdir -p "$LOG_DIR"

TASKS=(
    # "single_panda_gripper.CloseDoubleDoor"
    # "single_panda_gripper.CloseDrawer"
    # "single_panda_gripper.CloseSingleDoor"
    # "single_panda_gripper.CoffeePressButton"
    # "single_panda_gripper.CoffeeServeMug"
    # "single_panda_gripper.CoffeeSetupMug"
    # "single_panda_gripper.OpenDoubleDoor"
    # "single_panda_gripper.OpenDrawer"
    # "single_panda_gripper.OpenSingleDoor"
    "single_panda_gripper.PnPCabToCounter"
    "single_panda_gripper.PnPCounterToCab"
    "single_panda_gripper.PnPCounterToMicrowave"
    "single_panda_gripper.PnPCounterToSink"
    "single_panda_gripper.PnPCounterToStove"
    # "single_panda_gripper.PnPMicrowaveToCounter"
    # "single_panda_gripper.PnPSinkToCounter"
    # "single_panda_gripper.PnPStoveToCounter"
    # "single_panda_gripper.TurnOffMicrowave"
    # "single_panda_gripper.TurnOffSinkFaucet"
    # "single_panda_gripper.TurnOffStove"
    # "single_panda_gripper.TurnOnMicrowave"
    # "single_panda_gripper.TurnOnSinkFaucet"
    # "single_panda_gripper.TurnOnStove"
    # "single_panda_gripper.TurnSinkSpout"
)

TOTAL=${#TASKS[@]}
echo "Starting download of $TOTAL tasks with $N parallel processes..."
echo "Logs: $LOG_DIR"

download_task() {
    local TASK="$1"
    local LOG="$LOG_DIR/${TASK}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $TASK" | tee "$LOG"
    hf download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
        --repo-type dataset \
        --include "$TASK/*" \
        --local-dir "$LOCAL_DIR" >> "$LOG" 2>&1
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $TASK" | tee -a "$LOG"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $TASK" | tee -a "$LOG"
    fi
}

export -f download_task
export LOCAL_DIR LOG_DIR

printf '%s\n' "${TASKS[@]}" | xargs -P "$N" -I {} bash -c 'download_task "$@"' _ {}

echo ""
echo "All downloads completed. Check logs in $LOG_DIR"
