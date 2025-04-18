#!/bin/bash

# This script is used to run the RunPod API server and the RunPod CLI.
runpodctl config --apiKey $RUNPOD_API_KEY

# Verify installation
runpodctl version
