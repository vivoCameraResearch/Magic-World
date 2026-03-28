#!/bin/bash

echo "======================================================"
echo "Starting model download..."
echo "======================================================"

echo "Downloading Wan2.1-Fun-V1.1..."
huggingface-cli download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP --local-dir checkpoints/Wan2.1-Fun-V1.1-1.3B-InP

echo "Downloading MagicWorld..."
huggingface-cli download LuckyLiGY/MagicWorld --local-dir checkpoints/MagicWorld

echo "Downloading Reward Forcing..."
huggingface-cli download JaydenLu666/Reward-Forcing-T2V-1.3B --local-dir checkpoints/Reward-Forcing-T2V-1.3B

echo "======================================================"
echo "Finished downloading models!"
ls -R checkpoints
echo "======================================================"