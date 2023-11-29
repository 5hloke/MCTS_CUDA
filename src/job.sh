#!/bin/bash
#SBATCH --job-name=MCTS
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f23_class
#SBATCH --partition=gpu

compute-sanitizer --log-file sanitizer.txt ./play.out > mcts.txt
