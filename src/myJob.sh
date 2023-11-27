#!/bin/bash
#SBATCH --job-name=MCTS
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f23_class
#SBATCH --partition=gpu




#make sure to load the cuda module before running
#module load cuda
#make sure to compile your program using nvcc
#nvcc -o example1 example1.cu
#nvcc -arch=sm_70 test_board.cu board.cu -o ./mcts.out
./mcts.out > mcts.txt