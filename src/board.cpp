#include "board.h"

using namespace std;

__global__ void valid_moves_kernel(const Board::Token *device_board,
                                   int board_size,
                                   int winning_length,
                                   std::pair<int, int> *valid_moves,
                                   int *valid_moves_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < board_size * board_size)
    {
        int row = tid / board_size;
        int col = tid % board_size;

        if (device_board[row * board_size + col] == Board::Token::EMPTY && row >= 0 && row < board_size && col >= 0 && col < board_size)
        {

            int index = atomicAdd(valid_moves_count, 1);
            valid_moves[index] = std::make_pair(row, col);
        }
    }
}

Board::Board() : m_board(BOARD_SIZE, std::vector<Token>(BOARD_SIZE, Token::EMPTY))
{
}

Board::Board(const Board &other) : m_board(other.m_board) {} // this copy constructor is not correct

bool Board::valid_move(int row, int col) const
{
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE && m_board[row][col] == Token::EMPTY;
}

bool Board::make_move(int row, int col, Token player)
{
    if (!valid_move(row, col))
    {
        return false;
    }
    m_board[row][col] = player;
    return true;
}

bool Board::has_winner() const
{
    return get_winner() != Token::EMPTY;
}

// CUDA kernel for get_winner needs to be written over here

/*
{ //This needs to be more efficeient can be done on CUDA
        // Check rows
        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col <= BOARD_SIZE - WINNING_LENGTH; ++col) {
                Token winner = check_winner(row, col, 0, 1);
                if (winner != Token::EMPTY) {
                    return winner;
                }
            }
        }

        // Check columns
        for (int row = 0; row <= BOARD_SIZE - WINNING_LENGTH; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                Token winner = check_winner(row, col, 1, 0);
                if (winner != Token::EMPTY) {
                    return winner;
                }
            }
        }

        // Check diagonals
        for (int row = 0; row <= BOARD_SIZE - WINNING_LENGTH; ++row) {
            for (int col = 0; col <= BOARD_SIZE - WINNING_LENGTH; ++col) {
                Token winner = check_winner(row, col, 1, 1);
                if (winner != Token::EMPTY) {
                    return winner;
                }
            }
        }
        for (int row = WINNING_LENGTH - 1; row < BOARD_SIZE; ++row) {
            for (int col = 0; col <= BOARD_SIZE - WINNING_LENGTH; ++col) {
                Token winner = check_winner(row, col, -1, 1);
                if (winner != Token::EMPTY) {
                    return winner;
                }
            }
        }

        return Token::EMPTY;
    }
*/

std::vector<std::pair<int, int>> Board::get_valid_moves_cuda() const
{
    // Copy the board to the device
    int board_size = Board::BOARD_SIZE;
    std::vector<std::vector<Token>> *device_board;
    cudaMalloc(&device_board, board_size * board_size * sizeof(Board::Token));
    cudaMemcpy(device_board, m_board.data(), board_size * board_size * sizeof(Board::Token), cudaMemcpyHostToDevice);

    // Allocate memory for valid moves on the device
    std::pair<int, int> *device_valid_moves;
    cudaMalloc(&device_valid_moves, board_size * board_size * sizeof(std::pair<int, int>));

    // Initialize valid_moves_count on the host and copy to the device
    int valid_moves_count = 0;
    int *device_valid_moves_count;
    cudaMalloc(&device_valid_moves_count, sizeof(int));
    cudaMemcpy(device_valid_moves_count, &valid_moves_count, sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int blockSize = 256; // 16 x 16
    int numBlocks = (board_size * board_size) / blockSize + 1;

    valid_moves_kernel<<<numBlocks, blockSize>>>(device_board, board_size, WINNING_LENGTH, device_valid_moves, device_valid_moves_count);

    // Copy the result back to the host
    cudaMemcpy(&valid_moves_count, device_valid_moves_count, sizeof(int), cudaMemcpyDeviceToHost);
    std::pair<int, int> *host_valid_moves = new std::pair<int, int>[valid_moves_count];
    cudaMemcpy(host_valid_moves, device_valid_moves, valid_moves_count * sizeof(std::pair<int, int>), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_board);
    cudaFree(device_valid_moves);
    cudaFree(device_valid_moves_count);

    // Convert the result to std::vector
    std::vector<std::pair<int, int>> result(host_valid_moves, host_valid_moves + valid_moves_count);
    delete[] host_valid_moves;

    return result;
}

// GPU implementation of get valid moves needs to be written over here
/*
Board::valid_moves() const { // This can be done on CUDA as well
        std::vector<std::pair<int, int>> valid_moves;
        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                if (valid_move(row, col)) {
                    valid_moves.emplace_back(row, col);
                }
            }
        }
        return valid_moves;
    }
*/

Board::Token Board::get_Token(int row, int col) const
{
    return m_board[row][col];
}

// CUDA kernel for check_winner needs to be written over here
/* // This can be done on CUDA as well
        Token first_Token = m_board[row][col];
        if (first_Token == Token::EMPTY) {
            return Token::EMPTY;
        }
        for (int i = 1; i < WINNING_LENGTH; ++i) {
            int r = row + i * row_delta;
            int c = col + i * col_delta;
            if (m_board[r][c] != first_Token) {
                return Token::EMPTY;
            }
        }
        return first_Token;
    }*/
