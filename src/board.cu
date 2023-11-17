#include "./../include/board.h"

Board::Board()
{
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        std::vector<Token> row;
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            row.push_back(Token::EMPTY);
        }
        m_board.push_back(row);
    }
}

// Board::Board(const Board &other) : m_board(other.m_board) {} // this copy constructor is not correct

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

void Board::move_to_gpu()
{
    // This function is not required for the assignment
    Token *dummy = new Token[BOARD_SIZE * BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; i < BOARD_SIZE; j++)
        {
            dummy[i * BOARD_SIZE + j] = m_board[i][j];
        }
    }
    cudaMalloc(&d_board, BOARD_SIZE * BOARD_SIZE * sizeof(Token)); // TODO :: Create a new variable for GPU . Do not use m_board. This has to be passed into the kernel
    cudaMemcpy(d_board, dummy, BOARD_SIZE * BOARD_SIZE * sizeof(Token), cudaMemcpyHostToDevice);
    delete[] dummy;
}
void Board::move_to_cpu()
{
    // This function is not required for the assignment
    Token *dummy = new Token[BOARD_SIZE * BOARD_SIZE];
    cudaMemcpy(dummy, d_board, BOARD_SIZE * BOARD_SIZE * sizeof(Token), cudaMemcpyDeviceToHost); // TODO :: Create a new variable for GPU . Do not use m_board. This has to be passed into the kernel
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; i < BOARD_SIZE; j++)
        {
            m_board[i][j] = dummy[i * BOARD_SIZE + j];
        }
    }

    // CUDA kernel for get_winner needs to be written over here
    /*void Board::clear_space()
    {
        // This function is not required for the assignment
        cudaFree(d_board);
    }*/

    // If theres no winner returns Token::EMPTY, if there is a winner return the player Token::BLACK/Token::WHITE
    Token Board::get_winner() const
    {
        Token winner = Token::EMPTY;
        dim3 block(8, 8);
        dim3 grid(BOARD_SIZE / block.x + 1, BOARD_SIZE / block.y + 1);
        Token *d_winner;
        cudaMalloc(&d_winner, sizeof(Token));
        check_winner_kernel<<<grid, block>>>(d_board, d_winner, BOARD_SIZE, WINNING_LENGTH);
        cudaMemcpy(&winner, d_winner, sizeof(Token), cudaMemcpyDeviceToHost);
        cudaFree(d_winner);
        return winner;
    }

    Token Board::get_Token(int row, int col) const
    {
        return m_board[row][col];
    }

    __global__ void check_winner_kernel(Token * board, Token * winner, int size, int win_len)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i > size || j > size)
        {
            return
        }
        Token player = board[i * size + j];
        if (player == Token::EMPTY)
        {
            *winner = Token::EMPTY;
            return;
        }
        Token *vertical_up[win_len - 1] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
        Token *vertical_down[win_len - 1] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
        Token *horizontal_left[win_len - 1] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
        Token *horizontal_right[win_len - 1] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
        Token *diag1[win_len - 1] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
        Token *diag2[win_len - 1] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
        Token *diag3[win_len - 1] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
        Token *diag4[win_len - 1] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};

        // Vertical Checks, horizontal checks, diagonals
        for (int k = 0; k < win_len - 1; k++)
        {
            if (i + k < size)
            {
                vertical_up[k] = board[(i + k) * size + j];
            }
            if (i - k > 0)
            {
                vertical_down[k] = board[(i - k) * size + j];
            }
            if (j - k > 0)
            {
                horizontal_left[k] = board[i * size + (j - k)];
            }
            if (j + k < size)
            {
                horizontal_right[k] = board[i * size + (j + k)];
            }
            if (i + k < size && j + k < size)
            {
                diag1[k] = board[(i + k) * size + (j + k)];
            }
            if (i - k > 0 && j - k > 0)
            {
                diag2[k] = board[(i - k) * size + (j - k)];
            }
            if (i - k > 0 && j + k < size)
            {
                diag3[k] = board[(i - k) * size + (j + k)];
            }
            if (i + k < size && j - k > 0)
            {
                diag4[k] = board[(i + k) * size + (j - k)];
            }
        }
        // Check for winner
        int up, down, left, right, diag1, diag2, diag3, diag4 = 1;
        for (int k = 0; k < win_len; k++)
        {
            if (vertical_up[k] != player && up != 0)
            {
                up = 0;
            }
            if (vertical_down[k] != player && down != 0)
            {
                down = 0;
            }
            if (horizontal_left[k] != player && left != 0)
            {
                left = 0;
            }
            if (horizontal_right[k] != player && right != 0)
            {
                right = 0;
            }
            if (diag1[k] != player && diag1 != 0)
            {
                diag1 = 0;
            }
            if (diag2[k] != player && diag2 != 0)
            {
                diag2 = 0;
            }
            if (diag3[k] != player && diag3 != 0)
            {
                diag3 = 0;
            }
            if (diag4[k] != player && diag4 != 0)
            {
                diag4 = 0;
            }
        }
        if (up == 1 || down == 1 || left == 1 || right == 1 || diag1 == 1 || diag2 == 1 || diag3 == 1 || diag4 == 1)
        {
            *winner = player;
            return;
        }
    }

    __global__ void valid_moves_kernel(Token * device_board,
                                       int board_size,
                                       int winning_length,
                                       Position *valid_moves,
                                       int *valid_moves_count)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i > board_size || j > board_size)
        {
            return
        }

        if (device_board[i * board_size + j] == Token::EMPTY)
        {

            int index = atomicAdd(valid_moves_count, 1);
            Position pos = {i, j};
            valid_moves[index] = pos;
        }
    }

    std::vector<Position> Board::get_valid_moves()
    {
        // Copy the board to the device
        int board_size = Board::BOARD_SIZE;

        // Allocate memory for valid moves on the device
        Position *device_valid_moves;
        cudaMalloc(&device_valid_moves, board_size * board_size * sizeof(Position));

        // Initialize valid_moves_count on the host and copy to the device
        int valid_moves_count = 0;
        int *device_valid_moves_count;
        cudaMalloc(&device_valid_moves_count, sizeof(int));
        cudaMemcpy(device_valid_moves_count, &valid_moves_count, sizeof(int), cudaMemcpyHostToDevice);

        dim3 block(8, 8);
        dim3 grid(BOARD_SIZE / block.x + 1, BOARD_SIZE / block.y + 1);
        valid_moves_kernel<<<grid, block>>>(d_board, board_size, WINNING_LENGTH, device_valid_moves, device_valid_moves_count);

        // Copy the result back to the host
        cudaMemcpy(&valid_moves_count, device_valid_moves_count, sizeof(int), cudaMemcpyDeviceToHost);
        Position *host_valid_moves = new Position[valid_moves_count];
        cudaMemcpy(host_valid_moves, device_valid_moves, valid_moves_count * sizeof(Position), cudaMemcpyDeviceToHost);

        // Free device memory
        // clear_space();
        cudaFree(device_valid_moves);
        cudaFree(device_valid_moves_count);

        // Convert the result to std::vector
        std::vector<Position> result(host_valid_moves, host_valid_moves + valid_moves_count);
        delete[] host_valid_moves;

        return result;
    }