#include <vector>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>

#ifndef BOARD_H
#define BOARD_H

struct Position
{
    int row;
    int col;
};
enum Token
{
    EMPTY,
    BLACK,
    WHITE
};

class Board
{
public:
    static const int BOARD_SIZE = 16;
    static const int WINNING_LENGTH = 5;
    int on_gpu = 0;
    int num_valid_moves = -1;
    // __device__ int num_valid_moves_d;
    Token winner = Token::EMPTY;

    __host__ __device__ Board();
    // Board(const Board &other) = default;
    __host__ __device__ void update_board(Board &other);
    __host__ __device__ bool valid_move(int row, int col) const;

    __host__ __device__ bool make_move(int row, int col, Token player);
    void move_to_cpu();
    void move_to_gpu();
    // void clear_space();

    // Returns False if there is no winner
    __host__ bool has_winner_host();
    __device__ bool has_winner_device();

    // Returns EMPTY if there is no winner
    __host__ Token get_winner_host();
    __device__ Token get_winner_device();

    __host__ __device__ bool is_draw();

    __host__ Position *get_valid_moves_host(int &num_moves);
    __device__ Position *get_valid_moves_device(int & num_moves);

    Token get_Token(int row, int col) const;
    std::vector<std::vector<Token>> get_board() const;

    __host__ __device__ void print_board();

private:
    // Create a 2D dynamic array of size BOARD_SIZE x BOARD_SIZE
    Token m_board[16][16];
    Token *d_board;
    __host__ __device__ void set_device_board();
};

#endif // BOARD_H