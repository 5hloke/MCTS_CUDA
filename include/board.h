#include <vector>
#include <iomanip>
#include <iostream>

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

    __host__ __device__ Board();
    // Board(const Board &other) = default;
    __host__ __device__ void update_board(Board &other);
    bool valid_move(int row, int col) const;

    __host__ __device__ bool make_move(int row, int col, Token player);
    void move_to_cpu();
    void move_to_gpu();
    // void clear_space();

    // Returns False if there is no winner
    __host__ __device__ bool has_winner() const;

    // Returns EMPTY if there is no winner
    __host__ __device__ Token get_winner() const;

    __host__ __device__ bool is_draw() const;

    __host__ __device__ Position *get_valid_moves(int &num_moves);

    Token get_Token(int row, int col) const;
    std::vector<std::vector<Token>> get_board() const;

    void print_board();

private:
    // Create a 2D dynamic array of size BOARD_SIZE x BOARD_SIZE
    Token m_board[16][16];
    Token *d_board;
};

#endif // BOARD_H