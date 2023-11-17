#include <vector>
// include Cuda libraries
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

    Board();
    Board(const Board &other) = default;
    bool valid_move(int row, int col) const;

    bool make_move(int row, int col, Token player);
    void move_to_cpu();
    void move_to_gpu();
    void clear_space();

    bool has_winner() const;

    Token get_winner() const;

    std::vector<Position> get_valid_moves();

    Token get_Token(int row, int col) const;
    std::vector<std::vector<Token>> get_board() const;

private:
    // Create a 2D dynamic array of size BOARD_SIZE x BOARD_SIZE
    std::vector< std::vector<Token>>m_board;
    Token *d_board;
};
