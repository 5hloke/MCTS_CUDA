#include <vector>

class Board {
public:
    static const int BOARD_SIZE = 16;
    static const int WINNING_LENGTH = 5;

    enum Token {
        EMPTY,
        BLACK,
        WHITE
    };

    Board();
    Board(const Board& other) = default;
    bool valid_move(int row, int col) const;

    bool make_move(int row, int col, Token player);
    bool has_winner();

    Token get_winner() const ;

    std::vector<std::pair<int, int>> get_valid_moves();

    Token get_Token(int row, int col);

private:
    std::vector<std::vector<Token>> m_board;

    Token check_winner(int row, int col, int row_delta, int col_delta) const;
};
