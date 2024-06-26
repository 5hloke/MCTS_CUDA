#include "./../include/board.h"

int main()
{
    Board b1;

    std::vector<Position> moves = b1.get_valid_moves();
    for (int i = 0; i < moves.size(); i++)
    {
        std::cout << "MOVE:" << moves[i].row << " " << moves[i].col << ";";
    }
    std::cout << std::endl;
    b1.make_move(5, 7, Token::BLACK);
    b1.make_move(0, 0, Token::WHITE);
    b1.make_move(5, 8, Token::BLACK);
    b1.make_move(0, 1, Token::WHITE);
    b1.make_move(5, 9, Token::BLACK);
    b1.make_move(0, 5, Token::WHITE);
    b1.make_move(5, 10, Token::BLACK);
    b1.make_move(0, 9, Token::WHITE);
    b1.make_move(5, 11, Token::BLACK);
    b1.print_board();
    b1.move_to_gpu();
    int winner = b1.get_winner();
    moves = b1.get_valid_moves();
    for (int i = 0; i < moves.size(); i++)
    {
        std::cout << "MOVE:" << moves[i].row << " " << moves[i].col << ";";
    }
    std::cout << std::endl;

    std::cout << "Winner is: " << winner << std::endl;
}
