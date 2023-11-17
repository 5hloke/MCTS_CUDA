#include "./../include/board.h"

int main()
{
    Board b1;
    b1.move_to_gpu();
    Token winner = b1.get_winner();
    std::cout << "Winner is: " << winner << std::endl;
    b1.make_move(5, 7, Token::BLACK);
    b1.make_move(0, 0, Token::WHITE);
    b1.make_move(5, 8, Token::BLACK);
    b1.make_move(0, 1, Token::WHITE);
    b1.make_move(5, 9, Token::BLACK);
    b1.make_move(0, 5, Token::WHITE);
    b1.make_move(5, 10, Token::BLACK);
    b1.make_move(0, 9, Token::WHITE);
    winner = b1.get_winner();
    std::cout << "Winner is: " << winner << std::endl;
    b1.make_move(5, 11, Token::BLACK);
    winner = b1.get_winner();
    std::cout << "Winner is: " << winner << std::endl;
}