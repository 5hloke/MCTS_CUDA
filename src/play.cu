#include "../include/MCTS_tree.h"
#include "../include/board.h"

int main()
{
    Board test_board;
    Position test_move;
    test_move.row = 5;
    test_move.col = 5;

    MonteCarloTree test_tree = MonteCarloTree(test_board, Token::BLACK, test_move);
    int count = 0;
    // test_tree.root->board.move_to_gpu();
    // Position* vals = test_tree.root->board.get_valid_moves(count);
    // std::cout << "Count: " <<count << std::endl;
    // test_tree.root->board.print_board();

    test_move = test_tree.simulate(test_tree.root);
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << test_move.row << ", " << test_move.col << std::endl;
    return 0;
}