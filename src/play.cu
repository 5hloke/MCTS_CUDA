#include "../include/MCTS_tree.h"
#include "../include/board.h"

int main()
{
    Board test_board;
    Position test_move;
    test_move.row = 0;
    test_move.col = 0;

    MonteCarloTree test_tree = MonteCarloTree(test_board, 1, test_move);
    test_tree.root->board.print_board();

    // test_move = test_tree.simulate(test_tree->root);
    return 0;
}