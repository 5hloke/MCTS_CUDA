#include "../include/MCTS_tree.h"
#include "../include/board.h"


int main()
{
    Board test_board;
    Position test_move;
    test_move.row = 5;
    test_move.col = 5;

    MonteCarloTree test_tree = MonteCarloTree(test_board, Token::BLACK, test_move);
    // int count = 0;
    // test_tree.root->board.move_to_gpu();
    // Position* vals = test_tree.root->board.get_valid_moves(count);
    // std::cout << "Count: " <<count << std::endl;
    // test_tree.root->board.print_board();

    test_move = test_tree.simulate(test_tree.root);
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Simulation finished" << std::endl;
    return 0;
}


// __global__ void test_kernel(Node *test)
// {
//     Token winner = test[0].board.get_winner_device();
//     printf("Winner: %d", winner);
// }

// int main()
// {
//     Board b1;
//     b1.make_move(5, 7, Token::BLACK);
//     b1.make_move(0, 0, Token::WHITE);
//     b1.make_move(5, 8, Token::BLACK);
//     b1.make_move(0, 1, Token::WHITE);
//     b1.make_move(5, 9, Token::BLACK);
//     b1.make_move(0, 5, Token::WHITE);
//     b1.make_move(5, 10, Token::BLACK);
//     b1.make_move(0, 9, Token::WHITE);
//     b1.make_move(5, 11, Token::BLACK);
//     b1.print_board();
//     Node test;
//     test.board.update_board(b1);
//     Node *d_test;
//     cudaMalloc(&d_test, sizeof(Node));
//     cudaMemcpy(d_test, &test, sizeof(Node), cudaMemcpyHostToDevice);
//     dim3 block(1, 1);
//     dim3 grid(1, 1);
//     test_kernel<<<grid, block>>>(d_test);
//     return 0;
// }