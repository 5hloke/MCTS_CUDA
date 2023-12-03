#include "../include/MCTS_tree.h"
#include "../include/board.h"


int main()
{
    Board test_board;
    Position test_move;
    test_move.row = 3;
    test_move.col = 2;
    test_board.make_move(5,7,Token::BLACK);
    test_board.make_move(1,1,Token::WHITE);
    

    test_board.make_move(5,6,Token::BLACK);
    test_board.make_move(3,4,Token::WHITE);

    test_board.make_move(5,5,Token::BLACK);
    test_board.make_move(1,3,Token::WHITE);

    test_board.make_move(5,4,Token::BLACK);
    //test_board.make_move(3,2,Token::WHITE);

    MonteCarloTree test_tree = MonteCarloTree(test_board, Token::WHITE, test_move);
    // int count = 0;
    // test_tree.root->board.move_to_gpu();
    // Position* vals = test_tree.root->board.get_valid_moves(count);
    // std::cout << "Count: " <<count << std::endl;
    // test_tree.root->board.print_board();
    Node* pos;
    pos = test_tree.simulate(test_tree.root);
    std::cout << "Move: " << pos->move.row << "," << pos->move.col <<std::endl;
    std::cout << std::endl;
    // if (test_tree.root->player == Token::BLACK){
    //    test_tree.root->player = Token::WHITE; 
    // }
    // else{
    //     test_tree.root->player = Token::BLACK; 
    // }
    // test_tree.root->board.make_move(test_move.row, test_move.col, test_tree.root->player);
    test_tree.root = pos;
    test_tree.root->board.print_board();
    /*
    test_tree.root->num_children = 0;
    pos = test_tree.simulate(test_tree.root);
    pos->board.print_board();
    */
    // std::cout << "Move: " << test_move.row << "," << test_move.col <<std::endl;
    // std::cout << std::endl;
    // if (test_tree.root->player == Token::BLACK){
    //    test_tree.root->player = Token::WHITE; 
    // }
    // else{
    //     test_tree.root->player = Token::BLACK; 
    // }
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
    for (int i = 0; i < 9; i++){
        std::cout << "Num Sims: " << i << std::endl;
        Node* pos;
        pos = test_tree.simulate(test_tree.root);
        // std::cout << "Move: " << test_move.row << "," << test_move.col <<std::endl;
        std::cout << std::endl;
        // if (test_tree.root->player == Token::BLACK){
        //    test_tree.root->player = Token::WHITE; 
        // }
        // else{
        //     test_tree.root->player = Token::BLACK; 
        // }
        // test_tree.root->board.make_move(test_move.row, test_move.col, test_tree.root->player);
        test_tree.root = pos;
        test_tree.root->board.print_board();
        test_tree.root->num_children = 0;
        pos = test_tree.simulate(test_tree.root);
        pos->board.print_board();
    }
    // std::cout << "Move: " << test_move.row << "," << test_move.col <<std::endl;
    // std::cout << std::endl;
    // if (test_tree.root->player == Token::BLACK){
    //    test_tree.root->player = Token::WHITE; 
    // }
    // else{
    //     test_tree.root->player = Token::BLACK; 
    // }
    std::cout << "Simulation finished: "  << test_tree.totalsims << std::endl;
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