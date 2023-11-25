#include "../include/MCTS_tree.h"


MonteCarloTreeSearch::MonteCarloTreeSearch(Board board, int player, Position move) {
    root = new Node();
    root->board = board;
    root->parent = nullptr;
    root->player = player;
    root->visited = 0;
    root->sims = 0;
    root->wins = 0;
    root->score = 0;
    root->move = move;
    this->expand(root);
}

MonteCarloTreeSearch::~MonteCarloTreeSearch() {
    delete root;
}

void MonteCarloTreeSearch::expand(Node *node) {
    std::vector<Position> moves = node->board.get_valid_moves();
    for (auto move : moves) {
        Node *child = new Node();
        child->board.update_board(node->board);
        if (node->player == 1){
            child->board.make_move(move.row, move.col, 2);
            child->player = 2;
        }
        else {
            child->board.make_move(move.row, move.col, 1);
            child->player = 1;
        }
        child->parent = node;
        child->visited = 0;
        child->sims = 0;
        child->wins = 0;
        child->score = 0;
        child->move = move;
        node->children.push_back(child);
    }
}

