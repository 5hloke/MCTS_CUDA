#include "board.h"
#include <iostream>
#include <iomanip>

using namespace std;



// Black win -1, White win 1, Draw 0 -> score
struct Node
{
    int visited;
    int wins;
    int sims;
    int score;
    Node *parent;
    Node* children = new Node[16*16];
    int num_children = 0;
    bool expanded = false;
    Board board;
    int player;
    Position move;
    void expand(){
        std::vector<Position> moves = parent->board.get_valid_moves();
        for (auto move : moves)
        {
            Node *child = new Node();
            child->board.update_board(board);
            if (player == 1)
            {
                child->board.make_move(move.row, move.col, 2);
                child->player = 2;
            }
            else
            {
                child->board.make_move(move.row, move.col, 1);
                child->player = 1;
            }
            child->parent = node;
            child->visited = 0;
            child->sims = 0;
            child->wins = 0;
            child->score = 0;
            child->move = move;
            child->children = new Node[16*16];
            child->num_children = 0;
            num_children += 1;

            expanded = true;
    }
    }
};
class MonteCarloTree
{
private:
    Node *root;

public:
    MonteCarloTree(Board board, int player, Position move);
    void print_tree();
    void print_tree(Node *node, int depth);
    void find_move(int sim); // can possibly be done on the GPU probably
    Node create_node(Node *parent, Board board);
    Node *get_root();
    void set_root(Node *node);
    void delete_tree();
    Node run();
    vector<Node *> get_children(Node *node);
    Node *get_parent(Node *node);
    void ~MonteCarloTree();

private:
    void simulate(Node *node);                 // These can be done on the GPU
    int backpropagate(Node *node, int winner); // These can be done on the GPU
    void delete_tree(Node *node);
    void print_node(Node *node);
    void print_node(Node *node, int depth);
}
