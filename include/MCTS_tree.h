#include "board.h"
#include <iostream>
#include <iomanip>

using namespace std;

struct Node
{
    int visited;
    int wins;
    int sims;
    int score;
    Node *parent;
    vector<Node *> children;
    Board board;
    int player;
    Position move;
};
class MonteCarloTree
{
private:
    Node *root;

public:
    MonteCarloTree(Board board, int player, Position move);
    void print_tree();
    void expand(Node *node);
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