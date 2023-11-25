#include "board.h" 
#include <iostream> 
#include <iomanip> 

using namespace std;

struct Node{
    int visited;
    int wins;
    int sims;
    Node* parent;
    vector<Node*> children;
    Board board;
    int player;
    Position move;
}; 
class MonteCarloTree{
    private:
    Node* root;
    public:
    MonteCarloTree(Board board);
    void print_tree();
    void print_tree(Node* node, int depth);
    void find_move(int sim); // can possibly be done on the GPU probably
    Node create_node(Node* parent, Board board);
    Node* get_root();
    void set_root(Node* node);
    void delete_tree();
    Node run();
    vector<Node*> get_children(Node* node);
    Node* get_parent(Node* node);


    private:
    void simulate(Node* node); // These can be done on the GPU
    void backpropagate(Node* node, int winner); // These can be done on the GPU
    void delete_tree(Node* node);


}