#include "board.h" 
#include <iostream> 
#include <iomanip> 

using namespace std;

 
class MonteCarloTree{
    private:
    struct Node{
        int visited;
        int wins;
        Node* parent;
        vector<Node*> children;
        Board board;
    };

    Node* root;
    public:
    MonteCarloTree(Board board);
    void print_tree();
    void print_tree(Node* node, int depth);
    void find_move(int sim); // can possibly be done on the GPU probably

    private:
    void simulate(Node* node); // These can be done on the GPU
    void backpropagate(Node* node, int winner); // These can be done on the GPU
    void delete_tree(Node* node);


}