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
    Node *children;
    // Node *simmed_chldren;
    int num_children = 0;
    // bool expanded = false;
    Board board;
    Token player;
    Position move;
    __host__ void expand_host()
    {
        // printf("Inside Expand\n");
        int num_moves = 0;
        Position *moves = board.get_valid_moves_host(num_moves);
        // printf("Running ? %d \n", num_moves);
        for (int i = 0; i < num_moves; i++)
        {
            Position move = moves[i];
            Node child;
            child.board.update_board(board);
            if (player == Token::BLACK)
            {
                child.board.make_move(move.row, move.col, Token::WHITE);
                child.player = Token::WHITE;
            }
            else
            {
                child.board.make_move(move.row, move.col, Token::BLACK);
                child.player = Token::BLACK;
            }
            child.parent = this;
            child.visited = 0;
            child.sims = 0;
            child.wins = 0;
            child.score = 0;
            child.move = move;
            // printf("In expand: %d, %d ", child.move.row, child.move.col);

            child.children = new Node[16 * 16];
            child.num_children = 0;
            this->children[num_children] = child;
            this->num_children += 1;
            // this->expanded = true;
        }
        delete [] moves;
    }

    __device__ Position * expand_device()
    {
        int num_moves = 0;
        // printf("getting valid moves\n");
        // return;
        Position *moves = board.get_valid_moves_device(num_moves);
        __syncthreads();
        // printf("Got valid moves ? %d \n", num_moves);
        this->num_children = num_moves;
        return moves;
        // this->children = new Node[16 * 16];
        // for (int i = 0; i < num_moves; i++)
        // {
        //     // printf("Iterating %d\n", i);
        //     Position move = moves[i];
        //     Node child;
        //     child.board.update_board(board);

        //     if (player == Token::BLACK)
        //     {
        //         child.board.make_move(move.row, move.col, Token::WHITE);
        //         child.player = Token::WHITE;
        //     }
        //     else
        //     {
        //         child.board.make_move(move.row, move.col, Token::BLACK);
        //         child.player = Token::BLACK;
        //     }
        //     child.parent = this;
        //     child.visited = 0;
        //     child.sims = 0;
        //     child.wins = 0;
        //     child.score = 0;
        //     child.move = move;
        //     // printf("In expand: %d, %d ", child.move.row, child.move.col);
        //     child.num_children = 0;
        //     this->children[num_children] = child;
        //     this->num_children += 1;
        //     this->expanded = true;
        // }
    }
};
class MonteCarloTree
{
public:
    Node *root;
    MonteCarloTree(Board board, Token player, Position move);
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
    ~MonteCarloTree();
    Position simulate(Node *node); // These can be done on the GPU

private:
    int backpropagate(Node *node, int winner); // These can be done on the GPU
    void delete_tree(Node *node);
    // void print_node(Node *node);
    // void print_node(Node *node, int depth);
};
