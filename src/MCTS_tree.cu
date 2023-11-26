#include "../include/MCTS_tree.h"

MonteCarloTree::MonteCarloTree(Board board, int player, Position move)
{
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

MonteCarloTree::~MonteCarloTree()
{
    delete root;
}

void MonteCarloTree::expand(Node *node)
{
    std::vector<Position> moves = node->board.get_valid_moves();
    for (auto move : moves)
    {
        Node *child = new Node();
        child->board.update_board(node->board);
        if (node->player == 1)
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
        child->children = new vector<Node *>();
        node->children.push_back(child);
    }
}

void MonteCarloTree::print_tree()
{
    print_node(root);
}

void MonteCarloTree::print_node(Node *node)
{
    std::cout << "Move made - row: " << node->move.row << ", col: " << node->move.col << std::endl;
    for (auto child : node->children)
    {
        print_node(child);
    }
}

void MonteCarloTree::print_node(Node *node, int depth)
{
    if (depth < 0)
        return;
    std::cout << "Move made - row: " << node->move.row << ", col: " << node->move.col << std::endl;
    for (auto child : node->children)
    {
        print_node(child, depth - 1);
    }
}

Node *MonteCarloTree::getRoot()
{
    return root;
}

void MonteCarloTree::set_root(Node *node)
{
    root = node;
}

vector<Node *> MonteCarloTree::get_parent(Node *node)
{
    std::queue<Node *> q;
    q.push(root);

    while (!q.empty())
    {
        Node *current = q.front();
        q.pop();
        for (Node *child : current->children)
        {
            if (child == node)
            {
                return current;
            }
            q.push(child);
        }
    }
    return nullptr;
}