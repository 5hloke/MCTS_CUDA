#include "../include/MCTS_tree.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void simulatekernel(Node *children, long long rate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // int stride = blockDim.x * gridDim.x;
    curandState_t state;
    Node *parent = children[i];
    curant_init(587, i, 0, &state);
    long long int start = clock64();
    long long int end = start;
    double elapsedTime = static_cast<double>(end - start) / rate;
    while (elapseTime < 1000)
    {
        if (!parent->expanded)
        {
            parent->expand();
        }
        // pick a random number between 0 and 1
        double rand = curand_uniform(&state);
        int chosen = static_cast<int>(rand * parent->num_children());
        Node *child = parent->children[chosen];
        child->visited++;
        child->sims++;

        if (!child->expanded)
        {
            child->expand();
        }
        // Highly unoptimized - multiple calls to get-valid_moves
        if (child->board.has_winner())
        {

            Token won = child->board.get_winner();
            parent = child;
            while (parent != children[i])
            {
                if (won == Token::Black)
                {
                    parent->score -= 5;
                }
                else
                {
                    parent->score += 5;
                }
                parent = parent->parent;
            }
            if (won == Token::Black)
            {
                parent->score -= 5;
            }
            else
            {
                parent->score += 5;
            }
        }
        else if (child->board.is_draw())
        {
            int player = child->player;
            parent = child;
            while (parent != children[i])
            {
                if (player == 1)
                {
                    parent->score -= 2;
                }
                else
                {
                    parent->score += 2;
                }
                parent = parent->parent;
                player = parent->player;
            }
            if (player == 1)
            {
                parent->score -= 2;
            }
            else
            {
                parent->score += 2;
            }
        }

        end += clock64();
    }
}
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
    root->expand();
}

MonteCarloTree::~MonteCarloTree()
{
    delete root;
}

// void expand(Node *node)
// {

// }

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

Position MonteCarloTree::simulate(Node *node)
{
    node->expand();
    Node *childs;
    cudaMalloc(&childs, 16 * 16 * sizeof(Node));
    cudaMemcpy(childs, &node->children, 16 * 16 * sizeof(Node), cudaMemcpyHostToDevice);

    dim3 block(8, 8);
    dim3 grid(BOARD_SIZE / block.x + 1, BOARD_SIZE / block.y + 1);

    long long rate;

    cudaError_t cudaStat = cudaDeviceGetAttribute(&rate, cudaDevAttrClockRate, 0);

    simulatekernel << grid, block >> (childs, rate);

    cudaMemcpy(node->children, &childs, 16 * 16 * sizeof(Node), cudaMemcpyHostToDevice);
    cudaFree(childs);
    Position move = node->children[0].move;
    int max_score = node->children[0].score;
    for (Node *child : children)
    {
        if (child->score > max_score)
        {
            max_score = child->score;
            move = child->move;
        }
    }

    return move;
    // Code to select best possibble action. If all equal should we randomize?
}