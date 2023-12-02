#include "../include/MCTS_tree.h"
#include <curand_kernel.h>
#include <queue>

__global__ void simulatekernel(Node *children, long long rate, int num_children)
// __global__ void simulatekernel(long long rate, int num_children)
{
    // printf("indx: %d", threadIdx.x);
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = threadIdx.x;

    if (i < num_children)
    {
        curandState_t state;
    
        // Node *parent = new Node();
        Node* parent = &children[i];
        curand_init(587, i, 0, &state);
        long long int start = clock64();
        long long int end = start;
        double elapsedTime = static_cast<double>(end - start) / rate;
        //  printf("rate: %d \n", rate);
    
        printf("Hmm: %d, %d \n", i, num_children);
        // }
        __syncthreads();
    

        while (elapsedTime < 10000)
        {
            // printf("Here ?\n");
            // if (i == 0){
            printf("Simulating: %d \n", i);
            // __syncthreads();
            // if(i == 0){
            //    parent->board.print_board(); 
            // }
            // __syncthreads();
            // if (i == 2){
            //     parent->board.print_board();
            // }
            // }
            // if (!parent->expanded)
            // {
                // printf("Here 1: %d, %d?\n", i, num_children);
                // if (i == 0){
                // printf("expansion 1: %d \n", i);
                // }
            // printf("before expansion\n");
            // parent->board.print_board();
            __syncthreads();
            Position* moves = parent->expand_device(i);
            // cudaThreadSynchronize();
            // printf("After expansion\n");
            // parent->board.print_board();
                // return;
                // printf("Here 2?\n");
                // if (i == 0){
                // printf("expansion 2 \n");
                // }
            // }
            // return;
            // pick a random number between 0 and 1
            __syncthreads();
            double random = curand_uniform(&state);
            printf("Number of Children: %d, %d \n", parent->num_children, i);

            int chosen = static_cast<int>(random * parent->num_children);
            printf("Chosen rand %d\n", chosen);

            // if (i == 0)
            // {
            //     printf("Child chosen \n");
            // }

            Position child_move = moves[chosen];
            // printf("Before deleting \n");
            // parent->board.print_board();
            delete [] moves;
            // printf("After deleeting \n");
            // parent->board.print_board();
            Node* child = new Node();
            // printf("creating New child \n");
            // parent->board.print_board();

            child->board.update_board(parent->board);
            __syncthreads();
            if (parent->player == Token::BLACK)
            {
                child->board.make_move(child_move.row, child_move.col, Token::WHITE);
                child->player = Token::WHITE;
            }
            else
            {
                child->board.make_move(child_move.row, child_move.col, Token::BLACK);
                child->player = Token::BLACK;
            }
            child->parent = parent;
            child->visited = 0;
            child->sims = 0;
            child->wins = 0;
            child->score = 0;
            child->move = child_move;
            child->num_children = 0;
            // stack[depth] = child;
            // depth++;
            // delete [] parent->children;
            // return;
            child->visited++;

            child->sims++;
            printf("The chosen one %d, %d \n", child->visited, child->sims);

            // Highly unoptimized - multiple calls to get-valid_moves
            return;
            if (child->board.has_winner_device())
            {
                // cudaThreadSynchronize();
                Token won = child->board.winner;
                parent = child;
                while (parent != &children[i])
                {
                    if (won == Token::BLACK)
                    {
                        parent->score -= 5;
                    }
                    else
                    {
                        parent->score += 5;
                    }
                    parent = parent->parent;
                }
                if (won == Token::BLACK)
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
                while (parent != &children[i])
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
            else{
                printf("Switching child to: %d, %d \n", child->move.row, child->move.col);
                parent = child;
            }
            child = nullptr;
            delete child;
            // return;
            // printf("end_board \n");
            // parent->board.print_board();
            printf("Clock\n\n\n\n\n");
            end = clock64();
            elapsedTime = static_cast<double>(end - start) / rate;
        }
        printf("Outside the while");
    }
    else
    {
        return;
    }
    // }
    // else
    // {
    //     return;
    // }
}

MonteCarloTree::MonteCarloTree(Board board, Token player, Position move)
{
    root = new Node();
    root->children = new Node[16 * 16];
    root->board = board;
    root->board.make_move(move.row, move.col, player);
    root->parent = nullptr;
    root->player = player;
    root->visited = 0;
    root->sims = 0;
    root->wins = 0;
    root->score = 0;
    root->move = move;
    // root->expand();
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
    // print_node(root);
}
/*
void MonteCarloTree::print_node(Node *node)
{
    std::cout << "Move made - row: " << node->move.row << ", col: " << node->move.col << std::endl;
    for (int i = 0; i < sizeof(node->children) / sizeof(node->children[0]); i++)
    {
        print_node(node->children[i]);
    }
}

void MonteCarloTree::print_node(Node *node, int depth)
{
    if (depth < 0)
        return;
    std::cout << "Move made - row: " << node->move.row << ", col: " << node->move.col << std::endl;
    for (int i = 0; i < sizeof(node->children) / sizeof(node->children[0]); i++)
    {
        print_node(node->children[i], depth - 1);
    }
}
*/
void MonteCarloTree::set_root(Node *node)
{
    root = node;
}

Node *MonteCarloTree::get_parent(Node *node)
{
    /*
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
    */
    return nullptr;
}

Position MonteCarloTree::simulate(Node *node)
{
    node->board.move_to_gpu();
    node->expand_host();
    std::cout << "expanded " << node->num_children << std::endl;
    Node *childs;
    cudaMalloc(&childs, 16 * 16 * sizeof(Node));
    std::cout << "Move: " <<  node->children[(16*14)].move.row << ", " << node->children[(16*14) - 1].move.col <<std::endl;
    cudaMemcpy(childs, node->children, 16 * 16 * sizeof(Node), cudaMemcpyHostToDevice);

    dim3 block(8, 8);
    dim3 grid(16 / block.x + 1, 16 / block.y + 1);

    int temp_rate;

    cudaError_t cudaStat = cudaDeviceGetAttribute(&temp_rate, cudaDevAttrClockRate, 0);

    long long rate = (long long)temp_rate;

    std::cout << "Inside the kernel " << node->num_children << std::endl;
    // return node->move;
    // for (int index = 0; index < 3; index++){
    simulatekernel<<<1, 256>>>(childs, rate, node->num_children); // node->num_children);
    // }
    // simulatekernel<<<grid, block>>>(rate, node->num_children);
    // cudaError_t err = cudaGetLastError();
    // std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    // cudaDeviceSynchronize();
    std::cout << "Simulation Finished" << std::endl;
    return node->move;
    cudaDeviceSynchronize();
    std::cout << "Outside the kernel " << std::endl;

    cudaMemcpy(node->children, &childs, 16 * 16 * sizeof(Node), cudaMemcpyHostToDevice);
    node->board.move_to_cpu();
    cudaFree(childs);
    return node->move;
    Position move = node->children[0].move;
    int max_score = node->children[0].score;
    for (int i = 0; i < sizeof(node->children) / sizeof(node->children[0]); i++)
    {
        if (node->children[i].score > max_score)
        {
            max_score = node->children[i].score;
            move = node->children[i].move;
        }
    }
    return move;
    // Code to select best possibble action. If all equal should we randomize?
}