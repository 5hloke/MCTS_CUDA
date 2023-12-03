#include "../include/MCTS_tree.h"
#include <curand_kernel.h>
#include <queue>

__global__ void simulatekernel(Node *children, long long rate, int num_children, int batch)
// __global__ void simulatekernel(long long rate, int num_children)
{
    // printf("indx: %d", threadIdx.x);
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = batch * 16 + threadIdx.x;
    
    if (i < num_children)
    {
        curandState_t state;

        // Node *parent = new Node();
        Node *parent = &children[i];
        curand_init(700, i, 0, &state);
        long long int start = clock64();
        long long int end = start;
        double elapsedTime = static_cast<double>(end - start) / rate;
        //  printf("rate: %d \n", rate);

        // printf("Hmm: %d, %d \n", i, num_children);
        // }
        __syncthreads();

        while (elapsedTime < 9000)
        {
            // cudaDeviceSynchronize();
            //  printf("Here ?\n");
            //  if (i == 0){
            //  printf("Simulating: %d \n", i);
            //  __syncthreads();
            //  if(i == 0){
            //     parent->board.print_board();
            //  }
            //  __syncthreads();
            //  if (i == 2){
            //      parent->board.print_board();
            //  }
            //  }
            //  if (!parent->expanded)
            //  {
            //  printf("Here 1: %d, %d?\n", i, num_children);
            //  if (i == 0){
            //  printf("expansion 1: %d \n", i);
            //  }
            //  printf("before expansion\n");
            //  parent->board.print_board();
            __syncthreads();
            Position *moves = parent->expand_device(i);
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
            // printf("Number of Children: %d, %d \n", parent->num_children, i);

            int chosen = static_cast<int>(random * parent->num_children);
            // printf("Chosen rand %d\n", chosen);

            // if (i == 0)
            // {
            //     printf("Child chosen \n");
            // }

            Position child_move = moves[chosen];
            // printf("Before deleting \n");
            // parent->board.print_board();
            delete[] moves;
            // printf("After deleeting \n");
            // parent->board.print_board();
            Node *child = new Node();
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
            // printf("The chosen one %d, %d \n", child->visited, child->sims);

            // Highly unoptimized - multiple calls to get-valid_moves

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
                parent->sims += 1;
            }
            else if (child->board.is_draw())
            {
                // printf("Backtracking draw\n");
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
                parent->sims += 1;
            }
            else
            {
                // printf("Switching child to: %d, %d \n", child->move.row, child->move.col);
                parent = child;
            }

            // parent=child;
            // return;
            // child = nullptr;
            // delete child;
            //  return;
            //  printf("end_board \n");
            //  parent->board.print_board();
            //printf("Clock\n\n\n\n\n");
            end = clock64();
            elapsedTime = static_cast<double>(end - start) / rate;
        }
        __syncthreads();
        //printf("score(%d): %d\n", i, children[i].score);

        printf("Outside the while\n");
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
    root->children = new Node[8 * 8];
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

Node *MonteCarloTree::simulate(Node *node)
{
    node->board.move_to_gpu();
    node->expand_host();
    std::cout << "expanded " << node->num_children << std::endl;
    for (int i = 0; i < 4; i++)
    {
        node->board.move_to_gpu();
        Node *childs;
        cudaMalloc(&childs, 8 * 8 * sizeof(Node));
        std::cout << "Move: " << node->children[(8 * 6)].move.row << ", " << node->children[(8 * 6) - 1].move.col << std::endl;
        cudaMemcpy(childs, node->children, 8 * 8 * sizeof(Node), cudaMemcpyHostToDevice);

        dim3 block(8, 8);
        dim3 grid(8 / block.x + 1, 8 / block.y + 1);

        int temp_rate;

        cudaError_t cudaStat = cudaDeviceGetAttribute(&temp_rate, cudaDevAttrClockRate, 0);

        long long rate = (long long)temp_rate;

        std::cout << "Inside the kernel " << node->num_children << std::endl;
        // return node->move;
        // for (int index = 0; index < 3; index++){
        simulatekernel<<<1, 16>>>(childs, rate, node->num_children, i); // node->num_children);
        // }
        // simulatekernel<<<grid, block>>>(rate, node->num_children);
        // cudaError_t err = cudaGetLastError();
        // std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
        // cudaDeviceSynchronize();
        // std::cout << "Simulation Finished" << std::endl;

        cudaDeviceSynchronize();
        std::cout << "Outside the kernel " << std::endl;
        cudaError_t err{cudaGetLastError()};

        if (err != cudaSuccess)

        {

            std::cout << "CUDA Runtime Error: " << cudaGetErrorString(err) << std::endl;
            // We don't exit when we encounter CUDA errors in this example.

            // std::exit(EXIT_FAILURE);
        }
        cudaMemcpy(node->children, childs, 8 * 8 * sizeof(Node), cudaMemcpyDeviceToHost);
        node->board.move_to_cpu();
        cudaFree(childs);
        cudaDeviceReset();
    }
    std::cout << "Cumming" << std::endl;
    // return node->move;
    Node *move = &node->children[0]; //.move;
    int max_score = -10000000;
    int min_score = 10000000;
    int selected = 0;
    for (int i = 0; i < node->num_children; i++)
    {
        std::cout << "Score " << i << ", " << selected << ": " << node->children[i].score << ", " << node->children[i].sims << std::endl;
        if (root->player == Token::BLACK)
        {
            int player_score = node->children[i].score * node->children[i].sims;
            if (player_score < min_score)
            {
                min_score = player_score;
                move = &node->children[i]; //.move;
                selected = i;
            }
        }
        if (root->player == Token::WHITE)
        {
            int player_score = node->children[i].score * node->children[i].sims;
            if (player_score > max_score)
            {
                min_score = player_score;
                move = &node->children[i]; //.move;
                selected = i;
            }
        }
    }
    return move;
    // for (int i = 0; i < 64; i++)
    // {
    // std::cout<<"Score for thread "<<i<<" is: "<<node->children[i].score<<std::endl;
    //     if (node->children[i].score > max_score)
    //     {
    //         max_score = node->children[i].score;
    //         move = node->children[i].move;
    //     }
    // }
    // return move;
    // Code to select best possibble action. If all equal should we randomize?
}
