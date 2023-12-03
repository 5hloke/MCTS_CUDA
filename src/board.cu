#include "./../include/board.h"
__global__ void check_winner_kernel(Token *board, Token *winner, int size, int win_len, int *count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // return;
   // printf("Checking Winner: %d ", *winner);
    if (i >= size || j >= size)
    {
        return;
    }
    //printf("Pos num: %d\n",i*size+j);
    Token player = board[i * size + j];

    if (player == Token::EMPTY)
    {
	atomicAdd(count,1);
        return;
    }
    // printf("i:%d,j:%d\n",i,j);
    const int n_len = win_len - 1;
    Token vertical_up[4] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
    Token vertical_down[4] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
    Token horizontal_left[4] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
    Token horizontal_right[4] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
    Token diag1[4] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
    Token diag2[4] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
    Token diag3[4] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
    Token diag4[4] = {Token::EMPTY, Token::EMPTY, Token::EMPTY, Token::EMPTY};
    // Vertical Checks, horizontal checks, diagonals

    for (int k = 0; k < n_len; k++)
    {
        if (i + (k + 1) < size)
        {
            vertical_up[k] = board[(i + (k + 1)) * size + j];
        }
        if (i - (k + 1) >= 0)
        {
            vertical_down[k] = board[(i - (k + 1)) * size + j];
        }
        if (j - (k + 1) >= 0)
        {
            horizontal_left[k] = board[i * size + (j - (k + 1))];
        }
        if (j + (k + 1) < size)
        {
            horizontal_right[k] = board[i * size + (j + (k + 1))];
        }
        if (i + (k + 1) < size && j + (k + 1) < size)
        {
            diag1[k] = board[(i + (k + 1)) * size + (j + (k + 1))];
        }
        if (i - (k + 1) >= 0 && j - (k + 1) >= 0)
        {
            diag2[k] = board[(i - (k + 1)) * size + (j - (k + 1))];
        }
        if (i - (k + 1) >= 0 && j + (k + 1) < size)
        {
            diag3[k] = board[(i - (k + 1)) * size + (j + (k + 1))];
        }
        if (i + (k + 1) < size && j - (k + 1) >= 0)
        {
            diag4[k] = board[(i + (k + 1)) * size + (j - (k + 1))];
        }
    }
    // Check for winne
    // return;
    int up = 1, down = 1, left = 1, right = 1, d1 = 1, d2 = 1, d3 = 1, d4 = 1;
    for (int k = 0; k < n_len; k++)
    {
        if (vertical_up[k] != player && up != 0)
        {
            up = 0;
        }
        if (vertical_down[k] != player && down != 0)
        {
            down = 0;
        }
        if (horizontal_left[k] != player && left != 0)
        {

            left = 0;
        }
        if (horizontal_right[k] != player && right != 0)
        {
            right = 0;
        }
        if (diag1[k] != player && d1 != 0)
        {
            d1 = 0;
        }
        if (diag2[k] != player && d2 != 0)
        {
            d2 = 0;
        }
        if (diag3[k] != player && d3 != 0)
        {
            d3 = 0;
        }
        if (diag4[k] != player && d4 != 0)
        {
            d4 = 0;
        }
    }
    
    if ((up == 1 || down == 1 || left == 1 || right == 1 || d1 == 1 || d2 == 1 || d3 == 1 || d4 == 1) && *winner == Token::EMPTY)
    {
        *winner = player;
        // printf("Winner is (inside kernel): %d %d %d %d %d %d %d\n\n", *winner, i, j, up, down, left, right);
        atomicAdd(count,1);
	return;
    }
    else
    {
	atomicAdd(count,1);
        return;
    }
    
}

__global__ void valid_moves_kernel(Token *device_board,
                                   int board_size,
                                   Position *valid_moves,
                                   int *valid_moves_count)
{
    // printf("Inside device count: %d, %d, %d, %d, %d, %d\n", blockDim.x, blockIdx.x, threadIdx.x, blockDim.y, blockIdx.y, threadIdx.y);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("Board position value: %d \n", device_board[i * board_size + j]);
    if (i >= board_size || j >= board_size)
    {
        // printf("Not getting valid moves %d, %d\n", i, j);
        return;
    }
    // printf("Board position value: %d \n", device_board[i * board_size + j]);
    if (device_board[i * board_size + j] == Token::EMPTY)
    {
        int index = atomicAdd(valid_moves_count, 1);
        Position pos = {i, j};
        // printf("index %d \n", index);
        valid_moves[index] = pos;
        // printf("Score: \n");
        return;
    }
}
__global__ void valid_moves_kernel_tail()
{
     // printf("Inside device count: %d, %d, %d, %d, %d, %d\n", blockDim.x, blockIdx.x, threadIdx.x, blockDim.y, blockIdx.y, threadIdx.y);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
//     // printf("Board position value: %d \n", device_board[i * board_size + j]);
//     // if (i >= board_size || j >= board_size)
//     // {
//     //     // printf("Not getting valid moves %d, %d\n", i, j);
//     //     return;
//     // }
//     // // printf("Board position value: %d \n", device_board[i * board_size + j]);
//     // if (device_board[i * board_size + j] == Token::EMPTY)
//     // {
//     //     int index = atomicAdd(valid_moves_count, 1);
//     //     Position pos = {i, j};
//     //     valid_moves[index] = pos;
//     //     // printf("Score: \n");
//     //     return;
//     // }
// }
}

__host__ __device__ Board::Board()
{
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            m_board[i][j] = Token::EMPTY;
        }
    }
}

// Board::Board(const Board &other) : m_board(other.m_board) {} // this copy constructor is not correct

__host__ __device__ bool Board::valid_move(int row, int col) const
{
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE && m_board[row][col] == Token::EMPTY;
}

__host__ __device__ void Board::update_board(Board &other)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            m_board[i][j] = other.m_board[i][j];
        }
    }
}

__host__ __device__ bool Board::make_move(int row, int col, Token player)
{
    if (!valid_move(row, col))
    {
        return false;
    }
    m_board[row][col] = player;
    return true;
}

__device__ bool Board::has_winner_device()
{
    return get_winner_device() != Token::EMPTY;
}

__host__ bool Board::has_winner_host()
{
    return get_winner_host() != Token::EMPTY;
}

void Board::move_to_gpu()
{
    // This function is not required for the assignment
    Token *dummy = new Token[BOARD_SIZE * BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            dummy[i * BOARD_SIZE + j] = m_board[i][j];
            // std::cout << "Checking the board data: " << dummy[i*BOARD_SIZE + j] << std::endl;
        }
    }
    cudaMalloc(&d_board, BOARD_SIZE * BOARD_SIZE * sizeof(Token)); // TODO :: Create a new variable for GPU . Do not use m_board. This has to be passed into the kernel
    cudaMemcpy(d_board, dummy, BOARD_SIZE * BOARD_SIZE * sizeof(Token), cudaMemcpyHostToDevice);
    on_gpu = 1;
    delete[] dummy;
}

__host__ __device__ void Board::print_board()
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("%d ",m_board[i][j]);
        }
        printf("\n");
        // std::cout << std::endl;
    }
}

void Board::move_to_cpu()
{
    // This function is not required for the assignment
    Token *dummy = new Token[BOARD_SIZE * BOARD_SIZE];
    cudaMemcpy(dummy, d_board, BOARD_SIZE * BOARD_SIZE * sizeof(Token), cudaMemcpyDeviceToHost); // TODO :: Create a new variable for GPU . Do not use m_board. This has to be passed into the kernel
    // return;
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            m_board[i][j] = dummy[i * BOARD_SIZE + j];
            // std::cout << "Where is the segmentation ? " << i << ", " << j << std::endl;
        }
    }
    // return;
    delete[] dummy;
    on_gpu = 0;
    cudaFree(d_board);
}
// CUDA kernel for get_winner needs to be written over here
/*void Board::clear_space()
{
    // This function is not required for the assignment
    cudaFree(d_board);
}*/
__host__ __device__ void Board::set_device_board()
{
    Token *dummy = new Token[BOARD_SIZE * BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            // printf("Hmm checking \n")
            dummy[i * BOARD_SIZE + j] = m_board[i][j];
            // printf("Where is the error man \n");
        }
    }
    d_board = dummy;
    // delete [] dummy;
}
// If theres no winner returns Token::EMPTY, if there is a winner return the player Token::BLACK/Token::WHITE
__host__ Token Board::get_winner_host()
{
    // move_to_gpu();
    // return Token(0);
    Token winner = Token::EMPTY;
    Token dummy = Token::EMPTY;
    dim3 block(8, 8);
    dim3 grid(BOARD_SIZE / block.x + 1, BOARD_SIZE / block.y + 1);
    Token *d_winner = &dummy;

    cudaMalloc(&d_winner, sizeof(Token));
    int *count;
    cudaMalloc(&count,sizeof(int));
    // cudaMemcpy(d_winner, &dummy, sizeof(Token), cudaMemcpyHostToDevice);

    check_winner_kernel<<<grid, block>>>(d_board, d_winner, BOARD_SIZE, WINNING_LENGTH,count);
    // cudaDeviceSynchronize();
    // std::cout << winner << std::endl;
    cudaMemcpy(&winner, d_winner, sizeof(Token), cudaMemcpyDeviceToHost);
    cudaFree(d_winner);
    cudaFree(count);
    // move_to_cpu();
    this->winner = winner;
    return winner;
}
__device__ Token Board::get_winner_device()
{
    Token winner = Token::EMPTY;
    Token dummy = Token::EMPTY;
    set_device_board();
    // printf("In get winner\n");
    dim3 block(8,8);
    dim3 grid(BOARD_SIZE / block.x + 1, BOARD_SIZE / block.y + 1);
    Token *d_winner = new Token[1];
    d_winner[0] = dummy;
    int *count= new int[1];
    count[0]=0;
    // printf("About to start kernel\n");
    check_winner_kernel<<<grid, block>>>(d_board, d_winner, BOARD_SIZE, WINNING_LENGTH,count);
    int dum = 0;
    // while(count[0]<256){
	//     printf("current count:%d \n",count[0]);//Removing this creates an endless loop for some reaso
    //     // dum = count[0] + 1;
    // }
    /*
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    */
    cudaDeviceSynchronize();
    // printf("count outside: %d\n",count[0]); 
    this->winner = d_winner[0];
    winner = d_winner[0];
    // printf("Kernel ended, winner is: %d\n",winner);
    // move_to_cpu();
    return winner;
}

Token Board::get_Token(int row, int col) const
{
    return m_board[row][col];
}
__device__ Position *Board::get_valid_moves_device(int &num_moves)
{
    int board_size = Board::BOARD_SIZE;
    set_device_board();
    // printf("Set device board\n");
    // Allocate memory for valid moves on the device
    Position *device_valid_moves = new Position[8 * 8];

    // Initialize valid_moves_count on the host and copy to the device
    int valid_moves_count = 0;
    int *device_valid_moves_count = new int[1];
    device_valid_moves_count[0] = valid_moves_count;
    // this->num_valid_moves = 0;
    // printf("Launching Kernel 4\n");
    dim3 block(8, 8);
    dim3 grid(BOARD_SIZE / block.x + 1, BOARD_SIZE / block.y + 1);
    // move_to_gpu();
    // for (int i = 0; i < BOARD_SIZE; i++)
    // {
    //     for (int j = 0; j < BOARD_SIZE; j++)
    //     {
    //         // printf("Hmm checking \n")
    //         // dummy[i * BOARD_SIZE + j] = m_board[i][j];
    //         printf("Board Values:  %d\n", d_board[i*board_size + j]);
    //     }
    // }
    // printf("About to launch the kernel \n");
    valid_moves_kernel<<<grid, block>>>(d_board, board_size, device_valid_moves, device_valid_moves_count);
    /*
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    valid_moves_kernel_tail<<<grid, block, 0, cudaStreamTailLaunch>>>();
    */
    cudaDeviceSynchronize();
    //cudaError_t err{cudaGetLastError()};
    
    //if (err != cudaSuccess)printf("This is the error: %s\n",cudaGetErrorString(err));
    // return device_valid_moves;
    // printf("Valid moves: %d\n", valid_moves_count);
    // this->num_valid_moves = *device_valid_moves_count;
    // num_moves = *device_valid_moves_count;
    num_moves = device_valid_moves_count[0];
    // printf("Number of moves: %d \n", num_moves);
    //  Free device memory
    //  clear_space();
    // move_to_cpu();

    return device_valid_moves;
}
__host__ Position *Board::get_valid_moves_host(int &num_moves)
{
    // Copy the board to the device
    int board_size = Board::BOARD_SIZE;

    // Allocate memory for valid moves on the device
    Position *device_valid_moves;
    cudaMalloc(&device_valid_moves, board_size * board_size * sizeof(Position));

    // Initialize valid_moves_count on the host and copy to the device
    int valid_moves_count = 0;
    int *device_valid_moves_count;
    cudaMalloc(&device_valid_moves_count, sizeof(int));
    // printf("Launching Kernel 3\n");
    cudaMemcpy(device_valid_moves_count, &valid_moves_count, sizeof(int), cudaMemcpyHostToDevice);
    // printf("Launching Kernel 4\n");
    dim3 block(8, 8);
    dim3 grid(BOARD_SIZE / block.x + 1, BOARD_SIZE / block.y + 1);
    // move_to_gpu();
    valid_moves_kernel<<<grid, block>>>(d_board, board_size, device_valid_moves, device_valid_moves_count);

    // Copy the result back to the host
    cudaMemcpy(&valid_moves_count, device_valid_moves_count, sizeof(int), cudaMemcpyDeviceToHost);
    Position *host_valid_moves = new Position[valid_moves_count];
    cudaMemcpy(host_valid_moves, device_valid_moves, valid_moves_count * sizeof(Position), cudaMemcpyDeviceToHost);
    // printf("Valid moves: %d\n", valid_moves_count);
    this->num_valid_moves = valid_moves_count;
    num_moves = valid_moves_count;
    //  Free device memory
    //  clear_space();
    // move_to_cpu();
    cudaFree(device_valid_moves);
    cudaFree(device_valid_moves_count);

    return host_valid_moves;
}

__host__ __device__ bool Board::is_draw()
{
    int num_moves = 0;
    // printf("I am in draw now: %d \n", this->num_valid_moves);
    // Position *valid_moves = get_valid_moves(num_moves);
    // delete[] valid_moves;
    return num_moves == this->num_valid_moves;
}

