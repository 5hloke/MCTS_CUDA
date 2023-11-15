#include "board.h"

using namespace std;

Board::Board() : m_board(BOARD_SIZE, std::vector<Token>(BOARD_SIZE, Token::EMPTY)) {}

Board::Board(const Board& other) : m_board(other.m_board) {} // this copy constructor is not correct

bool Board::valid_move(int row, int col) const {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE && m_board[row][col] == Token::EMPTY;
}

bool Board::make_move(int row, int col, Token player) {
    if (!valid_move(row, col)) {
        return false;
    }
    m_board[row][col] = player;
    return true;
}

bool Board::has_winner() const {
    return get_winner() != Token::EMPTY;
}

void Board::move_to_cpu() {
    // This function is not required for the assignment
    cudaMalloc(&m_board, BOARD_SIZE * BOARD_SIZE * sizeof(Token));
    cudaMemcpy(m_board, m_board, BOARD_SIZE * BOARD_SIZE * sizeof(Token), cudaMemcpyHostToDevice);
}
void Board::move_to_gpu(){
    // This function is not required for the assignment
    cudaMemcpy(m_board, m_board, BOARD_SIZE * BOARD_SIZE * sizeof(Token), cudaMemcpyDeviceToHost);
}
// CUDA kernel for get_winner needs to be written over here 
void Board::clear_space(){
    // This function is not required for the assignment
    cudaFree(m_board);
}

/*
{ //This needs to be more efficeient can be done on CUDA 
        // Check rows
        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col <= BOARD_SIZE - WINNING_LENGTH; ++col) {
                Token winner = check_winner(row, col, 0, 1);
                if (winner != Token::EMPTY) {
                    return winner;
                }
            }
        }

        // Check columns
        for (int row = 0; row <= BOARD_SIZE - WINNING_LENGTH; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                Token winner = check_winner(row, col, 1, 0);
                if (winner != Token::EMPTY) {
                    return winner;
                }
            }
        }

        // Check diagonals
        for (int row = 0; row <= BOARD_SIZE - WINNING_LENGTH; ++row) {
            for (int col = 0; col <= BOARD_SIZE - WINNING_LENGTH; ++col) {
                Token winner = check_winner(row, col, 1, 1);
                if (winner != Token::EMPTY) {
                    return winner;
                }
            }
        }
        for (int row = WINNING_LENGTH - 1; row < BOARD_SIZE; ++row) {
            for (int col = 0; col <= BOARD_SIZE - WINNING_LENGTH; ++col) {
                Token winner = check_winner(row, col, -1, 1);
                if (winner != Token::EMPTY) {
                    return winner;
                }
            }
        }

        return Token::EMPTY;
    }
*/


// GPU implementation of get valid moves needs to be written over here
/*
 const { // This can be done on CUDA as well
        std::vector<std::pair<int, int>> valid_moves;
        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                if (valid_move(row, col)) {
                    valid_moves.emplace_back(row, col);
                }
            }
        }
        return valid_moves;
    }
*/

Token Board::get_Token(int row, int col) const {
    return m_board[row][col];
}

__global__ void check_winner_kernel(Token* board, Token* winner, int size, int win_len){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > size || j > size){
        return
    }
    Token player = board[i*size + j];
    Token* vertical_up[win_len - 1] = {Token::EMPTY};
    Token* vertical_down[win_len -1];
    Token* horizontal_left[win_len -1];
    Token* horizontal_right[win_len-1];
    Token* diag1[win_len-1];
    Token* diag2[win_len-1];
    Token* diag3[win_len-1];
    Token* diag4[win_len-1];

    // Vertical Checks, horizontal checks, diagonals
    for (int k = 0; k < win_len-1; k++){
        if(i+k < size){
            vertical_up[k] = board[(i+k)*size + j];
        }
        if (i-k > 0){
            vertical_down[k] = board[(i-k)*size + j];
        }
        if (j-k > 0){
            horizontal_left[k] = board[i*size + (j-k)];
        }
        if (j+k < size){
            horizontal_right[k] = board[i*size + (j+k)];
        }
        if (i+k < size && j+k < size){
            diag1[k] = board[(i+k)*size + (j+k)];
        }
        if (i-k > 0 && j-k > 0){
            diag2[k] = board[(i-k)*size + (j-k)];
        }
        if (i-k > 0 && j+k < size){
            diag3[k] = board[(i-k)*size + (j+k)];
        }
        if (i+k < size && j-k > 0){
            diag4[k] = board[(i+k)*size + (j-k)];
        }
    }
    // Check for winner
    int up, down, left, right, diag1, diag2, diag3, diag4 = 1;
    for (int k = 0; k < win_len; k++){
        if (vertical_up[k] != player && up != 0){
            up = 0;
        }
        if (vertical_down[k] != player && down != 0){
            down = 0;
        }
        if (horizontal_left[k] != player && left != 0){
            left = 0;
        }
        if (horizontal_right[k] != player && right != 0){
            right = 0;
        }
        if (diag1[k] != player && diag1 != 0){
            diag1 = 0;
        }
        if (diag2[k] != player && diag2 != 0){
            diag2 = 0;
        }
        if (diag3[k] != player && diag3 != 0){
            diag3 = 0;
        }
        if (diag4[k] != player && diag4 != 0){
            diag4 = 0;
        }   
    }
    if (up == 1 || down == 1 || left == 1 || right == 1 || diag1 == 1 || diag2 == 1 || diag3 == 1 || diag4 == 1){
        *winner = player;
        return;
    }
}
// CUDA kernel for check_winner needs to be written over here
/* // This can be done on CUDA as well 
        Token first_Token = m_board[row][col];
        if (first_Token == Token::EMPTY) {
            return Token::EMPTY;
        }
        for (int i = 1; i < WINNING_LENGTH; ++i) {
            int r = row + i * row_delta;
            int c = col + i * col_delta;
            if (m_board[r][c] != first_Token) {
                return Token::EMPTY;
            }
        }
        return first_Token;
    }*/

