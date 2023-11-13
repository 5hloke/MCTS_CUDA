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


// CUDA kernel for get_winner needs to be written over here 

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

