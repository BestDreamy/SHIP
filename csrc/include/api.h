#ifndef _API_H_
#define _API_H_
#include <cstdint>
#include <iostream>
#include <vector>

// template<typename T>
inline uint32_t ceil_div(uint32_t x, uint32_t y) {
    return ((x + y - 1) / y);
}

inline void print_transmit_information (
    const std::vector<uint32_t> &tokens_h,
    const std::vector<uint32_t> &indices_h,
    uint32_t localTokens,
    uint32_t hiddenDim,
    uint32_t expertsPerToken
) {
    for (int i = 0; i < localTokens; i ++) {
        std::cout << "Token " << i << ": ";
        for (int j = 0; j < hiddenDim; j ++) {
            std::cout << tokens_h[i * hiddenDim + j] << " ";
        }
        std::cout << "\n";
    }
    for (int i = 0; i < localTokens; i ++) {
        std::cout << "Token " << i << " will tranmit to expert: ";
        for (int j = 0; j < expertsPerToken; j ++) {
            std::cout << indices_h[i * expertsPerToken + j] << " ";
        }
        std::cout << "\n";
    }
}

#endif