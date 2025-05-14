#ifndef _API_H_
#define _API_H_
#include <cstdint>
#include <iostream>
#include <vector>
#include <assert.h>
#include <stdio.h>

#define RESET_TXT "\033[0m"
#define RED_TXT "\033[1;31m"
#define GREEN_TXT "\033[1;32m"
#define BLUE_TXT "\033[1;34m"
#define YELLOW_TXT "\033[1;33m"

#define dbg(a) std::cout << #a << " : " << (a) << std::endl

#define PRINT_RESULT(condition) \
    do { \
        if (condition) { \
            std::cout << "\033[1;32mPASS\033[0m" << std::endl; /* Green */ \
        } else { \
            std::cout << "\033[1;31mFAIL\033[0m" << std::endl; /* Red */ \
        } \
    } while (0)

#define Log(format, ...) \
    do { \
        printf(ANSI_FMT("[%s:%d %s] " format, BLUE_TXT) "\n", \
        __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
    } while(0)

inline void assert_fail_msg(const char* msg) {
    printf(RED_TXT "%s\n" RESET_TXT, msg);
}

#define Assert(cond, msg) \
    do { \
        if (!(cond)) { \
            assert_fail_msg(msg); \
        } \
        assert(cond); \
    } while(0)
    
#define Exit(cond, msg) \
    do { \
        if (!(cond)) { \
            assert_fail_msg(msg); \
        } \
        tfp -> close(); \
        if (!(cond)) exit(0); \
    } while(0)

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