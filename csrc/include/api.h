#ifndef _API_H_
#define _API_H_
#include <cstdint>

// template<typename T>
inline uint32_t ceil_div(uint32_t x, uint32_t y) {
    return ((x + y - 1) / y);
}

#endif