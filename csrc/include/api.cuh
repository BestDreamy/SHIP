#ifndef _API_H_
#define _API_H_

template<typename T>
T ceil_div(T x, T y) {
    return ((x + y - 1) / y);
}

#endif