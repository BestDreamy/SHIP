#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda.h>

#define WORLD_SIZE 8

#include <cuda.h>
#include <cstdint>
#include "../include/buffer.h"
#include "../all2all/intranode.h"
#include <assert.h>
#include "../include/api.h"
#include <fstream>
#include <tuple>
#include <iostream>
#include <iomanip>

using namespace ship;

using Time = std::pair<float, float>;

struct BenchConfig {
    uint32_t rank;
    uint32_t world_size;
    uint32_t localTokens;
    uint32_t hiddenDim;
    uint32_t numExperts;
    uint32_t expertsPerToken;
    uint32_t maxNumTokens;  
};

void genTestData(
    unsigned rank,
    unsigned world_size,
    uint32_t localTokens,
    uint32_t hiddenDim,
    uint32_t numExperts,
    uint32_t expertsPerToken,
    uint32_t maxNumTokens,
    std::vector<uint32_t> &tokens_h,
    std::vector<uint32_t> &indices_h,
    std::ofstream &logFile
) {
    Assert(numExperts / world_size == expertsPerToken, "Just for test, rank[i] and rank[i+1] buterfly transfer the same token");
    uint32_t numLocalExperts = numExperts / world_size;
    assert(numExperts % world_size == 0);

    logFile << "Total ranks: " << world_size << "\n";
    logFile << "Each rank will transfer tokens num: " << localTokens << "\n";
    logFile << "Each rank have experts num: " << numLocalExperts << "\n";
    logFile << "Each token have experts num: " << expertsPerToken << "\n";

    for (int i = 0; i < localTokens; i ++) {
        // For each token, assign it to other rank
        for (int j = 0; j < expertsPerToken; j ++) {
            // indices_h[i * expertsPerToken + j] = j;
            indices_h[i * expertsPerToken + j] = (rank ^ 0x1) * numLocalExperts + j;
        }
    }
    for (int i = 0; i < localTokens; i ++) {
        for (int j = 1; j < expertsPerToken; j ++) {
            Assert(indices_h[i * expertsPerToken] != indices_h[i * expertsPerToken + j], "The same token should not be assigned to the same expert");
        }
    }
        
    print_transmit_information(tokens_h, indices_h, localTokens, hiddenDim, expertsPerToken, rank, logFile);
}

Time average(const std::vector<float> &timesUs) {
    float sum = 0.0f, sumSquared = 0.0f;
    for (const float time : timesUs) {
        sum += time;
        sumSquared += time * time;
    }
    const float mean = sum / timesUs.size();
    const float stddev = std::sqrt(sumSquared / timesUs.size() - mean * mean);
    return std::make_pair(mean, stddev);
}

std::pair<Time, Time> benchDispatch(
    const unsigned &repeat,
    const BenchConfig &config,
    unsigned currentPE,
    unsigned numPEs
) {
    auto [
        rank,
        world_size, 
        localTokens, 
        hiddenDim, 
        numExperts, 
        expertsPerToken, 
        maxNumTokens
    ] = config;

    std::ofstream logFile("rank_" + std::to_string(rank) + ".log");
    
    std::vector<uint32_t> tokens_h(localTokens * hiddenDim, rank + 10); // All elements initialized to rank
    std::vector<uint32_t> indices_h(localTokens * expertsPerToken, 0);

    genTestData(
        rank, 
        world_size,
        localTokens,
        hiddenDim,
        numExperts,
        expertsPerToken,
        maxNumTokens,
        tokens_h,
        indices_h,
        logFile
    );

    // Device buffers
    DeviceBuffer<uint32_t> tokens_d(tokens_h);
    DeviceBuffer<uint32_t> indices_d(indices_h);

    const uint32_t hiddenDimBytes = hiddenDim * sizeof(tokens_d.get()[0]);

    AllToAllIntraNode allToAllIntranode(
        rank,
        world_size,
        localTokens,
        hiddenDim,
        hiddenDimBytes,
        numExperts,
        expertsPerToken,
        maxNumTokens
    );

    nvshmem_barrier_all();

    constexpr size_t numSamples = 1;

    std::tuple<cudaEvent_t, cudaEvent_t, cudaEvent_t> events[numSamples];
    for (size_t i = 0; i < numSamples; ++i) {
        cudaEventCreate(&std::get<0>(events[i]));
        cudaEventCreate(&std::get<1>(events[i]));
        cudaEventCreate(&std::get<2>(events[i]));
    }

    // Warmup
    auto run = [&]() -> std::pair<float, float> {
        // Dispatch.
        for (size_t i = 0; i < numSamples; i++) {
            cudaEventRecord(std::get<0>(events[i]));

            allToAllIntranode.dispatch(
                Stride1D<uint32_t>(tokens_d, hiddenDim),
                Stride2D<uint32_t>(indices_d, 1, expertsPerToken),
               logFile
            );

            cudaEventRecord(std::get<1>(events[i]));

            // allToAll.combine<T, U>(
            //     Strided1D<U>(outTokensDevice, config.hiddenDim),
            //     Strided2D<uint32_t>(indicesDevice, 1, config.expertsPerToken),
            //     Strided2D<float>(weightsDevice, 1, config.expertsPerToken),
            //     Strided2D<T>(
            //         outExpertDevice, config.hiddenDim, config.hiddenDim * config.numTokens * numPEs
            //     ),
            //     data.m,
            //     nullptr,
            //     SplitMode::NONE,
            //     stream
            // );

            // CUDACHECK(cudaEventRecord(std::get<2>(events[i]), stream));
        }

        float totalDispatchMs = 0.0f, totalCombineMs = 0.0f;
        for (size_t i = 0; i < numSamples; i++) {
            float dispatchMs = 0.0f, combineMs = 0.0f;
            cudaEventElapsedTime(&dispatchMs, std::get<0>(events[i]), std::get<1>(events[i]));
            // CUDACHECK(cudaEventElapsedTime(&combineMs, std::get<1>(events[i]), std::get<2>(events[i])));
            totalDispatchMs += dispatchMs;
            totalCombineMs += combineMs;
        }
        return {totalDispatchMs / numSamples, totalCombineMs / numSamples};
    };

    nvshmem_barrier_all();
//     nvtxRangePush("warmup");
    for (int i = 0; i < 10; i++) {
        run();
    }
//     nvtxRangePop();

    nvshmem_barrier_all();
//     nvtxRangePush("benchmark");
    std::vector<float> dispatchTimeUs, combineTimeUs;
    for (int i = 0; i < repeat; i++) {
        auto [dispatchTimeMs, combineTimeMs] = run();
        dispatchTimeUs.push_back(dispatchTimeMs * 1000);
        combineTimeUs.push_back(combineTimeMs * 1000);
    }
//     nvtxRangePop();

    return {average(dispatchTimeUs), average(combineTimeUs)};
}

int main(int argc, char **argv) {
    nvshmem_init();

    unsigned int my_pe = nvshmem_my_pe();
    unsigned int n_pes = nvshmem_n_pes();

    int deviceId = my_pe % 8;
    cudaSetDevice(deviceId);

    BenchConfig config = {
        .rank = my_pe,
        .world_size = n_pes,
        .localTokens = 4,
        .hiddenDim = 16,
        .numExperts = 8,
        .expertsPerToken = 2,
        .maxNumTokens = 10
    };

    auto maybe_print_bench_results = [](
        int const myPE,
        BenchConfig const &config,
        Time const &dispatch_time,
        Time const &combine_time
    ) {
        if (myPE == 0) {
        auto [dispatchMean, dispatchStddev] = dispatch_time;
        auto [combineMean, combineStddev] = combine_time;
        std::cout << std::setw(6) << config.localTokens << " " << std::setw(3)
                    << config.numExperts << " " << std::setw(3) << config.expertsPerToken << " "
                    << std::setw(4) << config.hiddenDim << " " << std::fixed << std::setprecision(3)
                    << "Dispatch: " << std::setw(10) << dispatchMean << "us ± " << dispatchStddev
                    << "us "
                    << "Combine: " << std::setw(10) << combineMean << "us ± " << combineStddev << "us"
                    << std::endl;
        }
    };

    auto [dispatch, combine] = benchDispatch(1, config, my_pe, n_pes);
    maybe_print_bench_results(my_pe, config, dispatch, combine);

    nvshmem_finalize();

    return 0;
}