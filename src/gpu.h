#pragma once

#include "common.h"

constexpr uint32_t seeds_per_gpu_iteration = UINT64_C(1) << 28;

struct SeedIterator {
    std::atomic_uint64_t pos;
    uint64_t end; // UINT64_MAX = unbounded

    SeedIterator(uint64_t start, uint64_t end = UINT64_MAX) : pos(start), end(end) {

    }

    // Returns UINT64_MAX if range exhausted
    uint64_t next(uint64_t count) {
        uint64_t cur = pos.fetch_add(count);
        return cur >= end ? UINT64_MAX : cur;
    }
};

struct GpuThread: Thread<GpuThread> {
    int device;
    SeedIterator &input;
    GpuOutputs &outputs;
    bool benchmark;

    GpuThread(int device, SeedIterator &input, GpuOutputs &outputs, bool benchmark = false);

    void run();
};
