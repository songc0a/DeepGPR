#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t error__ = (call);                                               \
    if (error__ != cudaSuccess) {                                               \
        std::cerr << cudaGetErrorString(error__) << " at " << __FILE__ << ':'  \
                  << __LINE__ << std::endl;                                     \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (0)

__global__ void touch_allocation(float* pointer, int iteration)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        pointer[0] = static_cast<float>(iteration);
    }
}

__global__ void graph_step(int* sink, int iteration)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *sink += iteration & 1;
    }
}

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double benchmark_malloc_free(std::size_t bytes, int iterations)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        float* pointer = nullptr;
        CUDA_CHECK(cudaMalloc(&pointer, bytes));
        touch_allocation<<<1, 1>>>(pointer, i);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(pointer));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return elapsed_ms(start, Clock::now()) / iterations;
}

static double benchmark_persistent_reuse(std::size_t bytes, int iterations)
{
    float* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(&pointer, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        touch_allocation<<<1, 1>>>(pointer, i);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double result = elapsed_ms(start, Clock::now()) / iterations;
    CUDA_CHECK(cudaFree(pointer));
    return result;
}

static double benchmark_malloc_free_async(std::size_t bytes, int iterations)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Prime the default device pool before timing it.
    float* warmup_pointer = nullptr;
    CUDA_CHECK(cudaMallocAsync(&warmup_pointer, bytes, stream));
    CUDA_CHECK(cudaFreeAsync(warmup_pointer, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        float* pointer = nullptr;
        CUDA_CHECK(cudaMallocAsync(&pointer, bytes, stream));
        touch_allocation<<<1, 1, 0, stream>>>(pointer, i);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFreeAsync(pointer, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double result = elapsed_ms(start, Clock::now()) / iterations;
    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
}

static void schedule_stream_event_work(
    cudaStream_t stream0, cudaStream_t stream1, cudaEvent_t* events)
{
    CUDA_CHECK(cudaEventRecord(events[0], stream0));
    CUDA_CHECK(cudaStreamWaitEvent(stream1, events[0], 0));
    CUDA_CHECK(cudaEventRecord(events[1], stream1));
    CUDA_CHECK(cudaStreamWaitEvent(stream0, events[1], 0));
    CUDA_CHECK(cudaEventRecord(events[2], stream0));
    CUDA_CHECK(cudaEventRecord(events[3], stream1));
}

static double benchmark_stream_event_create_destroy(int iterations)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        cudaStream_t streams[2] = {nullptr, nullptr};
        cudaEvent_t events[4] = {nullptr, nullptr, nullptr, nullptr};
        CUDA_CHECK(cudaStreamCreate(&streams[0]));
        CUDA_CHECK(cudaStreamCreate(&streams[1]));
        for (cudaEvent_t& event : events) CUDA_CHECK(cudaEventCreate(&event));
        schedule_stream_event_work(streams[0], streams[1], events);
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
        for (cudaEvent_t event : events) CUDA_CHECK(cudaEventDestroy(event));
        CUDA_CHECK(cudaStreamDestroy(streams[0]));
        CUDA_CHECK(cudaStreamDestroy(streams[1]));
    }
    return elapsed_ms(start, Clock::now()) / iterations;
}

static double benchmark_stream_event_reuse(int iterations)
{
    cudaStream_t streams[2] = {nullptr, nullptr};
    cudaEvent_t events[4] = {nullptr, nullptr, nullptr, nullptr};
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));
    for (cudaEvent_t& event : events) CUDA_CHECK(cudaEventCreate(&event));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        schedule_stream_event_work(streams[0], streams[1], events);
    }
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    double result = elapsed_ms(start, Clock::now()) / iterations;

    for (cudaEvent_t event : events) CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    return result;
}

struct GraphResult {
    double direct_ms;
    double replay_ms;
    double instantiate_ms;
};

static GraphResult benchmark_graph(int time_steps, int kernels_per_step, int repeats)
{
    int* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));
    CUDA_CHECK(cudaMemset(sink, 0, sizeof(int)));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto direct_start = Clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int step = 0; step < time_steps; ++step) {
            for (int kernel = 0; kernel < kernels_per_step; ++kernel) {
                graph_step<<<1, 1>>>(sink, step + kernel);
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double direct_ms = elapsed_ms(direct_start, Clock::now()) / repeats;

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int step = 0; step < time_steps; ++step) {
        for (int kernel = 0; kernel < kernels_per_step; ++kernel) {
            graph_step<<<1, 1, 0, stream>>>(sink, step + kernel);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    auto instantiate_start = Clock::now();
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    double instantiate_ms = elapsed_ms(instantiate_start, Clock::now());

    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto graph_start = Clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double replay_ms = elapsed_ms(graph_start, Clock::now()) / repeats;

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(sink));
    return {direct_ms, replay_ms, instantiate_ms};
}

template<class Function>
static double median_of_five(Function function)
{
    std::vector<double> samples;
    for (int trial = 0; trial < 5; ++trial) samples.push_back(function());
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main(int argc, char** argv)
{
    std::size_t bytes = argc > 1
        ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
        : 3ULL * 1024ULL * 1024ULL;
    int iterations = argc > 2 ? std::atoi(argv[2]) : 200;
    int time_steps = argc > 3 ? std::atoi(argv[3]) : 1200;
    int kernels_per_step = argc > 4 ? std::atoi(argv[4]) : 8;
    int graph_repeats = argc > 5 ? std::atoi(argv[5]) : 10;

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    CUDA_CHECK(cudaFree(nullptr));

    double malloc_free = median_of_five(
        [&] { return benchmark_malloc_free(bytes, iterations); });
    double reuse = median_of_five(
        [&] { return benchmark_persistent_reuse(bytes, iterations); });
    double async = median_of_five(
        [&] { return benchmark_malloc_free_async(bytes, iterations); });
    double stream_create = median_of_five(
        [&] { return benchmark_stream_event_create_destroy(iterations); });
    double stream_reuse = median_of_five(
        [&] { return benchmark_stream_event_reuse(iterations); });
    GraphResult graph = benchmark_graph(
        time_steps, kernels_per_step, graph_repeats);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "gpu=" << properties.name << '\n';
    std::cout << "allocation_bytes=" << bytes
              << " iterations=" << iterations << '\n';
    std::cout << "cudaMalloc_cudaFree_ms_per_call=" << malloc_free << '\n';
    std::cout << "persistent_reuse_ms_per_call=" << reuse << '\n';
    std::cout << "cudaMallocAsync_cudaFreeAsync_ms_per_call=" << async << '\n';
    std::cout << "stream_event_create_destroy_ms_per_call=" << stream_create << '\n';
    std::cout << "stream_event_persistent_reuse_ms_per_call=" << stream_reuse << '\n';
    std::cout << "graph_nodes=" << time_steps * kernels_per_step
              << " graph_repeats=" << graph_repeats << '\n';
    std::cout << "direct_kernel_sequence_ms=" << graph.direct_ms << '\n';
    std::cout << "cuda_graph_replay_ms=" << graph.replay_ms << '\n';
    std::cout << "cuda_graph_instantiate_ms=" << graph.instantiate_ms << '\n';
    return 0;
}
