#include <iostream>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <thread>
#include <chrono>

#include "brandes-kernels.h"

constexpr int MDEG = 4;


std::chrono::high_resolution_clock::time_point timers[4];
double measured_time[4] {0, 0, 0, 0};
cudaEvent_t start[4], stop[4];

enum Timer {
    ALL,
    HOST,
    MEMCPY,
    KERNEL
};

void start_timer(Timer timer) {
    if (timer == MEMCPY || timer == KERNEL) {
        cudaEventCreate(start + timer);
        cudaEventRecord(start[timer]);
    }
    else {
        timers[timer] = std::chrono::high_resolution_clock::now();
    }
}

void stop_timer(Timer timer) {
    if (timer == MEMCPY || timer == KERNEL) {
        cudaEventCreate(stop + timer);
        cudaEventRecord(stop[timer]);
        cudaEventSynchronize(stop[timer]);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start[timer], stop[timer]);

        cudaEventDestroy(start[timer]);
        cudaEventDestroy(stop[timer]);

        measured_time[timer] += milliseconds;
    }
    else {
        auto end = std::chrono::high_resolution_clock::now();
        measured_time[timer] += std::chrono::duration_cast<std::chrono::milliseconds>(end - timers[timer]).count();
    }
}

void delete_strideCSR(strideCSR &CSR) {
    delete[] CSR.offset;
    delete[] CSR.vmap;
    delete[] CSR.nvir;
    delete[] CSR.ptrs;
    delete[] CSR.adjs;
}

strideCSR create_stride_CSR(int N, int input_len, int mdeg, int *deg, int *CSR, int *vmap) {
    int size = 0;
    for (int u = 0; u < N; u++) {
        int old_u = vmap[u];
        size += (CSR[old_u + 1] - CSR[old_u] + mdeg - 1) / mdeg;
    }

    int new_N = 0;
    int *offset = new int[size];
    int *new_vmap = new int[size];
    int *nvir = new int[N];
    int *ptrs = new int[N + 1];
    int *adjs = new int[input_len * 2];

    std::map<int, int> rev_vmap;
    for (int u = 0; u < N; u++) {
        rev_vmap[vmap[u]] = u;
    }

    ptrs[0] = 0;
    for (int u = 0; u < N; u++) {
        int old_u = vmap[u];

        int off = 0;
        for (int p = CSR[old_u]; p < CSR[old_u + 1]; p++) {
            int v = CSR[p];
            if (deg[v] != 0) {
                adjs[ptrs[u] + off] = rev_vmap[v];
                off++;
            }
        }

        if (off != deg[old_u]) {
            fprintf(stderr, "Vertex degree is incorrect\n");
            exit(1);
        }

        off = 0;
        nvir[u] = 0;
        ptrs[u + 1] = ptrs[u] + deg[old_u];
        for (int i = 0; i < deg[old_u]; i += mdeg) {
            offset[new_N] = off;
            off++;
            new_vmap[new_N] = u;
            nvir[u]++;
            new_N++;
        }
    }

    return {new_N, offset, new_vmap, nvir, ptrs, adjs};
}

int* create_CSR(int N, int input_len, std::vector<int> &deg, std::vector<std::vector<int>> &graph) {
    int *CSR = new int[N + input_len * 2 + 1];
    std::vector<int> offset(N, 0);

    CSR[0] = N + 1;
    for (int u = 0; u < N; u++) {
        CSR[u + 1] = CSR[u] + deg[u];

        int cnt = 0;
        for (auto v : graph[u]) {
            CSR[CSR[u] + cnt] = v;
            cnt++;
        }
    }

    return CSR;
}

template<typename T>
inline cudaError_t fillMem(void *ptr, T val, int size) {
    T *arr = new T[size];
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }

    start_timer(MEMCPY);
    cudaError_t err = cudaMemcpy(ptr, arr, size * sizeof(T), cudaMemcpyHostToDevice);
    stop_timer(MEMCPY);
    delete[] arr;

    return err;
}

int main(int argc, char *argv[]) {
    start_timer(ALL);
    start_timer(HOST);

    int max_v = -1;
    std::map<int, int> v_mapping;
    std::set<int> V;
    std::vector<int> deg;
    std::vector<std::vector<int>> graph;
    std::vector<int> vmap;
    int N = 0, input_len = 0;

    // Read input
#ifdef PRINT_PROGRESS
    printf("Read input\n");
#endif
    {
        FILE *f_in = fopen(argv[1], "r");
        int v1, v2;
        while (fscanf(f_in, "%d%d", &v1, &v2) != EOF) {
            input_len++;
            max_v = std::max(max_v, std::max(v1, v2));

            if (v_mapping.count(v1) == 0) {
                v_mapping[v1] = N;
                vmap.push_back(v1);
                deg.push_back(0);
                graph.push_back(std::vector<int>{});
                N++;
            }

            if (v_mapping.count(v2) == 0) {
                v_mapping[v2] = N;
                vmap.push_back(v2);
                deg.push_back(0);
                graph.push_back(std::vector<int>{});
                N++;
            }

            int v1m = v_mapping[v1];
            int v2m = v_mapping[v2];

            deg[v1m]++;
            deg[v2m]++;

            graph[v1m].push_back(v2m);
            graph[v2m].push_back(v1m);
        }
        fclose(f_in);
    }

    // Create CSR
#ifdef PRINT_PROGRESS
    printf("Create CSR\n");
#endif
    int *initialCSR = create_CSR(N, input_len, deg, graph);

    // Remove 1 degs
#ifdef PRINT_PROGRESS
    printf("Remove 1 deg\n");
#endif

#ifndef SKIP_1DEG
    int num_removed;
    int *reach = new int[N];

    int *dev_deg;
    int *dev_csr;
    double *dev_bc;
    int *dev_reach;
    {
        bool *dev_cont;
        int *dev_removed;
        int *dev_num_removed;

        cudaMalloc(&dev_deg, N * sizeof(int));
        cudaMalloc(&dev_csr, (N + input_len * 2 + 1) * sizeof(int));
        cudaMalloc(&dev_bc, N * sizeof(double));
        cudaMalloc(&dev_cont, sizeof(bool));
        cudaMalloc(&dev_removed, N * sizeof(int));
        cudaMalloc(&dev_reach, N * sizeof(int));
        cudaMalloc(&dev_num_removed, sizeof(int));

        start_timer(MEMCPY);
        cudaMemcpy(dev_deg, deg.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_csr, initialCSR, (N + input_len * 2 + 1) * sizeof(int), cudaMemcpyHostToDevice);
        stop_timer(MEMCPY);
        fillMem(dev_bc, 0, N);
        fillMem(dev_removed, 0, N);
        fillMem(dev_reach, 1, N);
        fillMem(dev_num_removed, 0, 1);

        int threads = std::min(N, 1024);
        int blocks = (N + threads - 1) / threads;
        int step = 1;
        bool cont = false;
        
        do {
            fillMem(dev_cont, false, 1);

            stop_timer(HOST);
            start_timer(KERNEL);
            remove_1deg<<<threads, blocks>>>(N, step, dev_removed, dev_deg, dev_csr, dev_reach, dev_cont, dev_bc, dev_num_removed);
            stop_timer(KERNEL);
            start_timer(HOST);

            start_timer(MEMCPY);
            cudaMemcpy(&cont, dev_cont, sizeof(bool), cudaMemcpyDeviceToHost);
            stop_timer(MEMCPY);
            step++;
        } while(cont);

        start_timer(MEMCPY);
        cudaMemcpy(&num_removed, dev_num_removed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(reach, dev_reach, N * sizeof(int), cudaMemcpyDeviceToHost);
        stop_timer(MEMCPY);

        cudaFree(dev_cont);
        cudaFree(dev_removed);
        cudaFree(dev_num_removed);
    }
#else
    int num_removed = 0;
    int *reach = new int[N];

    int *dev_deg;
    int *dev_csr;
    double *dev_bc;
    int *dev_reach;

    cudaMalloc(&dev_deg, N * sizeof(int));
    cudaMalloc(&dev_csr, (N + input_len * 2 + 1) * sizeof(int));
    cudaMalloc(&dev_bc, N * sizeof(double));
    cudaMalloc(&dev_reach, N * sizeof(int));

    start_timer(MEMCPY);
    cudaMemcpy(dev_deg, deg.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csr, initialCSR, (N + input_len * 2 + 1) * sizeof(int), cudaMemcpyHostToDevice);
    stop_timer(MEMCPY);
    fillMem(dev_bc, 0, N);
    fillMem(dev_reach, 1, N);

    start_timer(MEMCPY);
    cudaMemcpy(reach, dev_reach, N * sizeof(int), cudaMemcpyDeviceToHost);
    stop_timer(MEMCPY);
#endif

    // Order by BFS
#ifdef PRINT_PROGRESS
    printf("Order by BFS\n");
#endif

#ifndef SKIP_BFS
    int *new_vmap = new int[N];

    int *dev_new_vmap;
    {
        int *visited = new int[N];

        bool *dev_cont;
        int *dev_visited;
        int *dev_num_visited;

        cudaMalloc(&dev_new_vmap, N * sizeof(int));
        cudaMalloc(&dev_cont, sizeof(bool));
        cudaMalloc(&dev_visited, N * sizeof(int));
        cudaMalloc(&dev_num_visited, sizeof(int));

        fillMem(dev_visited, 0, N);
        fillMem(dev_new_vmap, 0, N);
        fillMem(dev_num_visited, 0, 1);

        int num_visited = 0;
        int should_visit = N - num_removed;
        int last_v_checked = 0;
        int threads = std::min(N, 1024);
        int blocks = (N + threads - 1) / threads;
        int step = 1;
        
        while(should_visit - num_visited) {
            start_timer(MEMCPY);
            cudaMemcpy(visited, dev_visited, N * sizeof(int), cudaMemcpyDeviceToHost);
            stop_timer(MEMCPY);

            while (visited[last_v_checked] != 0 || deg[last_v_checked] <= 0) {
                last_v_checked++;
                
                if (last_v_checked == N) {
                    fprintf(stderr, "BFS can't visit all vertices\n");
                    exit(1);
                }
            }
            
            fillMem(dev_visited + last_v_checked, step, 1);


            bool cont = false;
            do {
                cudaError_t t = fillMem(dev_cont, false, 1);
                stop_timer(HOST);
                start_timer(KERNEL);
                BFS_order<<<threads, blocks>>>(N, step, dev_deg, dev_csr, dev_cont, dev_visited, dev_new_vmap, dev_num_visited);
                stop_timer(KERNEL);
                start_timer(HOST);

                start_timer(MEMCPY);
                cudaMemcpy(&num_visited, dev_num_visited, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&cont, dev_cont, sizeof(bool), cudaMemcpyDeviceToHost);
                stop_timer(MEMCPY);
                step++;
            } while(cont);
        }

        start_timer(MEMCPY);
        cudaMemcpy(new_vmap, dev_new_vmap, N * sizeof(int), cudaMemcpyDeviceToHost);
        stop_timer(MEMCPY);

        cudaFree(dev_cont);
        cudaFree(dev_visited);
        cudaFree(dev_num_visited);
        delete[] visited;
    }
#else
    int *new_vmap = new int[N];
    {
        int *v_deg = new int[N];
    
        start_timer(MEMCPY);
        cudaMemcpy(v_deg, dev_deg, N * sizeof(int), cudaMemcpyDeviceToHost);
        stop_timer(MEMCPY);

        int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (v_deg[i] <= 0) {
                continue;
            }

            new_vmap[cnt] = i;
            cnt++;
        }

        delete[] v_deg;
    }

    int *dev_new_vmap;
    cudaMalloc(&dev_new_vmap, N * sizeof(int));
    start_timer(MEMCPY);
    cudaMemcpy(dev_new_vmap, new_vmap, N * sizeof(int), cudaMemcpyHostToDevice);
    stop_timer(MEMCPY);
#endif

    int N_without_1deg = N - num_removed;
    int *v_deg = new int[N];
    
    start_timer(MEMCPY);
    cudaMemcpy(v_deg, dev_deg, N * sizeof(int), cudaMemcpyDeviceToHost);
    stop_timer(MEMCPY);
    cudaFree(dev_deg);

    // Create Stride CSR
#ifdef PRINT_PROGRESS
    printf("Create stride CSR\n");
#endif

#ifndef SKIP_STRIDE
    strideCSR strideCSR = create_stride_CSR(N_without_1deg, input_len, MDEG, v_deg, initialCSR, new_vmap);
#else
    strideCSR strideCSR = create_stride_CSR(N_without_1deg, input_len, 2 * input_len, v_deg, initialCSR, new_vmap);
#endif

    delete[] v_deg;

    double *bc = new double[N];
    cudaMemcpy(bc, dev_bc, N * sizeof(double), cudaMemcpyDeviceToHost);

    int *o = new int[N_without_1deg];
    double *delta = new double[N_without_1deg];
    int *dev_offset;
    int *dev_vmap;
    int *dev_nvir;
    int *dev_ptrs;
    int *dev_adjs;
    cudaMalloc(&dev_offset, strideCSR.N * sizeof(int));
    cudaMalloc(&dev_vmap, strideCSR.N * sizeof(int));
    cudaMalloc(&dev_nvir, N_without_1deg * sizeof(int));
    cudaMalloc(&dev_ptrs, (N_without_1deg + 1) * sizeof(int));
    cudaMalloc(&dev_adjs, input_len * 2 * sizeof(int));
    struct strideCSR dev_strideCSR{strideCSR.N, dev_offset, dev_vmap, dev_nvir, dev_ptrs, dev_adjs};
    start_timer(MEMCPY);
    cudaMemcpy(dev_offset, strideCSR.offset, strideCSR.N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vmap, strideCSR.vmap, strideCSR.N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_nvir, strideCSR.nvir, N_without_1deg * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptrs, strideCSR.ptrs, (N_without_1deg + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_adjs, strideCSR.adjs, input_len * 2 * sizeof(int), cudaMemcpyHostToDevice);
    stop_timer(MEMCPY);


    int *dev_o;
    bool *dev_cont;
    int *dev_d;
    double *dev_delta;

    cudaMalloc(&dev_delta, N_without_1deg * sizeof(double));
    cudaMalloc(&dev_d, N_without_1deg * sizeof(int));
    cudaMalloc(&dev_o, N_without_1deg * sizeof(int));
    cudaMalloc(&dev_cont, sizeof(bool));

    for (int s = 0; s < N_without_1deg; s++) {
#ifdef PRINT_PROGRESS
        if (s % 5000 == 0) {
            printf("Progress: %d / %d\n", s, N_without_1deg);
        }
#endif
        // Forward
        int l = 0;
        int threads = std::min(strideCSR.N, 1024);
        int blocks = (strideCSR.N + threads - 1) / threads;
        {
            fillMem(dev_d, -1, N_without_1deg);
            fillMem(dev_o, 0, N_without_1deg);

            fillMem(dev_d + s, 0, 1);
            fillMem(dev_o + s, 1, 1);

            
            bool cont = false;
            do {
                fillMem(dev_cont, false, 1);

                stop_timer(HOST);
                start_timer(KERNEL);
                forward<<<threads, blocks>>>(dev_strideCSR, dev_d, dev_o, l, dev_cont);
                stop_timer(KERNEL);
                start_timer(HOST);
                start_timer(MEMCPY);
                cudaMemcpy(&cont, dev_cont, sizeof(bool), cudaMemcpyDeviceToHost);
                stop_timer(MEMCPY);
                l++;
            } while(cont);
        }

        int threads2 = std::min(N_without_1deg, 1024);
        int blocks2 = (N_without_1deg + threads2 - 1) / threads2;
        stop_timer(HOST);
        start_timer(KERNEL);
        update_delta<<<threads2, blocks2>>>(N_without_1deg, dev_delta, dev_o, dev_reach, dev_new_vmap);
        stop_timer(KERNEL);
        start_timer(HOST);

        // Backward
        {

            while (l) {
                l--;
                stop_timer(HOST);
                start_timer(KERNEL);
                backward<<<threads, blocks>>>(dev_strideCSR, l, dev_d, dev_delta);
                stop_timer(KERNEL);
                start_timer(HOST);
            }
        }

        stop_timer(HOST);
        start_timer(KERNEL);
        update_bc<<<threads2, blocks2>>>(N_without_1deg, s, dev_bc, dev_delta, dev_o, dev_reach, dev_new_vmap);
        stop_timer(KERNEL);
        start_timer(HOST);
    }
    
    start_timer(MEMCPY);
    cudaMemcpy(bc, dev_bc, N * sizeof(double), cudaMemcpyDeviceToHost);
    stop_timer(MEMCPY);
 
    FILE *f = fopen(argv[2], "w");
    for (int i = 0; i <= max_v; i++) {
        if (v_mapping.count(i)) {
            fprintf(f, "%lf\n", bc[v_mapping[i]]);
        }
        else {
            fprintf(f, "0\n");
        }
    }
    fclose(f);

    // Cleanup
    cudaFree(dev_new_vmap);
    cudaFree(dev_offset);
    cudaFree(dev_reach);
    cudaFree(dev_vmap);
    cudaFree(dev_nvir);
    cudaFree(dev_ptrs);
    cudaFree(dev_adjs);
    cudaFree(dev_cont);
    cudaFree(dev_o);
    cudaFree(dev_delta);
    cudaFree(dev_d);
    cudaFree(dev_bc);

    delete_strideCSR(strideCSR);
    delete[] o;
    delete[] bc;
    delete[] initialCSR;
    delete[] reach;
    delete[] new_vmap;

    // Print metrics
    stop_timer(HOST);
    stop_timer(ALL);

#ifdef DETAILED_METRICS
    fprintf(stderr, "%d\n", int(measured_time[ALL]));
    fprintf(stderr, "%d\n", int(measured_time[KERNEL]));
    fprintf(stderr, "%d\n", int(measured_time[HOST]));
    fprintf(stderr, "%d\n", int(measured_time[MEMCPY]));
#else
    fprintf(stderr, "%d\n", int(measured_time[KERNEL]));
    fprintf(stderr, "%d\n", int(measured_time[KERNEL] + measured_time[MEMCPY]));
#endif
}