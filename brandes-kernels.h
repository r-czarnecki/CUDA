#include <iostream>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <thread>
#include <chrono>

struct strideCSR {
    int N;
    int *offset;
    int *vmap;
    int *nvir;
    int *ptrs;
    int *adjs;
};

__global__ void remove_1deg(int N, int step, int *removed, int *deg, int *CSR, int *reach, bool *cont, double *bc, int *num_removed) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < N) {
        if (deg[u] == 1) {
            int vminr = N - reach[u];
            bc[u] += (reach[u] - 1) * vminr;
            *cont = true;
            removed[u] = step;
            deg[u] = 0;
            atomicAdd(num_removed, 1);

            int end = CSR[u + 1];
            for (int p = CSR[u]; p < end; p++) {
                int v = CSR[p];
                if ((removed[v] == 0 && deg[v] == 1) || removed[v] == step) {
                    if (u > v) {
                        break;
                    }

                    atomicAdd(reach + v, reach[u]);
                    atomicAdd(deg + v, -1);
                }
                else if (removed[v] == 0) {
                    atomicAdd(bc + v, reach[u] * (vminr - 1));
                    atomicAdd(reach + v, reach[u]);
                    int old = atomicAdd(deg + v, -1);

                    if (old == 1) {
                        atomicAdd(num_removed, 1);
                    }
                    break;
                }
            }
        }
    }
}

__global__ void BFS_order(int N, int step, int *deg, int *CSR, bool *cont, int *visited, int *new_vmap, int *num_visited) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < N && deg[u] > 0 && visited[u] == step) {
        int u_num = atomicAdd(num_visited, 1);
        new_vmap[u_num] = u;


        int end = CSR[u + 1];
        for (int p = CSR[u]; p < end; p++) {
            int v = CSR[p];
            if (deg[v] <= 0) {
                continue;
            }

            int old_v = atomicCAS(visited + v, 0, step + 1);

            if (old_v == 0) {
                *cont = true;
            }
        }
    }
}

__global__ void forward(strideCSR CSR, int *d, int *o, int l, bool *cont) {
    int vir_u = blockIdx.x * blockDim.x + threadIdx.x;

    if (vir_u < CSR.N) {
        int offset = CSR.offset[vir_u];
        int u = CSR.vmap[vir_u];
        int nvir = CSR.nvir[u];

        if (d[u] == l) {
            for (int p = CSR.ptrs[u] + offset; p < CSR.ptrs[u + 1]; p += nvir) {
                int v = CSR.adjs[p];
                if (d[v] == -1) {
                    d[v] = l + 1;
                    *cont = true;
                }

                if (d[v] == l + 1) {
                    atomicAdd(o + v, o[u]);
                }
            }
        }
    }
}

__global__ void backward(strideCSR CSR, int l, int *d, double *delta) {
    int vir_u = blockIdx.x * blockDim.x + threadIdx.x;

    if (vir_u < CSR.N) {
        int offset = CSR.offset[vir_u];
        int u = CSR.vmap[vir_u];
        int nvir = CSR.nvir[u];

        if (d[u] == l) {
            double sum = 0;
            for (int p = CSR.ptrs[u] + offset; p < CSR.ptrs[u + 1]; p += nvir) {
                int v = CSR.adjs[p];

                if (d[v] == l + 1) {
                    sum += delta[v];
                }
            }

            atomicAdd(delta + u, sum);
        }
    }
}

__global__ void update_delta(int N, double *delta, int *o, int *reach, int *vmap) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v < N) {
        delta[v] = 1. / o[v];
        delta[v] *= reach[vmap[v]];
    }
}

__global__ void update_bc(int N, int s, double *bc, double *delta, int *o, int *reach, int *vmap) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v < N) {
        if (v != s) {
            int org_v = vmap[v];
            bc[org_v] += (delta[v] * o[v] - 1) * reach[vmap[s]];
        }
    }
}