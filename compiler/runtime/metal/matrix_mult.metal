#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 32;
constant int VECTOR_SIZE = 4;

struct WorkChunkInfo {
    int startRow;
    int endRow;
    int startCol;
    int endCol;
    int size;
};

kernel void matrix_multiply(
    device const int* matrixA [[buffer(0)]],
    device const int* matrixB [[buffer(1)]],
    device int* result [[buffer(2)]],
    constant WorkChunkInfo* chunk [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    int matrixSize = chunk->size;
    threadgroup int tileA[TILE_SIZE][TILE_SIZE];
    threadgroup int tileB[TILE_SIZE][TILE_SIZE];
    int accum[VECTOR_SIZE][VECTOR_SIZE];
    for (int i = 0; i < VECTOR_SIZE; i++) {
        for (int j = 0; j < VECTOR_SIZE; j++) {
            accum[i][j] = 0;
        }
    }
    int blockRowOffset = chunk->startRow + gid.x * TILE_SIZE;
    int blockColOffset = chunk->startCol + gid.y * TILE_SIZE;
    int numTiles = (matrixSize + TILE_SIZE - 1) / TILE_SIZE;
    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        int tileOffset = tileIdx * TILE_SIZE;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            for (int j = 0; j < VECTOR_SIZE; j++) {
                int localRowA = lid.x * VECTOR_SIZE + i;
                int localColA = lid.y * VECTOR_SIZE + j;
                if (localRowA < TILE_SIZE && localColA < TILE_SIZE) {
                    int globalRowA = blockRowOffset + localRowA;
                    int globalColA = tileOffset + localColA;
                    if (globalRowA < chunk->endRow && globalColA < matrixSize) {
                        tileA[localRowA][localColA] = matrixA[globalRowA * matrixSize + globalColA];
                    } else {
                        tileA[localRowA][localColA] = 0;
                    }
                    int globalRowB = tileOffset + localRowA;
                    int globalColB = blockColOffset + localColA;
                    if (globalRowB < matrixSize && globalColB < chunk->endCol) {
                        tileB[localRowA][localColA] = matrixB[globalRowB * matrixSize + globalColB];
                    } else {
                        tileB[localRowA][localColA] = 0;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < VECTOR_SIZE; i++) {
            int localRow = lid.x * VECTOR_SIZE + i;
            if (localRow >= TILE_SIZE) continue;
            int globalRow = blockRowOffset + localRow;
            if (globalRow >= chunk->endRow) continue;
            for (int j = 0; j < VECTOR_SIZE; j++) {
                int localCol = lid.y * VECTOR_SIZE + j;
                if (localCol >= TILE_SIZE) continue;
                int globalCol = blockColOffset + localCol;
                if (globalCol >= chunk->endCol) continue;
                int sum = 0;
                int k = 0;
                while (k + 7 < TILE_SIZE) {
                    sum += tileA[localRow][k] * tileB[k][localCol];
                    sum += tileA[localRow][k+1] * tileB[k+1][localCol];
                    sum += tileA[localRow][k+2] * tileB[k+2][localCol];
                    sum += tileA[localRow][k+3] * tileB[k+3][localCol];
                    sum += tileA[localRow][k+4] * tileB[k+4][localCol];
                    sum += tileA[localRow][k+5] * tileB[k+5][localCol];
                    sum += tileA[localRow][k+6] * tileB[k+6][localCol];
                    sum += tileA[localRow][k+7] * tileB[k+7][localCol];
                    k += 8;
                }
                for (; k < TILE_SIZE; k++) {
                    sum += tileA[localRow][k] * tileB[k][localCol];
                }
                accum[i][j] += sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < VECTOR_SIZE; i++) {
        int globalRow = blockRowOffset + lid.x * VECTOR_SIZE + i;
        if (globalRow >= chunk->endRow) continue;
        for (int j = 0; j < VECTOR_SIZE; j++) {
            int globalCol = blockColOffset + lid.y * VECTOR_SIZE + j;
            if (globalCol >= chunk->endCol) continue;
            result[globalRow * matrixSize + globalCol] = accum[i][j];
        }
    }
}