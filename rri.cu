#define idx(i,j,lda) ( (j) + ((i)*(lda)) )

#define K %d
#define NPTS_PER_BLOCK %d
__global__ void get_rri_feature(float *pc, int32_t N, int32_t *k_idx, float *feat)
{
    __shared__ float T_p[NPTS_PER_BLOCK][K][3]; // thread block = (N=64,K=16,1)
 
    unsigned int tid = threadIdx.x;
    unsigned int k = threadIdx.y; // k-nn index 0.. K-1
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x; // point id == 0.. N-1
    
    if(id >= N) return;

    float px = pc[idx(id, 0, 3)];
    float py = pc[idx(id, 1, 3)];
    float pz = pc[idx(id, 2, 3)];

    float p_len = sqrtf(px*px + py*py + pz*pz);
    float pxn = px/p_len;
    float pyn = py/p_len;
    float pzn = pz/p_len;

    int32_t _k_idx = k_idx[idx(id, k, K)];

    float rx = pc[idx(_k_idx, 0, 3)];
    float ry = pc[idx(_k_idx, 1, 3)];
    float rz = pc[idx(_k_idx, 2, 3)];
    
    float r_len = sqrtf(rx*rx + ry*ry + rz*rz);
    float dot_ik = (pxn*rx + pyn*ry + pzn*rz)/r_len;
    
    float minPsi = 9e9f;
    float sin_psi, cos_psi, psi = 0.0f;
    float cx, cy, cz;
    bool j_eq_k;    
    
    dot_ik = max(-1.0f, min(dot_ik, 1.0f));
   
    T_p[tid][k][0] = rx - dot_ik*px;
    T_p[tid][k][1] = ry - dot_ik*py;
    T_p[tid][k][2] = rz - dot_ik*pz;
     
    __syncthreads();

    for(int j = 0; j < K; j++)
    {
        j_eq_k = j == k;
    
        cx = T_p[tid][k][1] * T_p[tid][j][2] - T_p[tid][j][1] * T_p[tid][k][2];
        cy = T_p[tid][j][0] * T_p[tid][k][2] - T_p[tid][k][0] * T_p[tid][j][2];
        cz = T_p[tid][k][0] * T_p[tid][j][1] - T_p[tid][j][0] * T_p[tid][k][1];
        
        sin_psi = cx*pxn + cy*pyn + cz*pzn;
     
        cos_psi = T_p[tid][k][0]*T_p[tid][j][0] + T_p[tid][k][1]*T_p[tid][j][1] + T_p[tid][k][2]*T_p[tid][j][2];

        psi = atan2(sin_psi, cos_psi);
        psi = psi + (psi < 0)*2*3.14159265f;
        if(psi < minPsi && !j_eq_k)
            minPsi = psi;
    }

    feat[idx(idx(id, k, K), 0, 4)] = p_len;
    feat[idx(idx(id, k, K), 1, 4)] = r_len;
    feat[idx(idx(id, k, K), 2, 4)] = acos(dot_ik);
    feat[idx(idx(id, k, K), 3, 4)] = minPsi;
}
