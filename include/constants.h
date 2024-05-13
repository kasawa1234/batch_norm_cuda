/*          PARAMETERS          */
// For sum
#define BLOCK_SIZE_BATCH 256        // only support batch_size <= 256 this time
#define BLOCK_SIZE_FEATURE 1

// For batch norm
#define BLOCK_SIZE_BN_X 16
#define BLOCK_SIZE_BN_Y 16

// parallel batch norm forward
#define BLOCK_SIZE_BN_HW 32
#define BLOCK_SIZE_BN_BATCH 16


#define EPSILON 1e-5
#define BLOCK_SIZE_DEFAULT 256


#define BLOCK_SIZE_HW 1024
#define BLOCK_SIZE_H 32
#define BLOCK_SIZE_W 32
