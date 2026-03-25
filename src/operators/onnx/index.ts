// Import all ONNX operator builders to register them in the registry
import './unary.js';
import './binary.js';
import './activation.js';
import './conv.js';
import './gemm.js';
import './normalization.js';
import './pool.js';
import './reduce.js';
import './common.js';
import './misc.js';
// Composite / advanced ops
import './lstm.js';
import './gru.js';
import './einsum.js';
import './mha.js';
import './gqa.js';
import './rotary.js';
import './matmul_nbits.js';
import './gather_block_quantized.js';
