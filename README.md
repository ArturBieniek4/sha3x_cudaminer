# SHA3x CUDA Miner

A simple and fast GPU miner for **Tari(XTM)** using **SHA3x** written in **CUDA + Rust**.

> **SHA3x** here means:  
> `SHA3-256(SHA3-256(SHA3-256(msg)))`

Achieving up to **700MH/s** on RTX 4070 Super, vs:
 - **655MH/s** on [Graxil](https://github.com/tari-project/graxil)
 - **14.8MH/s** on R9 7900X 24 logical cores using Rust SHA3 library
 - **33MH/s** expected performance on **DE1-SoC FPGA** - a project coming soon
---

## What it computes

For each nonce, the miner hashes:

- First hash input: **41 bytes**
  - `nonce (8B, little-endian)`
  - `mining_hash (32B)`
  - trailing byte `0x01`
- Next two hashes: **32 bytes** (the previous digest)

Padding is standard **NIST SHA3** domain separation for SHA3-256:
- XOR `0x06` after the message,
- XOR `0x80` at the last byte of the rate block.

Since inputs are 41B and 32B, they always fit into **one SHA3-256 rate block (136B)** (no multi-block sponge streaming).

---

## Architecture

### CUDA backend
- Keccak state is kept in registers as `uint64_t a0..a24`
- Job constants (`mining_hash`, target difficulty, optional XN) live in `__constant__` memory
- Batched kernel launches: each launch tests a large chunk of nonces for stable performance
- Result is written only on success to a **pinned + mapped** host buffer (zero per-batch memcpy)
- Live hashrate computed from `cudaEventElapsedTime()` using `hashes_done / elapsed_time`

### Rust frontend
- Connects to the pool
- Receives job updates (difficulty/target + `mining_hash`)
- Updates GPU constants through a small C ABI
- Submits found shares back to the pool
- Prints basic stats: hashrate + accepted/rejected

---

## Build

### Requirements
- NVIDIA GPU (tested on Ada / RTX 4070 Super)
- CUDA Toolkit (nvcc)
- Rust stable toolchain (`cargo`)
- Linux recommended (Arch works great)

### Running (release)
```bash
RUST_LOG=info cargo run --release -- --pool pool.sha3x.supportxtm.com:6118 --wallet 124EwiFUuFPK4cKGhxdVrmghaaunawwysM356wEZrd224EPkfCZ7uFYm4otVYMZZ67RoAqs2srwUMyGfiKqhDmzpD8g --worker cudaminer