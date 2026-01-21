use anyhow::{bail, Result};

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct Sha3xBatchResult {
    pub found: u32,
    pub _pad0: u32,
    pub nonce: u64,
    pub hash: [u8; 32],
    pub hashes_done: u64,
    pub elapsed_ms: f32,
    pub _pad1: u32,
}

extern "C" {
    fn sha3x_cuda_init(
        device: i32,
        requested_blocks: i32,
        requested_threads: i32,
        out_blocks: *mut i32,
        out_threads: *mut i32,
    ) -> i32;

    fn sha3x_cuda_set_job(mining_hash32: *const u8, target_u64: u64, xn: u16, xn_enabled: i32)
        -> i32;

    fn sha3x_cuda_run_batch(start: u64, iters_per_thread: u32, out: *mut Sha3xBatchResult)
        -> i32;

    fn sha3x_cuda_shutdown();
}

pub struct CudaMiner {
    blocks: i32,
    threads: i32,
}

impl CudaMiner {
    pub fn init(device: i32, requested_blocks: i32, requested_threads: i32) -> Result<Self> {
        let mut blocks = 0i32;
        let mut threads = 0i32;

        let rc = unsafe {
            sha3x_cuda_init(
                device,
                requested_blocks,
                requested_threads,
                &mut blocks as *mut i32,
                &mut threads as *mut i32,
            )
        };
        if rc != 0 {
            bail!("sha3x_cuda_init failed (rc={})", rc);
        }

        Ok(Self { blocks, threads })
    }

    pub fn blocks(&self) -> i32 {
        self.blocks
    }
    pub fn threads(&self) -> i32 {
        self.threads
    }

    pub fn set_job(&mut self, mining_hash: &[u8; 32], target_u64: u64, xn: Option<u16>) -> Result<()> {
        let (xn_val, xn_enabled) = match xn {
            Some(v) => (v, 1),
            None => (0u16, 0),
        };

        let rc = unsafe {
            sha3x_cuda_set_job(mining_hash.as_ptr(), target_u64, xn_val, xn_enabled)
        };
        if rc != 0 {
            bail!("sha3x_cuda_set_job failed (rc={})", rc);
        }
        Ok(())
    }

    /// `start` is:
    /// - normal mode: starting nonce
    /// - xn mode: starting 48-bit counter (nonce = (counter<<16)|xn)
    pub fn run_batch(&mut self, start: u64, iters_per_thread: u32) -> Result<Sha3xBatchResult> {
        let mut out = Sha3xBatchResult {
            found: 0,
            _pad0: 0,
            nonce: 0,
            hash: [0u8; 32],
            hashes_done: 0,
            elapsed_ms: 0.0,
            _pad1: 0,
        };

        let rc = unsafe { sha3x_cuda_run_batch(start, iters_per_thread, &mut out as *mut _) };
        if rc != 0 {
            bail!("sha3x_cuda_run_batch failed (rc={})", rc);
        }

        Ok(out)
    }
}

impl Drop for CudaMiner {
    fn drop(&mut self) {
        unsafe { sha3x_cuda_shutdown() };
    }
}
