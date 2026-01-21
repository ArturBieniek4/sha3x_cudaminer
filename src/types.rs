use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct PoolJob {
    pub job_id: String,
    pub target: String,
    pub algo: String,
    pub height: u64,

    #[serde(default)]
    pub difficulty: Option<u64>,

    #[serde(default)]
    pub blob: Option<String>,

    #[serde(default)]
    pub xn: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MiningJob {
    pub job_id: String,
    pub mining_hash: [u8; 32],
    pub target_u64: u64,
    pub height: u64,
    pub xn: Option<u16>,
}

impl MiningJob {
    pub fn from_pool(job: PoolJob) -> Result<Self> {
        if job.algo.to_lowercase() != "sha3x" {
            bail!("unsupported algo: {}", job.algo);
        }

        let blob_hex = job.blob.clone().ok_or_else(|| anyhow!("job missing blob"))?;
        let blob = hex::decode(blob_hex).context("blob hex decode failed")?;
        if blob.len() != 32 {
            bail!("blob decoded length != 32 (got {})", blob.len());
        }
        let mut mining_hash = [0u8; 32];
        mining_hash.copy_from_slice(&blob);

        // Pool sends target as hex of 8 bytes little-endian.
        // If difficulty is provided, derive a target threshold from it.
        let target_u64 = if let Some(diff) = job.difficulty {
            if diff == 0 {
                u64::MAX
            } else {
                u64::MAX / diff
            }
        } else {
            let t = hex::decode(&job.target).context("target hex decode failed")?;
            if t.len() < 8 {
                bail!("target decoded length < 8 (got {})", t.len());
            }
            u64::from_le_bytes(t[0..8].try_into().unwrap())
        };

        let xn = if let Some(xn_hex) = job.xn {
            let b = hex::decode(xn_hex).context("xn hex decode failed")?;
            if b.len() != 2 {
                None
            } else {
                Some(u16::from_le_bytes([b[0], b[1]]))
            }
        } else {
            None
        };

        Ok(Self {
            job_id: job.job_id,
            mining_hash,
            target_u64,
            height: job.height,
            xn,
        })
    }

    pub fn target_difficulty_estimate(&self) -> u64 {
        if self.target_u64 == 0 {
            u64::MAX
        } else {
            u64::MAX / self.target_u64
        }
    }
}
