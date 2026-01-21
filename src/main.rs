// src/main.rs
//
// - Stratum JSON-RPC: login, job, submit
// - Drives CUDA mining kernel
// - Live hashrate (from CUDA batch timing) + accepted/rejected counters
//

use anyhow::{Context, Result};
use clap::Parser;
use log::{info, warn};
use serde_json::json;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

mod gpu;
mod types;

use gpu::CudaMiner;
use types::{MiningJob, PoolJob};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    pool: String,

    #[arg(long)]
    wallet: String,

    #[arg(long, default_value = "worker1")]
    worker: String,

    #[arg(long, default_value_t = 0)]
    device: i32,

    /// 0 = auto (sm_count * 32)
    #[arg(long, default_value_t = 0)]
    blocks: i32,

    /// 0 = auto (256)
    #[arg(long, default_value_t = 0)]
    threads: i32,

    /// Work per thread per kernel launch (latency vs overhead trade)
    #[arg(long, default_value_t = 2048)]
    iters: u32,
}

#[derive(Debug, Clone)]
struct Share {
    job_id: String,
    nonce: u64,
    hash: [u8; 32],
}

fn user_agent() -> String {
    format!(
        "sha3x-cudaminer {} ({})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS
    )
}

async fn send_json(
    writer: &Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>,
    v: serde_json::Value,
) -> Result<()> {
    let mut w = writer.lock().await;
    let s = serde_json::to_string(&v)?;
    w.write_all(s.as_bytes()).await?;
    w.write_all(b"\n").await?;
    w.flush().await?;
    Ok(())
}

fn share_accepted(v: &serde_json::Value) -> bool {
    if let Some(err) = v.get("error") {
        if !err.is_null() {
            return false;
        }
    }
    match v.get("result") {
        Some(r) if r.is_null() => true,
        Some(r) if r.is_boolean() => r.as_bool().unwrap_or(false),
        Some(r) if r.is_object() => {
            if let Some(status) = r.get("status").and_then(|s| s.as_str()) {
                let s = status.to_ascii_lowercase();
                s == "ok" || s == "accepted"
            } else {
                true
            }
        }
        _ => false,
    }
}

/// Quick difficulty estimate from leading BE u64 of the 32-byte hash.
fn difficulty_from_hash_leading_u64_be(hash: &[u8; 32]) -> u64 {
    let lead = u64::from_be_bytes(hash[0..8].try_into().unwrap());
    if lead == 0 {
        u64::MAX
    } else {
        u64::MAX / lead
    }
}

fn gpu_thread(
    mut cuda: CudaMiner,
    current_job: Arc<Mutex<Option<MiningJob>>>,
    share_tx: tokio::sync::mpsc::UnboundedSender<Share>,
    hashes_total: Arc<AtomicU64>,
    shares_found: Arc<AtomicU64>,
    shares_accepted: Arc<AtomicU64>,
    shares_rejected: Arc<AtomicU64>,
    iters_per_thread: u32,
) -> Result<()> {
    let mut active_job_id = String::new();
    let mut nonce_base: u64 = 0;
    let mut last_log = Instant::now();
    let start_time = Instant::now();
    let mut last_batch_hashes_per_ms = 0.0f64;

    loop {
        let job = { current_job.lock().unwrap().clone() };
        let Some(job) = job else {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        };

        if job.job_id != active_job_id {
            active_job_id = job.job_id.clone();

            cuda.set_job(&job.mining_hash, job.target_u64, job.xn)
                .context("cuda.set_job failed")?;

            info!(
                "NEW JOB id={} height={} target_u64=0x{:016x} diff≈{} xn={:?}",
                job.job_id,
                job.height,
                job.target_u64,
                job.target_difficulty_estimate(),
                job.xn
            );
        }

        let r = cuda
            .run_batch(nonce_base, iters_per_thread)
            .context("cuda.run_batch failed")?;

        let mut add_hashes = r.hashes_done;
        if r.found != 0 {
            if last_batch_hashes_per_ms > 0.0 {
                add_hashes = (r.elapsed_ms as f64 * last_batch_hashes_per_ms) as u64;
            }
        } else if r.elapsed_ms > 0.0 {
            last_batch_hashes_per_ms = (r.hashes_done as f64) / (r.elapsed_ms as f64);
        }
        hashes_total.fetch_add(add_hashes, Ordering::Relaxed);

        // advance nonce space
        nonce_base = nonce_base.wrapping_add(add_hashes);

        if r.found != 0 {
            shares_found.fetch_add(1, Ordering::Relaxed);

            let diff = difficulty_from_hash_leading_u64_be(&r.hash);
            info!(
                "💎 FOUND SHARE job={} nonce_le={} diff={} (target≈{})",
                active_job_id,
                hex::encode(r.nonce.to_le_bytes()),
                diff,
                job.target_difficulty_estimate()
            );

            let _ = share_tx.send(Share {
                job_id: active_job_id.clone(),
                nonce: r.nonce,
                hash: r.hash,
            });
        }

        if last_log.elapsed() >= Duration::from_secs(1) {
            last_log = Instant::now();
            let total = hashes_total.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
            let rate_mhs = (total as f64 / elapsed) / 1e6;
            let ok = shares_accepted.load(Ordering::Relaxed);
            let bad = shares_rejected.load(Ordering::Relaxed);
            let found = shares_found.load(Ordering::Relaxed);

            info!(
                "hashrate≈{:.2} MH/s | total_hashes={} | found={} | acc/rej={}/{}",
                rate_mhs,
                total,
                found,
                ok,
                bad
            );
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let current_job: Arc<Mutex<Option<MiningJob>>> = Arc::new(Mutex::new(None));
    let session_id: Arc<tokio::sync::Mutex<Option<String>>> =
        Arc::new(tokio::sync::Mutex::new(None));

    let hashes_total = Arc::new(AtomicU64::new(0));
    let shares_found = Arc::new(AtomicU64::new(0));
    let shares_accepted = Arc::new(AtomicU64::new(0));
    let shares_rejected = Arc::new(AtomicU64::new(0));
    let submit_id = Arc::new(AtomicU64::new(10));

    let (share_tx, mut share_rx) = tokio::sync::mpsc::unbounded_channel::<Share>();

    // CUDA init
    let cuda = CudaMiner::init(args.device, args.blocks, args.threads)
        .context("CudaMiner::init failed")?;
    info!(
        "CUDA ready: blocks={} threads={} iters={}",
        cuda.blocks(),
        cuda.threads(),
        args.iters
    );

    // Start GPU mining thread
    {
        let current_job = current_job.clone();
        let share_tx = share_tx.clone();
        let hashes_total = hashes_total.clone();
        let shares_found = shares_found.clone();
        let shares_accepted = shares_accepted.clone();
        let shares_rejected = shares_rejected.clone();
        let iters = args.iters;

        std::thread::spawn(move || {
            if let Err(e) = gpu_thread(
                cuda,
                current_job,
                share_tx,
                hashes_total,
                shares_found,
                shares_accepted,
                shares_rejected,
                iters,
            ) {
                eprintln!("gpu thread died: {:#}", e);
            }
        });
    }

    let stream = TcpStream::connect(&args.pool)
        .await
        .with_context(|| format!("connect failed: {}", args.pool))?;
    stream.set_nodelay(true)?;
    info!("Connected to pool {}", args.pool);

    let (read_half, write_half) = stream.into_split();
    let writer = Arc::new(tokio::sync::Mutex::new(write_half));

    let login = json!({
        "id": 1,
        "jsonrpc": "2.0",
        "method": "login",
        "params": {
            "login": args.wallet,
            "pass": args.worker,
            "agent": user_agent(),
            "algo": ["sha3x"]
        }
    });
    send_json(&writer, login).await?;
    info!("Login sent");

    {
        let writer_submit = writer.clone();
        let session_id_submit = session_id.clone();
        let submit_id_submit = submit_id.clone();
        let shares_accepted_submit = shares_accepted.clone();
        let shares_rejected_submit = shares_rejected.clone();

        tokio::spawn(async move {
            while let Some(share) = share_rx.recv().await {
                let sid = {
                    let g = session_id_submit.lock().await;
                    g.clone()
                };

                let Some(sid) = sid else {
                    warn!("No session id yet, dropping share");
                    continue;
                };

                let id = submit_id_submit.fetch_add(1, Ordering::Relaxed);
                let nonce_hex = hex::encode(share.nonce.to_le_bytes());
                let result_hex = hex::encode(share.hash);

                let submit = json!({
                    "id": id,
                    "jsonrpc": "2.0",
                    "method": "submit",
                    "params": {
                        "id": sid,
                        "job_id": share.job_id,
                        "nonce": nonce_hex,
                        "result": result_hex
                    }
                });

                if let Err(e) = send_json(&writer_submit, submit).await {
                    warn!("submit write failed: {}", e);
                    continue;
                }

                info!("Submitted share (id={})", id);

                let _ = (&shares_accepted_submit, &shares_rejected_submit); // keep clones used
            }
        });
    }

    let mut lines = BufReader::new(read_half).lines();
    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                warn!("bad json: {} | {}", e, line);
                continue;
            }
        };

        if let Some(method) = v.get("method").and_then(|m| m.as_str()) {
            if method == "job" {
                if let Some(params) = v.get("params") {
                    match serde_json::from_value::<PoolJob>(params.clone())
                        .ok()
                        .and_then(|pj| MiningJob::from_pool(pj).ok())
                    {
                        Some(job) => {
                            *current_job.lock().unwrap() = Some(job);
                        }
                        None => warn!("failed to parse job params"),
                    }
                }
                continue;
            }
        }

        // responses
        if let Some(id) = v.get("id").and_then(|x| x.as_u64()) {
            if id == 1 {
                // login response
                if let Some(result) = v.get("result") {
                    if let Some(sid) = result.get("id").and_then(|x| x.as_str()) {
                        *session_id.lock().await = Some(sid.to_string());
                        info!("Login OK, session id set");
                    } else {
                        warn!("login response missing session id");
                    }

                    if let Some(jobv) = result.get("job") {
                        if let Ok(pj) = serde_json::from_value::<PoolJob>(jobv.clone()) {
                            if let Ok(job) = MiningJob::from_pool(pj) {
                                *current_job.lock().unwrap() = Some(job);
                            }
                        }
                    }
                } else {
                    warn!("login response missing result");
                }
            } else {
                // share response
                if share_accepted(&v) {
                    shares_accepted.fetch_add(1, Ordering::Relaxed);
                    info!("✅ share accepted (id={})", id);
                } else {
                    shares_rejected.fetch_add(1, Ordering::Relaxed);
                    warn!("❌ share rejected (id={})", id);
                }
            }
        }
    }

    Ok(())
}
