use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=cuda/sha3x_cuda.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();

    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| {
            if target.contains("windows") {
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let obj = if target.contains("windows") {
        out_dir.join("sha3x_cuda.obj")
    } else {
        out_dir.join("sha3x_cuda.o")
    };

    // Tune for RTX 4070 Super (Ada = sm_89). If you want portability, add more gencodes.
    let mut cmd = Command::new(&nvcc);
    cmd.arg("-O3")
        .arg("-std=c++17")
        .arg("--use_fast_math")
        .arg("-arch=sm_89")
        .arg("-c")
        .arg("cuda/sha3x_cuda.cu")
        .arg("-o")
        .arg(&obj);

    let status = cmd.status().expect("failed to run nvcc");
    if !status.success() {
        panic!("nvcc compile failed");
    }

    // Link CUDA runtime
    if target.contains("windows") {
        let lib_dir = PathBuf::from(cuda_home).join("lib").join("x64");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    } else {
        let lib_dir = PathBuf::from(cuda_home).join("lib64");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }

    // Link the object directly
    println!("cargo:rustc-link-arg={}", obj.display());

    println!("cargo:rustc-link-lib=cudart");

    // If nvcc used a C++ compiler, you may need stdc++ on non-Windows targets.
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

