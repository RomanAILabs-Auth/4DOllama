// bench_4d.rs — same scalar work as bench_4d.py (timed).
//
// Build & run:
//   rustc examples/bench_4d.rs -O -o bench_4d_rust && ./bench_4d_rust
//   rustc examples/bench_4d.rs -O -o bench_4d_rust.exe && .\bench_4d_rust.exe
//
// Windows — pick ONE toolchain:
//   • MSVC (default rustup host): needs Visual Studio Build Tools + C++ (link.exe).
//   • GNU / MinGW (same family as `r4d` on Windows): install the stdlib for that target, then build:
//       rustup target add x86_64-pc-windows-gnu
//       rustc examples/bench_4d.rs -O -o bench_4d_rust.exe --target x86_64-pc-windows-gnu
//     Error "can't find crate for `std`" (E0463) means this target is not installed yet.
//
// Or from repo root:  .\examples\run_bench_4d.ps1

use std::time::Instant;

const N: usize = 5_000_000;

fn main() {
    let t0 = Instant::now();
    let mut s = 0.0_f64;
    for i in 0..N {
        let x = i as f64 * 1e-7;
        s += x * x + (1.0 - x) * 0.5;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("bench_4d rust: N={N} sum={s:.6} wall_s={dt:.4}");
}
