//! Cooperative cancellation of `Inflate::decompress_with_cancel`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use zlib_rs::{CancelledOr, Deflate, DeflateFlush, Inflate, InflateFlush, NeverCancel, Status};

/// A run of `n` empty stored DEFLATE blocks followed by a final empty block.
/// Decodes to zero bytes but forces the inflate state machine to chew through
/// `5 * (n + 1)` input bytes — the "tiny output, huge input" shape on which an
/// output-size check would never fire.
fn empty_stored_block_bomb(n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(5 * (n + 1));
    for _ in 0..n {
        v.extend_from_slice(&[0x00, 0x00, 0x00, 0xFF, 0xFF]); // non-final stored, len 0
    }
    v.extend_from_slice(&[0x01, 0x00, 0x00, 0xFF, 0xFF]); // final stored, len 0
    v
}

/// A cancel that returns `true` once it has been polled more than `after` times,
/// together with the shared poll counter so the test can inspect it.
fn stop_after(after: usize) -> (impl Fn() -> bool + Send + Sync, Arc<AtomicUsize>) {
    let polls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&polls);
    (
        move || polls.fetch_add(1, Ordering::Relaxed) >= after,
        counter,
    )
}

#[test]
fn zero_output_bomb_is_cancellable() {
    let bomb = empty_stored_block_bomb(1_000_000); // ~5 MB input, 0 output
    let mut inflate = Inflate::new(false, 15);
    let mut out = [0u8; 64];
    let always_stop = || true;
    let err = inflate
        .decompress_with_cancel(&bomb, &mut out, InflateFlush::Finish, &always_stop)
        .unwrap_err();
    assert_eq!(err, CancelledOr::Cancelled);
    assert!(err.is_cancelled());
}

#[test]
fn bomb_is_cancelled_partway() {
    let bomb = empty_stored_block_bomb(1_000_000);
    let (cancel, polls) = stop_after(8);
    let mut inflate = Inflate::new(false, 15);
    let mut out = [0u8; 64];
    let err = inflate
        .decompress_with_cancel(&bomb, &mut out, InflateFlush::Finish, &cancel)
        .unwrap_err();
    assert_eq!(err, CancelledOr::Cancelled);
    // The check is polled per state-machine step and fired after a handful of
    // them, long before the millions of input bytes were consumed.
    assert_eq!(polls.load(Ordering::Relaxed), 9);
}

#[test]
fn never_firing_stop_decodes_to_completion() {
    let bomb = empty_stored_block_bomb(1000); // small: decode it fully
    let mut inflate = Inflate::new(false, 15);
    let mut out = [0u8; 64];
    let never = || false;
    let status = inflate
        .decompress_with_cancel(&bomb, &mut out, InflateFlush::Finish, &never)
        .unwrap();
    assert_eq!(status, Status::StreamEnd);
}

#[test]
fn unstoppable_matches_plain_decompress() {
    let bomb = empty_stored_block_bomb(1000);

    let mut a = Inflate::new(false, 15);
    let mut out_a = [0u8; 64];
    let status_a = a
        .decompress_with_cancel(&bomb, &mut out_a, InflateFlush::Finish, &NeverCancel)
        .unwrap();

    let mut b = Inflate::new(false, 15);
    let mut out_b = [0u8; 64];
    let status_b = b
        .decompress(&bomb, &mut out_b, InflateFlush::Finish)
        .unwrap();

    assert_eq!(status_a, Status::StreamEnd);
    assert_eq!(status_a, status_b);
}

#[test]
fn round_trip_with_never_firing_stop() {
    let payload = b"the quick brown fox jumps over the lazy dog".repeat(50);

    // Compress to a raw deflate stream.
    let mut comp = Deflate::new(6, false, 15);
    let mut cbuf = vec![0u8; payload.len() + 1024];
    let status = comp
        .compress(&payload, &mut cbuf, DeflateFlush::Finish)
        .unwrap();
    assert_eq!(status, Status::StreamEnd);
    let clen = comp.total_out() as usize;

    // Decompress it with a never-firing cancel -> identical bytes back.
    let mut inflate = Inflate::new(false, 15);
    let mut decoded = vec![0u8; payload.len()];
    let never = || false;
    let status = inflate
        .decompress_with_cancel(&cbuf[..clen], &mut decoded, InflateFlush::Finish, &never)
        .unwrap();
    assert_eq!(status, Status::StreamEnd);
    assert_eq!(decoded, payload);
}

#[test]
fn compress_is_cancellable() {
    let payload = vec![0u8; 1 << 20]; // 1 MiB
    let mut comp = Deflate::new(6, false, 15);
    let mut out = vec![0u8; 4096];
    let always = || true;
    let err = comp
        .compress_with_cancel(&payload, &mut out, DeflateFlush::Finish, &always)
        .unwrap_err();
    assert_eq!(err, CancelledOr::Cancelled);
    assert!(err.is_cancelled());
}

#[test]
fn compress_is_cancelled_partway() {
    // Pseudo-random (≈incompressible) and large, so the debounced check is
    // polled many times before the compression could finish.
    let payload: Vec<u8> = (0..(1u32 << 22))
        .map(|i| i.wrapping_mul(2654435761) as u8)
        .collect();
    // Fires on the 4th debounced poll — early, long before the input is consumed.
    let (cancel, polls) = stop_after(3);
    let mut comp = Deflate::new(9, false, 15);
    let mut out = vec![0u8; payload.len() + 1024];
    let err = comp
        .compress_with_cancel(&payload, &mut out, DeflateFlush::Finish, &cancel)
        .unwrap_err();
    assert_eq!(err, CancelledOr::Cancelled);
    // The check was polled repeatedly inside the strategy loop and fired partway.
    assert!(polls.load(Ordering::Relaxed) > 1);
}

#[test]
fn compress_unstoppable_round_trips() {
    let payload = b"the quick brown fox jumps over the lazy dog".repeat(50);

    let mut comp = Deflate::new(6, false, 15);
    let mut cbuf = vec![0u8; payload.len() + 1024];
    let status = comp
        .compress_with_cancel(&payload, &mut cbuf, DeflateFlush::Finish, &NeverCancel)
        .unwrap();
    assert_eq!(status, Status::StreamEnd);
    let clen = comp.total_out() as usize;

    let mut inflate = Inflate::new(false, 15);
    let mut decoded = vec![0u8; payload.len()];
    let status = inflate
        .decompress(&cbuf[..clen], &mut decoded, InflateFlush::Finish)
        .unwrap();
    assert_eq!(status, Status::StreamEnd);
    assert_eq!(decoded, payload);
}
