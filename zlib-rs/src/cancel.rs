//! Cooperative cancellation for long-running (de)compression.

/// A cooperative cancellation check, polled periodically while decompressing.
///
/// Pass one to [`Inflate::decompress_with_cancel`](crate::Inflate::decompress_with_cancel).
/// It is polled between internal steps of a single `decompress` call, so it
/// bounds the time spent on adversarial input — for example a run of empty
/// stored blocks (tiny output, huge input) on which an output-size check would
/// never fire. When it fires, `decompress_with_cancel` returns
/// `Err(`[`CancelledOr::Cancelled`](crate::CancelledOr::Cancelled)`)`.
///
/// Any `Fn() -> bool` that is `Send + Sync` implements `CancelCheck`, so the common
/// case is a closure over an `AtomicBool` or a deadline. Use [`NeverCancel`]
/// (the default) when no cancellation is wanted; it is a zero-cost no-op.
///
/// This trait is intentionally minimal and dependency-free; its shape is
/// modelled on the `enough` crate's `Stop` trait.
pub trait CancelCheck: Send + Sync {
    /// Returns `true` to cancel the operation as soon as possible.
    ///
    /// Polled at coarse intervals, so it may be called many times during one
    /// operation — keep it cheap.
    fn is_cancelled(&self) -> bool;

    /// Returns `false` if this check can never fire, letting hot
    /// loops skip the check entirely. The default is `true`.
    #[inline]
    fn may_cancel(&self) -> bool {
        true
    }
}

/// A [`CancelCheck`] that never cancels: a zero-cost opt-out of cooperative cancellation.
///
/// Because [`may_cancel`](CancelCheck::may_cancel) returns `false`, the decoder elides the
/// cancellation check entirely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct NeverCancel;

impl CancelCheck for NeverCancel {
    #[inline(always)]
    fn is_cancelled(&self) -> bool {
        false
    }

    #[inline(always)]
    fn may_cancel(&self) -> bool {
        false
    }
}

impl<F: Fn() -> bool + Send + Sync> CancelCheck for F {
    #[inline]
    fn is_cancelled(&self) -> bool {
        self()
    }
}

/// The error type of the `*_with_cancel` methods: either the caller's
/// [`CancelCheck`] asked to stop, or the operation itself failed with `E`.
///
/// The plain methods (`decompress`, `compress`, …) keep their original
/// `Result<_, InflateError>` / `Result<_, DeflateError>` signatures; only the
/// `*_with_cancel` variants carry the `Cancelled` case, so cancellation never
/// touches the existing error enums.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum CancelledOr<E> {
    /// The [`CancelCheck`] returned `true`; the stream is left resumable.
    Cancelled,
    /// The operation failed with this error.
    Error(E),
}

impl<E> CancelledOr<E> {
    /// Returns `true` if this is [`CancelledOr::Cancelled`].
    pub fn is_cancelled(&self) -> bool {
        matches!(self, CancelledOr::Cancelled)
    }

    /// Returns the inner error, or `None` if this is [`CancelledOr::Cancelled`].
    pub fn error(self) -> Option<E> {
        match self {
            CancelledOr::Cancelled => None,
            CancelledOr::Error(error) => Some(error),
        }
    }
}

impl<E> From<E> for CancelledOr<E> {
    fn from(error: E) -> Self {
        CancelledOr::Error(error)
    }
}

/// Iterations of a hot (de)compression loop between real polls of the cancel
/// check.
///
/// The loops poll through a [`Debounced`]: every iteration it only decrements a
/// counter (one predicted branch), and just every `CANCEL_POLL_INTERVAL`-th
/// iteration makes the indirect call to the user's check. So the per-iteration
/// cost is the counter — not the closure — and that counter stands in for what
/// would otherwise be a vtable call on every iteration. Cancellation latency is
/// bounded to this many iterations of work, brief on any strategy.
pub(crate) const CANCEL_POLL_INTERVAL: usize = 8192;

/// A stack-local throttle over a [`CancelCheck`]: forwards to the underlying check
/// only every `interval`-th call (returning `false` in between), so a hot loop
/// can poll once per iteration with a plain `if cancel.is_cancelled()` while the
/// user's closure runs only periodically.
///
/// When the underlying check can never fire (the default — no cancel set), the
/// inner handle is `None`, so polling is a single predicted branch and the
/// closure is never touched.
pub(crate) struct Debounced<'a> {
    check: Option<&'a dyn CancelCheck>,
    interval: usize,
    countdown: usize,
}

impl<'a> Debounced<'a> {
    #[inline]
    pub(crate) fn new(cancel: &'a dyn CancelCheck, interval: usize) -> Debounced<'a> {
        Debounced {
            check: if cancel.may_cancel() {
                Some(cancel)
            } else {
                None
            },
            interval: interval.max(1),
            countdown: 0,
        }
    }

    /// Returns `true` if the operation should cancel. Consults the real check on
    /// the first call and then every `interval`-th call; returns `false`
    /// (cheaply) in between.
    #[inline]
    pub(crate) fn is_cancelled(&mut self) -> bool {
        let check = match self.check {
            Some(check) => check,
            None => return false,
        };
        if self.countdown == 0 {
            self.countdown = self.interval - 1;
            check.is_cancelled()
        } else {
            self.countdown -= 1;
            false
        }
    }
}
