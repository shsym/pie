//! **Saved-context resume on a GDN/hybrid model — KV pages AND the folded
//! recurrent state.**
//!
//! A hybrid context is two things: KV pages (`kv-working-set.update-index`)
//! and a folded recurrent state. Until `rs-working-set.update-index`, a saved
//! context carried only the first, and a guest reopening one had to refuse —
//! silently pairing reopened KV with a fresh recurrent state corrupts the
//! sequence rather than resuming it. This inferlet drives the full contract:
//!
//!  1. **save** — prefill with the in-forward fold, then index BOTH halves
//!     under one key, the RS half tagged with the committed token count.
//!  2. **resume** — reopen both at the exact boundary and decode one token.
//!     The resumed RS shares the snapshot's fold copy-on-write, so the saver
//!     also keeps decoding — both must run.
//!  3. **refuse** — `from-index` at any OTHER committed length is the named
//!     boundary refusal (a fold cannot be rewound), and a missing key is a
//!     typed `None`, never an error.
//!  4. **remove** — dropping the snapshot answers `true` once, `false` after.
//!
//! On the mock driver the logits are synthetic, so this asserts MECHANICS
//! (binds, counts, refusals), not state numerics — those are the device
//! (CUDA) half of the verification.

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};

const KEY: &[u8] = b"gdn-save-resume/ctx";

/// One in-forward-fold fire of `toks` at absolute position `pos`, greedy
/// argmax back. The same single-fire shape as `generate-gdn`'s prefill.
async fn fire(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    toks: &[u32],
    pos: u32,
    max_pages: u32,
    pipe: &Pipeline,
    tag: &str,
) -> Result<i32> {
    let t = toks.len() as u32;
    let end = pos + t;
    let ps = kv_page_size();
    let ch = |v: Vec<u32>| Channel::from(v);

    let fwd = ForwardPass::new();
    fwd.embed(
        &Channel::from_iter(toks.iter().map(|&x| x as i32)),
        &ch(vec![0, t]),
    )?;
    let kv_len = ch(vec![end]);
    let pages = ch((0..max_pages).collect());
    let page_indptr = ch(vec![0, end.div_ceil(ps)]);
    let w_slot = ch((pos..end).map(|p| p / ps).collect());
    let w_off = ch((pos..end).map(|p| p % ps).collect());
    let positions = ch((pos..end).collect());
    fwd.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: (pos / ps)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        }),
        std::slice::from_ref(rs),
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )
    .with_context(|| format!("{tag} state binding"))?;
    let out = Channel::new([1], dtype::i32).named("out");
    let sink = out.clone();
    fwd.epilogue(move || {
        sink.put(&reduce_argmax(intrinsics::logits()));
    });
    fwd.submit(pipe).with_context(|| format!("{tag} submit"))?;
    out.take_host::<i32>()
        .await
        .with_context(|| format!("{tag} take"))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    if wit_model::pass_kind() == wit_model::ForwardKind::Attention {
        return Ok("skipped: saved-context RS resume needs a linear model".to_string());
    }

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + 3).div_ceil(kv_page_size());
    let pipe = Pipeline::new();

    // ── 1. SAVE: prefill folds, then both halves index under one key ───────
    let ws = WorkingSet::new();
    let rs = RsWorkingSet::new();
    ws.reserve(max_pages).context("ws.reserve")?;
    let g0 = fire(&ws, &rs, &prompt, 0, max_pages, &pipe, "prefill").await?;
    ws.update_index(KEY).context("kv update_index")?;
    rs.update_index(KEY, n).context("rs update_index")?;

    // ── 3a. REFUSE off-boundary while the snapshot is fresh ────────────────
    // Behind the boundary and past it must BOTH take the named refusal.
    let mut refusals = 0;
    for requested in [n - 1, n + 1] {
        match RsWorkingSet::from_index(KEY, requested) {
            Err(e) if e.contains("boundary mismatch") => refusals += 1,
            Err(e) => return Err(format!("off-boundary resume: wrong diagnostic: {e}")),
            Ok(_) => {
                return Err(format!(
                    "resuming at {requested} against a boundary of {n} must refuse"
                ));
            }
        }
    }

    // ── 3b. A missing key is a typed None ──────────────────────────────────
    if RsWorkingSet::from_index(b"gdn-save-resume/absent", n)
        .context("missing-key lookup")?
        .is_some()
    {
        return Err("a never-saved key returned a working set".to_string());
    }

    // ── 2. RESUME at the exact boundary and decode one token ───────────────
    let ws2 = WorkingSet::from_index(KEY)
        .context("kv from_index")?
        .ok_or("saved KV index miss")?;
    let rs2 = RsWorkingSet::from_index(KEY, n)
        .context("rs from_index")?
        .ok_or("saved RS snapshot miss")?;
    let have = ws2.page_len();
    if max_pages > have {
        ws2.reserve(max_pages - have).context("ws2.reserve")?;
    }
    let resumed = fire(&ws2, &rs2, &[g0 as u32], n, max_pages, &pipe, "resumed").await?;

    // The saver keeps decoding too: its next fold copies-on-write off the
    // shared snapshot slot instead of scribbling on it.
    let original = fire(&ws, &rs, &[g0 as u32], n, max_pages, &pipe, "original").await?;

    // ── 4. REMOVE: true once, false after ──────────────────────────────────
    let removed_rs = RsWorkingSet::remove_index(KEY).context("rs remove_index")?;
    let removed_again = RsWorkingSet::remove_index(KEY).context("rs remove_index twice")?;
    let removed_kv = WorkingSet::remove_index(KEY).context("kv remove_index")?;
    pipe.close();

    let result = format!(
        "saveresume committed={n} resumed=1 original=1 refusals={refusals} missing=none \
         removed={}",
        removed_rs && removed_kv && !removed_again
    );
    eprintln!("[GDN_SAVE_RESUME] {result} (resumed_tok={resumed} original_tok={original})");
    Ok(result)
}
