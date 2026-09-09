use anyhow::{Context, Result, bail, ensure};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) type TokenId = u32;

type Rank = u32;

pub(crate) struct BpeTable {
    token_to_id: FxHashMap<Arc<[u8]>, TokenId>,
    merges: FxHashMap<(TokenId, TokenId), (Rank, TokenId)>,
    id_to_bytes: Vec<Option<Arc<[u8]>>>,
    byte_fallback_ids: [Option<TokenId>; 256],
}

impl BpeTable {
    pub(crate) fn from_decoder_map(map: HashMap<TokenId, Vec<u8>>) -> Result<Self> {
        if map.is_empty() {
            return Ok(Self {
                token_to_id: FxHashMap::default(),
                merges: FxHashMap::default(),
                id_to_bytes: Vec::new(),
                byte_fallback_ids: [None; 256],
            });
        }
        let vocab_size = u32::try_from(map.len()).context("BPE vocabulary is too large")?;
        let max_id = map.keys().copied().max().unwrap();
        ensure!(
            max_id.checked_add(1) == Some(vocab_size),
            "token IDs must be contiguous from 0 ({} entries, max ID {max_id})",
            map.len()
        );

        let mut id_to_bytes = vec![None; map.len()];
        let mut token_to_id: FxHashMap<Arc<[u8]>, TokenId> =
            FxHashMap::with_capacity_and_hasher(map.len(), Default::default());
        let mut merges =
            FxHashMap::with_capacity_and_hasher(map.len().saturating_mul(2), Default::default());

        for (id, bytes) in map {
            let bytes: Arc<[u8]> = bytes.into();
            token_to_id
                .entry(bytes.clone())
                .and_modify(|previous_id| *previous_id = (*previous_id).min(id))
                .or_insert(id);
            id_to_bytes[id as usize] = Some(bytes);
        }

        for (id, bytes) in id_to_bytes.iter().enumerate() {
            let bytes = bytes.as_deref().expect("contiguous IDs were validated");
            if bytes.len() < 2 {
                continue;
            }
            for split in 1..bytes.len() {
                let left = &bytes[..split];
                let right = &bytes[split..];
                if let (Some(&left_id), Some(&right_id)) =
                    (token_to_id.get(left), token_to_id.get(right))
                {
                    let rank = id as TokenId;
                    merges
                        .entry((left_id, right_id))
                        .and_modify(|e: &mut (Rank, TokenId)| {
                            if rank < e.0 {
                                *e = (rank, id as TokenId);
                            }
                        })
                        .or_insert((rank, id as TokenId));
                }
            }
        }

        Ok(BpeTable {
            token_to_id,
            merges,
            id_to_bytes,
            byte_fallback_ids: [None; 256],
        })
    }

    pub(crate) fn from_vocab_and_merges(
        vocab: &HashMap<String, u32>,
        merge_pairs: &[(String, String)],
        raw_byte_keys: bool,
    ) -> Result<Self> {
        ensure!(!vocab.is_empty(), "BPE vocabulary is empty");
        let vocab_size = u32::try_from(vocab.len()).context("BPE vocabulary is too large")?;
        let max_id = vocab.values().copied().max().unwrap();
        ensure!(
            max_id.checked_add(1) == Some(vocab_size),
            "token IDs must be contiguous from 0 ({} entries, max ID {max_id})",
            vocab.len()
        );

        let mut id_to_bytes = vec![None; vocab.len()];
        let mut token_to_id: FxHashMap<Arc<[u8]>, TokenId> =
            FxHashMap::with_capacity_and_hasher(vocab.len(), Default::default());
        let mut merges = FxHashMap::with_capacity_and_hasher(merge_pairs.len(), Default::default());

        for (token, &id) in vocab {
            let key = if raw_byte_keys {
                byte_level_token_to_bytes(token)
            } else {
                token.as_bytes().to_vec()
            };
            ensure!(
                id_to_bytes[id as usize].is_none(),
                "duplicate token ID {id}"
            );
            let key: Arc<[u8]> = key.into();
            if let Some(previous_id) = token_to_id.insert(key.clone(), id) {
                bail!("tokens {previous_id} and {id} decode to the same byte sequence");
            }
            id_to_bytes[id as usize] = Some(key);
        }

        for (idx, (a, b)) in merge_pairs.iter().enumerate() {
            let a_key = if raw_byte_keys {
                byte_level_token_to_bytes(a)
            } else {
                a.as_bytes().to_vec()
            };

            let b_key = if raw_byte_keys {
                byte_level_token_to_bytes(b)
            } else {
                b.as_bytes().to_vec()
            };

            let mut merged_key = a_key.clone();
            merged_key.extend_from_slice(&b_key);

            let left_id = token_to_id
                .get(a_key.as_slice())
                .copied()
                .with_context(|| format!("merge {idx} references unknown left token {a:?}"))?;
            let right_id = token_to_id
                .get(b_key.as_slice())
                .copied()
                .with_context(|| format!("merge {idx} references unknown right token {b:?}"))?;
            let merged_id = token_to_id
                .get(merged_key.as_slice())
                .copied()
                .with_context(|| format!("merge {idx} produces unknown token {a:?} + {b:?}"))?;
            let rank = u32::try_from(idx + 1).context("too many BPE merges")?;
            if merges
                .insert((left_id, right_id), (rank, merged_id))
                .is_some()
            {
                bail!("duplicate merge pair at index {idx}: {a:?} + {b:?}");
            }
        }

        let mut byte_fallback_ids = [None; 256];
        for byte in 0u16..=255 {
            let hex = format!("<0x{byte:02X}>");
            if let Some(&id) = vocab.get(&hex) {
                byte_fallback_ids[byte as usize] = Some(id);
            }
        }

        Ok(BpeTable {
            token_to_id,
            merges,
            id_to_bytes,
            byte_fallback_ids,
        })
    }

    #[inline]
    pub(crate) fn bytes_to_id(&self, bytes: &[u8]) -> Option<TokenId> {
        self.token_to_id.get(bytes).copied()
    }

    #[inline]
    pub(crate) fn id_to_bytes(&self, id: TokenId) -> Option<&[u8]> {
        self.id_to_bytes.get(id as usize)?.as_deref()
    }

    pub(crate) fn id_to_shared_bytes(&self, id: TokenId) -> Option<Arc<[u8]>> {
        self.id_to_bytes.get(id as usize)?.clone()
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    pub(crate) fn insert_added(&mut self, bytes: Vec<u8>, id: TokenId) -> Result<()> {
        ensure!(
            id as usize <= self.id_to_bytes.len(),
            "added token ID {id} creates a gap after {}",
            self.id_to_bytes.len()
        );
        let bytes: Arc<[u8]> = bytes.into();
        if let Some(previous_id) = self.token_to_id.get(bytes.as_ref()).copied() {
            ensure!(
                previous_id == id,
                "added token bytes already map to ID {previous_id}, not {id}"
            );
        } else {
            self.token_to_id.insert(bytes.clone(), id);
        }
        if id as usize == self.id_to_bytes.len() {
            self.id_to_bytes.push(Some(bytes));
        } else {
            self.id_to_bytes[id as usize] = Some(bytes);
        }
        Ok(())
    }

    pub(crate) fn has_complete_byte_fallback(&self) -> bool {
        self.byte_fallback_ids.iter().all(Option::is_some)
    }

    pub(crate) fn has_all_byte_atoms(&self) -> bool {
        (0u16..=255).all(|byte| self.bytes_to_id(&[byte as u8]).is_some())
    }

    pub(crate) fn has_unique_token_bytes(&self) -> bool {
        let mut seen = FxHashSet::default();
        self.id_to_bytes
            .iter()
            .flatten()
            .all(|bytes| seen.insert(bytes.as_ref()))
    }

    #[inline]
    fn pair_rank(&self, left: TokenId, right: TokenId) -> Rank {
        self.merges
            .get(&(left, right))
            .map_or(Rank::MAX, |&(r, _)| r)
    }

    #[inline]
    fn pair_merge(&self, left: TokenId, right: TokenId) -> Option<(Rank, TokenId)> {
        self.merges.get(&(left, right)).copied()
    }

    #[inline]
    fn byte_fallback(&self, byte: u8) -> Option<TokenId> {
        self.byte_fallback_ids[byte as usize]
    }

    pub(crate) fn decode_table(&self) -> &[Option<Arc<[u8]>>] {
        &self.id_to_bytes
    }

    pub(crate) fn merge_quads(&self) -> Vec<(TokenId, TokenId, Rank, TokenId)> {
        let mut quads: Vec<_> = self
            .merges
            .iter()
            .map(|(&(left, right), &(rank, merged))| (left, right, rank, merged))
            .collect();
        quads.sort_unstable_by_key(|&(left, right, _, _)| (left, right));
        quads
    }

    pub(crate) fn byte_fallback_table(&self) -> &[Option<TokenId>; 256] {
        &self.byte_fallback_ids
    }

    pub(crate) fn encode_map_is_derivable(&self) -> bool {
        let mut derived: FxHashMap<&[u8], TokenId> =
            FxHashMap::with_capacity_and_hasher(self.token_to_id.len(), Default::default());
        for (id, bytes) in self.id_to_bytes.iter().enumerate() {
            let Some(bytes) = bytes.as_deref() else {
                continue;
            };
            derived.entry(bytes).or_insert(id as TokenId);
        }
        derived.len() == self.token_to_id.len()
            && self
                .token_to_id
                .iter()
                .all(|(bytes, &id)| derived.get(bytes.as_ref()) == Some(&id))
    }

    pub(crate) fn from_canonical(
        vocab: Vec<Vec<u8>>,
        merges: &[(TokenId, TokenId, Rank, TokenId)],
        byte_fallback_ids: [Option<TokenId>; 256],
    ) -> Result<Self> {
        let mut token_to_id: FxHashMap<Arc<[u8]>, TokenId> =
            FxHashMap::with_capacity_and_hasher(vocab.len(), Default::default());
        let mut id_to_bytes = Vec::with_capacity(vocab.len());
        for (id, bytes) in vocab.into_iter().enumerate() {
            let bytes: Arc<[u8]> = bytes.into();
            token_to_id.entry(bytes.clone()).or_insert(id as TokenId);
            id_to_bytes.push(Some(bytes));
        }

        let vocab_size = u32::try_from(id_to_bytes.len()).context("BPE vocabulary is too large")?;
        let mut table = FxHashMap::with_capacity_and_hasher(merges.len(), Default::default());
        for &(left, right, rank, merged) in merges {
            for id in [left, right, merged] {
                ensure!(
                    id < vocab_size,
                    "merge ({left}, {right}) names token {id}, outside a vocabulary of {vocab_size}"
                );
            }
            ensure!(
                table.insert((left, right), (rank, merged)).is_none(),
                "duplicate merge pair ({left}, {right})"
            );
        }
        for id in byte_fallback_ids.iter().flatten() {
            ensure!(
                *id < vocab_size,
                "byte fallback names token {id}, outside a vocabulary of {vocab_size}"
            );
        }

        Ok(BpeTable {
            token_to_id,
            merges: table,
            id_to_bytes,
            byte_fallback_ids,
        })
    }
}

const fn is_direct_byte(b: u8) -> bool {
    matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)
}

const fn build_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    let mut b = 0u16;
    while b < 256 {
        if is_direct_byte(b as u8) {
            // SAFETY: direct bytes are valid Unicode code points (0x21..0xFF)
            table[b as usize] = match char::from_u32(b as u32) {
                Some(c) => c,
                None => '\0', // unreachable for valid direct bytes
            };
        } else {
            table[b as usize] = match char::from_u32(256 + n) {
                Some(c) => c,
                None => '\0', // unreachable, 256+n always valid
            };
            n += 1;
        }
        b += 1;
    }
    table
}

const fn build_char_to_byte() -> [Option<u8>; 324] {
    let b2u = build_byte_to_unicode();
    let mut table: [Option<u8>; 324] = [None; 324];
    let mut b = 0u16;
    while b < 256 {
        let idx = b2u[b as usize] as usize;
        if idx < 324 {
            table[idx] = Some(b as u8);
        }
        b += 1;
    }
    table
}

static CHAR_TO_BYTE: [Option<u8>; 324] = build_char_to_byte();

fn byte_level_token_to_bytes(token: &str) -> Vec<u8> {
    let mut decoded = Vec::with_capacity(token.len());
    for character in token.chars() {
        let index = character as usize;
        let Some(byte) = CHAR_TO_BYTE.get(index).copied().flatten() else {
            return token.as_bytes().to_vec();
        };
        decoded.push(byte);
    }
    decoded
}

struct Node {
    token_id: TokenId,
    rank: Rank,
    prev: u32,
    next: u32,
}

const NONE: u32 = u32::MAX;

fn bpe_merge(initial_ids: &[TokenId], ranks: &BpeTable) -> SmallVec<[TokenId; 16]> {
    if initial_ids.len() <= 32 {
        bpe_merge_linear(initial_ids, ranks)
    } else {
        bpe_merge_heap(initial_ids, ranks)
    }
}

fn build_nodes(ids: &[TokenId]) -> SmallVec<[Node; 32]> {
    let n = ids.len();
    let mut nodes: SmallVec<[Node; 32]> = SmallVec::with_capacity(n);
    for (i, &id) in ids.iter().enumerate() {
        nodes.push(Node {
            token_id: id,
            rank: Rank::MAX,
            prev: if i > 0 { (i - 1) as u32 } else { NONE },
            next: if i + 1 < n { (i + 1) as u32 } else { NONE },
        });
    }
    nodes
}

#[inline]
fn node_rank(nodes: &[Node], i: usize, ranks: &BpeTable) -> Rank {
    let j = nodes[i].next;
    if j == NONE {
        return Rank::MAX;
    }
    ranks.pair_rank(nodes[i].token_id, nodes[j as usize].token_id)
}

fn collect_ids(nodes: &[Node]) -> SmallVec<[TokenId; 16]> {
    let mut ids = SmallVec::new();
    let mut cur = 0u32;
    while cur != NONE {
        ids.push(nodes[cur as usize].token_id);
        cur = nodes[cur as usize].next;
    }
    ids
}

fn bpe_merge_linear(initial_ids: &[TokenId], ranks: &BpeTable) -> SmallVec<[TokenId; 16]> {
    let n = initial_ids.len();
    debug_assert!(n >= 2);

    let mut nodes = build_nodes(initial_ids);

    let mut min_rank: (Rank, u32) = (Rank::MAX, NONE);
    for i in 0..n.saturating_sub(1) {
        let rank = node_rank(&nodes, i, ranks);
        nodes[i].rank = rank;
        if rank < min_rank.0 {
            min_rank = (rank, i as u32);
        }
    }

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1 as usize;
        let j = nodes[i].next as usize;

        let (_, merged_id) = ranks
            .pair_merge(nodes[i].token_id, nodes[j].token_id)
            .expect("cached merge rank must have a merge entry");
        nodes[i].token_id = merged_id;

        let j_next = nodes[j].next;
        nodes[i].next = j_next;
        if j_next != NONE {
            nodes[j_next as usize].prev = i as u32;
        }

        nodes[i].rank = node_rank(&nodes, i, ranks);
        let pred = nodes[i].prev;
        if pred != NONE {
            nodes[pred as usize].rank = node_rank(&nodes, pred as usize, ranks);
        }

        min_rank = (Rank::MAX, NONE);
        let mut cur = 0u32;
        loop {
            if nodes[cur as usize].rank < min_rank.0 {
                min_rank = (nodes[cur as usize].rank, cur);
            }
            let next = nodes[cur as usize].next;
            if next == NONE {
                break;
            }
            cur = next;
        }
    }

    collect_ids(&nodes)
}

fn bpe_merge_heap(initial_ids: &[TokenId], ranks: &BpeTable) -> SmallVec<[TokenId; 16]> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = initial_ids.len();
    debug_assert!(n >= 2);

    let mut nodes = build_nodes(initial_ids);

    let mut heap: BinaryHeap<Reverse<(Rank, u32)>> = BinaryHeap::with_capacity(n);

    for i in 0..n.saturating_sub(1) {
        let rank = node_rank(&nodes, i, ranks);
        nodes[i].rank = rank;
        if rank != Rank::MAX {
            heap.push(Reverse((rank, i as u32)));
        }
    }

    while let Some(Reverse((rank, idx))) = heap.pop() {
        let i = idx as usize;

        if nodes[i].rank != rank {
            continue;
        }

        let j = nodes[i].next as usize;

        let (_, merged_id) = ranks
            .pair_merge(nodes[i].token_id, nodes[j].token_id)
            .expect("heap merge rank must have a merge entry");
        nodes[i].token_id = merged_id;

        let j_next = nodes[j].next;
        nodes[i].next = j_next;
        if j_next != NONE {
            nodes[j_next as usize].prev = i as u32;
        }
        nodes[j].rank = Rank::MAX;

        let new_rank = node_rank(&nodes, i, ranks);
        nodes[i].rank = new_rank;
        if new_rank != Rank::MAX {
            heap.push(Reverse((new_rank, i as u32)));
        }

        let pred = nodes[i].prev;
        if pred != NONE {
            let pred_rank = node_rank(&nodes, pred as usize, ranks);
            nodes[pred as usize].rank = pred_rank;
            if pred_rank != Rank::MAX {
                heap.push(Reverse((pred_rank, pred)));
            }
        }
    }

    collect_ids(&nodes)
}

pub(crate) fn bpe_encode_bytes(
    piece: &[u8],
    bpe: &BpeTable,
    prefer_whole_token: bool,
    byte_fallback: bool,
    unk_token_id: Option<TokenId>,
    out: &mut Vec<TokenId>,
) {
    if piece.is_empty() {
        return;
    }

    let n = piece.len();

    if n == 1 {
        match bpe.bytes_to_id(piece) {
            Some(id) => out.push(id),
            None => fallback_into(piece, bpe, byte_fallback, unk_token_id, out),
        }
        return;
    }

    if prefer_whole_token && let Some(id) = bpe.bytes_to_id(piece) {
        out.push(id);
        return;
    }

    let mut initial_ids: SmallVec<[TokenId; 32]> = SmallVec::with_capacity(n);
    let mut all_resolved = true;
    for i in 0..n {
        match bpe.bytes_to_id(&piece[i..i + 1]) {
            Some(id) => initial_ids.push(id),
            None => {
                all_resolved = false;
                initial_ids.push(TokenId::MAX);
            }
        }
    }

    if all_resolved {
        let merged = bpe_merge(&initial_ids, bpe);
        out.extend_from_slice(&merged);
        return;
    }

    let mut i = 0;
    while i < n {
        if initial_ids[i] == TokenId::MAX {
            fallback_into(&piece[i..i + 1], bpe, byte_fallback, unk_token_id, out);
            i += 1;
        } else {
            let start = i;
            while i < n && initial_ids[i] != TokenId::MAX {
                i += 1;
            }
            let seg = &initial_ids[start..i];
            if seg.len() == 1 {
                out.push(seg[0]);
            } else {
                out.extend_from_slice(&bpe_merge(seg, bpe));
            }
        }
    }
}

pub(crate) fn bpe_encode_chars(
    piece: &str,
    bpe: &BpeTable,
    prefer_whole_token: bool,
    byte_fallback: bool,
    unk_token_id: Option<TokenId>,
    out: &mut Vec<TokenId>,
) {
    let bytes = piece.as_bytes();
    if bytes.is_empty() {
        return;
    }

    let offsets: SmallVec<[usize; 32]> = piece
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(bytes.len()))
        .collect();
    let n = offsets.len() - 1;

    if n == 1 {
        match bpe.bytes_to_id(bytes) {
            Some(id) => out.push(id),
            None => fallback_into(bytes, bpe, byte_fallback, unk_token_id, out),
        }
        return;
    }

    if prefer_whole_token && let Some(id) = bpe.bytes_to_id(bytes) {
        out.push(id);
        return;
    }

    let mut initial_ids: SmallVec<[TokenId; 32]> = SmallVec::with_capacity(n);
    let mut all_resolved = true;
    for w in offsets.windows(2) {
        let span = &bytes[w[0]..w[1]];
        match bpe.bytes_to_id(span) {
            Some(id) => initial_ids.push(id),
            None => {
                all_resolved = false;
                initial_ids.push(TokenId::MAX);
            }
        }
    }

    if all_resolved {
        let merged = bpe_merge(&initial_ids, bpe);
        out.extend_from_slice(&merged);
        return;
    }

    let mut i = 0;
    while i < n {
        if initial_ids[i] == TokenId::MAX {
            let span = &bytes[offsets[i]..offsets[i + 1]];
            fallback_into(span, bpe, byte_fallback, unk_token_id, out);
            i += 1;
        } else {
            let start = i;
            while i < n && initial_ids[i] != TokenId::MAX {
                i += 1;
            }
            let seg = &initial_ids[start..i];
            if seg.len() == 1 {
                out.push(seg[0]);
            } else {
                out.extend_from_slice(&bpe_merge(seg, bpe));
            }
        }
    }
}

fn fallback_into(
    bytes: &[u8],
    bpe: &BpeTable,
    byte_fallback: bool,
    unk_token_id: Option<TokenId>,
    out: &mut Vec<TokenId>,
) {
    if byte_fallback {
        for &b in bytes {
            if let Some(id) = bpe.byte_fallback(b).or(unk_token_id) {
                out.push(id);
            }
        }
    } else if let Some(id) = unk_token_id {
        out.push(id);
    }
}
