pub type Hash256 = [u8; 32];

const DOMAIN_CONTEXT: &str = "pie kv cache domain v1";
const SLOT_CONTEXT: &str = "pie kv token-slot hash v1";
const OPAQUE_CONTEXT: &str = "pie kv opaque token-slot hash v1";
const PAGE_CONTEXT: &str = "pie kv page hash v1";
const PATH_CONTEXT: &str = "pie kv path hash v1";

pub fn cache_domain(seed: &[u8]) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(DOMAIN_CONTEXT);
    hasher.update(seed);
    *hasher.finalize().as_bytes()
}

pub fn chain_token_slot_hash(
    domain: &Hash256,
    prev: Option<&Hash256>,
    token: u32,
    position: u32,
) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(SLOT_CONTEXT);
    hasher.update(domain);
    match prev {
        Some(prev) => {
            hasher.update(&[1]);
            hasher.update(prev);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&token.to_le_bytes());
    hasher.update(&position.to_le_bytes());
    *hasher.finalize().as_bytes()
}

pub fn opaque_token_slot_hash(nonce: &Hash256, counter: u64) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(OPAQUE_CONTEXT);
    hasher.update(nonce);
    hasher.update(&counter.to_le_bytes());
    *hasher.finalize().as_bytes()
}

pub fn page_hash(slot_hashes: &[Option<Hash256>]) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(PAGE_CONTEXT);
    hasher.update(&(slot_hashes.len() as u64).to_le_bytes());
    for slot in slot_hashes {
        match slot {
            Some(hash) => {
                hasher.update(&[1]);
                hasher.update(hash);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    *hasher.finalize().as_bytes()
}

pub fn fold_path_hash(prev: Option<Hash256>, page_hashes: &[Hash256]) -> Option<Hash256> {
    let mut acc = prev;
    for page in page_hashes {
        let mut hasher = blake3::Hasher::new_derive_key(PATH_CONTEXT);
        match &acc {
            Some(acc) => {
                hasher.update(&[1]);
                hasher.update(acc);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(page);
        acc = Some(*hasher.finalize().as_bytes());
    }
    acc
}
