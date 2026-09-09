use std::collections::BTreeMap;

use crate::error::Result;
use crate::format::cbor::Value;
use crate::format::{canonical_term, check_shape, Blocks, Digest, Leaf, Plane, Term};
use crate::provide::store::StoreId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location {
    pub store: StoreId,
    pub offset: u64,
    pub len: u64,
}

impl Location {
    pub fn alignment(&self) -> u64 {
        if self.offset == 0 {
            return crate::provide::store::page_size();
        }
        1u64 << self.offset.trailing_zeros().min(63)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    At(Location),
    Encoded {
        at: Location,
        encoding: String,
        decoded_len: u64,
    },
    Opaque {
        store: StoreId,
        key: u64,
        decoded_len: u64,
    },
}

impl Payload {
    pub fn location(&self) -> Option<Location> {
        match self {
            Payload::At(at) => Some(*at),
            _ => None,
        }
    }

    pub fn store(&self) -> StoreId {
        match self {
            Payload::At(at) | Payload::Encoded { at, .. } => at.store,
            Payload::Opaque { store, .. } => *store,
        }
    }

    pub fn decoded_len(&self) -> u64 {
        match self {
            Payload::At(at) => at.len,
            Payload::Encoded { decoded_len, .. } | Payload::Opaque { decoded_len, .. } => {
                *decoded_len
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub shape: Vec<u64>,
    pub term: Option<Term>,
    pub layout: Option<String>,
    pub attributes: Option<Value>,
    pub payload: Payload,
    pub digest: Option<Digest>,
    pub blocks: Option<Blocks>,
}

impl Entry {
    pub fn leaf(shape: Vec<u64>, leaf: Leaf, at: Location) -> Self {
        Entry::at(shape, Term::Leaf(leaf), at)
    }

    pub fn at(shape: Vec<u64>, term: Term, at: Location) -> Self {
        Entry {
            shape,
            term: Some(term),
            layout: None,
            attributes: None,
            payload: Payload::At(at),
            digest: None,
            blocks: None,
        }
    }

    pub fn num_elements(&self) -> Result<u64> {
        check_shape(&self.shape)
    }

    pub fn planes(&self) -> Result<Vec<Plane>> {
        canonical_term(self.term.as_ref(), self.layout.as_deref())?.planes(&self.shape)
    }

    pub(crate) fn store(&self) -> StoreId {
        self.payload.store()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    entries: BTreeMap<String, Entry>,
    attributes: Option<Value>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, entry: Entry) -> Option<Entry> {
        self.entries.insert(name.into(), entry)
    }

    pub fn set_attributes(&mut self, attributes: Option<Value>) {
        self.attributes = attributes;
    }

    pub fn attributes(&self) -> Option<&Value> {
        self.attributes.as_ref()
    }

    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.get(name)
    }

    pub(crate) fn get_key_value(&self, name: &str) -> Option<(&str, &Entry)> {
        self.entries
            .get_key_value(name)
            .map(|(k, v)| (k.as_str(), v))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Entry)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub(crate) fn into_iter_sorted(self) -> impl Iterator<Item = (String, Entry)> {
        self.entries.into_iter()
    }

    pub(crate) fn renamed(self, f: impl Fn(&str) -> String) -> crate::Result<Catalog> {
        let attributes = self.attributes;
        let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
        for (name, entry) in self.entries {
            let renamed = f(&name);
            if let Some(previous) = entries.insert(renamed.clone(), entry) {
                let _ = previous;
                return Err(crate::Error::reject(
                    crate::Rule::NameCollision,
                    format!("renaming {name:?} to {renamed:?} displaces a tensor already there"),
                ));
            }
        }
        Ok(Catalog {
            entries,
            attributes,
        })
    }

    pub(crate) fn rebase(&mut self, f: impl Fn(StoreId) -> StoreId) {
        for entry in self.entries.values_mut() {
            match &mut entry.payload {
                Payload::At(at) => at.store = f(at.store),
                Payload::Encoded { at, .. } => at.store = f(at.store),
                Payload::Opaque { store, .. } => *store = f(*store),
            }
        }
    }
}
