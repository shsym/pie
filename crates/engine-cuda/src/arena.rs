use kernels_cuda::Tensor;
use model_compiler::ArenaMap;
use model_exec::store::arena::rect;
use model_ir::ValueId;

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::run::SlotTable;

#[derive(Debug)]
pub struct Arena {
    store: Buffer,
}

impl Arena {
    pub fn reserve(map: &ArenaMap) -> Result<Arena> {
        Ok(Arena {
            store: Buffer::zeroed(usize::try_from(map.bytes).unwrap_or(usize::MAX))?,
        })
    }

    #[must_use]
    pub fn base(&self) -> u64 {
        self.store.ptr()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }

    #[must_use]
    pub fn slots(&self, map: &ArenaMap, rows: model_compiler::FireRows) -> SlotTable {
        carve(self.base(), map, rows)
    }

    pub fn read(&self, at: u64, into: &mut [u8]) -> Result<()> {
        let base = self.base();
        let offset = at.checked_sub(base).ok_or(Fault::Ceiling {
            what: "bytes into the arena",
            need: at,
            have: base,
        })?;
        self.store.read(offset, into)
    }
}

#[must_use]
pub fn carve(base: u64, map: &ArenaMap, rows: model_compiler::FireRows) -> SlotTable {
    let mut cells: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);
        cells.push(rect(map, value, rows).map(|rect| {
            Tensor::new(base + rect.offset, rect.rows, rect.width, rect.dtype)
        }));
    }
    SlotTable(cells)
}

#[cfg(test)]
mod tests {
    use model_compiler::{Budget, DeviceProfile, compile};
    use model_dsl::Platform;
    
    use super::*;

    const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

    fn compiled() -> (model_ir::Trace, model_compiler::CompiledModel) {
        let trace = models::sku(SKU).expect("the catalog ships the smoke's SKU").trace;
        let trace = trace(Platform::Cuda);
        let compiled = compile(&trace, &Budget::new(4, 64), &DeviceProfile::default())
            .expect("the smoke's SKU bakes");
        (trace, compiled)
    }

    #[test]
    fn the_carve_fits_the_allocation_it_asks_for() {
        let (_, compiled) = compiled();
        let slots = carve(0, &compiled.arena, model_compiler::FireRows::text_only(64, 4));
        for handle in slots.0.iter().flatten() {
            let end = handle.ptr
                + handle.elements()
                    * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or(0);
            assert!(
                end <= compiled.arena.bytes,
                "a rectangle ending at {end} in an arena of {} bytes",
                compiled.arena.bytes
            );
        }
    }
}
