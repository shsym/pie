use crate::plan::pass::{Pass, Stage};

mod arena;
mod memory;
mod rewrite;
mod stage;
pub mod tile;
mod validate;

#[cfg(test)]
mod tests;

pub fn all() -> &'static [Pass] {
    &[
        Pass {
            name: "assign-persistent-offsets",
            stage: Stage::Rewrite,
            for_arena: false,
            run: arena::assign_persistent_offsets,
        },
        Pass {
            name: "stage-device-transforms",
            stage: Stage::Rewrite,
            for_arena: false,
            run: stage::stage_device_transforms,
        },
        Pass {
            name: "coalesce-persistent-arena-writes",
            stage: Stage::Rewrite,
            for_arena: true,
            run: rewrite::coalesce_persistent_arena_writes,
        },
        Pass {
            name: "hoist-bulk-arena-writes",
            stage: Stage::Rewrite,
            for_arena: true,
            run: rewrite::hoist_bulk_extent_writes,
        },
        Pass {
            name: "recompute-memory-plan",
            stage: Stage::Rewrite,
            for_arena: false,
            run: memory::recompute_memory_plan,
        },
        Pass {
            name: "lower-backend-tiling",
            stage: Stage::Rewrite,
            for_arena: false,
            run: tile::lower_backend_tiling,
        },
        Pass {
            name: "validate-fill-order",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_fill_order,
        },
        Pass {
            name: "validate-target-support",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_target_support,
        },
        Pass {
            name: "validate-bound-encodings",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_bound_encodings,
        },
        Pass {
            name: "validate-scale-factors",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_scale_factors,
        },
        Pass {
            name: "validate-persistent-layout",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_persistent_layout,
        },
        Pass {
            name: "validate-kernel-operands",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_kernel_operands,
        },
    ]
}
