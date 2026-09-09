use crate::error::Result;
use crate::extent::Extent;
use crate::plan::index::{PlanIndex, instr_by_id};
use crate::plan::passes::tile::{TileMapFacts, facts_of, kernel_for};
use crate::plan::{BufferDecl, DestExtent, LoadPlan, SourceExtent, StorageInstr};
use crate::types::{BufferId, Encoding, InstrId};

pub(super) fn stage_device_transforms(program: &mut LoadPlan) -> Result<usize> {
    if program.target.tile_map_mask == 0 {
        return Ok(0);
    }
    let index = PlanIndex::new(program);
    let old = program.instrs.clone();
    let schedule = program.schedule.clone();

    let mut decisions = Vec::with_capacity(schedule.len());
    let mut wanted: Vec<BufferId> = Vec::new();
    for id in &schedule {
        let instr = instr_by_id(&old, *id)?;
        let decision = decide(program, &index, instr)?;
        if let Some(decision) = &decision {
            wanted.extend(decision.arena_operands.iter().copied());
        }
        decisions.push(decision);
    }
    if decisions.iter().all(Option::is_none) {
        return Ok(0);
    }

    let mut rewritten: Vec<StorageInstr> = Vec::with_capacity(old.len() + decisions.len() * 2);
    let mut staged = 0usize;
    for (id, decision) in schedule.iter().zip(decisions) {
        let instr = instr_by_id(&old, *id)?.clone();
        let Some(decision) = decision else {
            rewritten.push(instr);
            continue;
        };
        let Some(stage) = decision.stage else {
            rewritten.push(instr);
            continue;
        };
        let buffer = declare_staging_buffer(program, &stage)?;
        wanted.push(buffer);
        rewritten.push(StorageInstr::Allocate {
            id: InstrId(0),
            buffer,
        });
        rewritten.push(StorageInstr::ExtentWrite {
            id: InstrId(0),
            source: stage.source.clone(),
            dest: DestExtent {
                buffer,
                offset: 0,
                stride: Extent::byte_run(stage.source.span_bytes),
            },
        });
        rewritten.push(read_from_buffer(instr, buffer)?);
        staged += 1;
    }
    super::rewrite::rewrite_program_instrs(program, rewritten)?;
    let placed = super::arena::place_in_scratch(program, &wanted)?;
    Ok(staged + placed)
}

struct Decision {
    stage: Option<Staging>,
    arena_operands: Vec<BufferId>,
}

struct Staging {
    source: SourceExtent,
    shape: Vec<i64>,
    encoding: Encoding,
    alignment: u32,
}

fn decide(program: &LoadPlan, index: &PlanIndex, instr: &StorageInstr) -> Result<Option<Decision>> {
    let StorageInstr::TileMap {
        source,
        dest,
        inputs,
        outputs,
        transform,
        ..
    } = instr
    else {
        return Ok(None);
    };
    let Some(facts) = facts_of(program, index, instr) else {
        return Ok(None);
    };
    let operands = || {
        let mut all: Vec<BufferId> = inputs.iter().chain(outputs).copied().collect();
        all.extend(dest.as_ref().map(|dest| dest.buffer));
        all
    };

    let Some(source) = source else {
        let facts = TileMapFacts {
            operands_in_arena: true,
            ..facts
        };
        return Ok(kernel_for(&facts, &program.target).map(|_| Decision {
            stage: None,
            arena_operands: operands(),
        }));
    };

    if transform.metadata_source.is_some() {
        return Ok(None);
    }
    let Some(raw) = index.source(program, source.tensor_id) else {
        return Ok(None);
    };
    if !matches!(raw.encoding, Encoding::Raw(_)) {
        return Ok(None);
    }
    let Some(primary) = outputs.first() else {
        return Ok(None);
    };
    let out = program.buffer(*primary)?;
    let shape = match dest {
        Some(dest) => dest.stride.dims.iter().map(|dim| dim.count).collect(),
        None => out.ty.shape.clone(),
    };
    let elements = crate::types::tensor_elements(&shape).unwrap_or(0);
    if elements == 0 || elements.saturating_mul(source.dtype.bytes_ceil()) != source.span_bytes {
        return Ok(None);
    }

    let staged_facts = TileMapFacts {
        has_source: false,
        compact_source: true,
        source_dtype: Some(source.dtype),
        in_place: false,
        operands_in_arena: true,
        ..facts
    };
    if kernel_for(&staged_facts, &program.target).is_none() {
        return Ok(None);
    }
    Ok(Some(Decision {
        stage: Some(Staging {
            source: source.clone(),
            shape,
            encoding: Encoding::Raw(source.dtype),
            alignment: out.alignment,
        }),
        arena_operands: operands(),
    }))
}

fn read_from_buffer(instr: StorageInstr, buffer: BufferId) -> Result<StorageInstr> {
    let StorageInstr::TileMap {
        id,
        kind,
        dest,
        mut inputs,
        outputs,
        tile,
        transform,
        ..
    } = instr
    else {
        return Err(crate::error::Error::Internal(
            "stage-device-transforms rewrote something that is not a TileMap".to_string(),
        ));
    };
    inputs.insert(0, buffer);
    Ok(StorageInstr::TileMap {
        id,
        kind,
        source: None,
        dest,
        inputs,
        outputs,
        tile,
        transform,
    })
}

fn declare_staging_buffer(program: &mut LoadPlan, stage: &Staging) -> Result<BufferId> {
    let id = BufferId(
        u32::try_from(program.buffers.len())
            .map_err(|_| crate::error::Error::Contract("too many buffers".to_string()))?,
    );
    program.buffers.push(BufferDecl {
        id,
        tensor: None,
        ty: crate::contract::TensorType::new(stage.shape.clone(), stage.encoding.clone()),
        bytes: stage.source.span_bytes,
        alignment: stage.alignment.max(program.target.preferred_alignment),
        temporary: true,
        persistent_offset: None,
        scratch_offset: None,
    });
    Ok(id)
}
