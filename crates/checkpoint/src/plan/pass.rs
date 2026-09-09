use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::plan::LoadPlan;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Rewrite,

    Check,
}

#[derive(Clone, Copy)]
pub struct Pass {
    pub name: &'static str,

    pub stage: Stage,

    pub for_arena: bool,

    pub run: fn(&mut LoadPlan) -> Result<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassStats {
    pub pass: String,
    pub instrs_before: usize,
    pub instrs_after: usize,
    pub rewrites: usize,
}

pub fn run_all(plan: &mut LoadPlan) -> Result<Vec<PassStats>> {
    run_passes(plan, super::passes::all())
}

pub fn run_arenaless(plan: &mut LoadPlan) -> Result<Vec<PassStats>> {
    let pipeline: Vec<Pass> = super::passes::all()
        .iter()
        .copied()
        .filter(|pass| !pass.for_arena)
        .collect();
    run_passes(plan, &pipeline)
}

pub(super) fn run_passes(plan: &mut LoadPlan, passes: &[Pass]) -> Result<Vec<PassStats>> {
    let mut stats = Vec::new();
    let mut checking = false;
    for pass in passes {
        match pass.stage {
            Stage::Check => checking = true,
            Stage::Rewrite if checking => {
                return Err(crate::error::Error::Internal(format!(
                    "pass '{}' rewrites the plan after a validator has already \
                     checked it; every rewrite must come before every check",
                    pass.name
                )));
            }
            Stage::Rewrite => {}
        }
        let before = plan.instrs.len();
        let rewrites = (pass.run)(plan)?;
        if pass.stage == Stage::Check && rewrites != 0 {
            return Err(crate::error::Error::Internal(format!(
                "pass '{}' is a validator but reports {rewrites} rewrites",
                pass.name
            )));
        }
        stats.push(PassStats {
            pass: pass.name.to_string(),
            instrs_before: before,
            instrs_after: plan.instrs.len(),
            rewrites,
        });
    }
    Ok(stats)
}
