use controller_api::{Health, RoutingTable};

#[derive(Debug, Clone, Copy)]
pub struct AdmissionConfig {
    pub kv_saturate_bucket: u8,
    pub max_inflight_per_worker: u32,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            kv_saturate_bucket: 240,
            max_inflight_per_worker: 256,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    Reject(RejectReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    ClusterSaturated,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::ClusterSaturated => {
                f.write_str("cluster saturated: no healthy worker has KV/seq headroom")
            }
        }
    }
}

pub fn admit(table: &RoutingTable, cfg: &AdmissionConfig) -> AdmissionDecision {
    let has_headroom = table.workers.iter().any(|w| {
        w.health == Health::Healthy
            && w.coarse_load.kv_pressure_bucket < cfg.kv_saturate_bucket
            && w.coarse_load.inflight < cfg.max_inflight_per_worker
    });
    if has_headroom {
        AdmissionDecision::Admit
    } else {
        AdmissionDecision::Reject(RejectReason::ClusterSaturated)
    }
}
