use std::net::IpAddr;

use anyhow::{Context, anyhow};
use axum::http::HeaderMap;
use ids::TenantId;

use crate::session::Identity;

pub const IDENTITY_HEADER: &str = "x-pie-identity";
pub const FORWARDED_FOR_HEADER: &str = "x-forwarded-for";
pub const REQUEST_ID_HEADER: &str = "x-request-id";

pub fn extract(headers: &HeaderMap) -> anyhow::Result<Identity> {
    let raw = headers
        .get(IDENTITY_HEADER)
        .ok_or_else(|| anyhow!("missing `{IDENTITY_HEADER}` (edge must inject verified identity)"))?
        .to_str()
        .context("`x-pie-identity` is not valid UTF-8")?;

    let (tenant, user) = parse_identity(raw)?;

    let client_ip = headers
        .get(FORWARDED_FOR_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_forwarded_for);

    let request_id = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty());

    Ok(Identity {
        tenant: TenantId(tenant),
        user,
        client_ip,
        request_id,
    })
}

fn parse_identity(raw: &str) -> anyhow::Result<(String, String)> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(anyhow!("empty `{IDENTITY_HEADER}`"));
    }
    let (tenant, user) = match raw.split_once('/') {
        Some((t, u)) => (t.trim(), u.trim()),
        None => ("default", raw),
    };
    if user.is_empty() {
        return Err(anyhow!("`{IDENTITY_HEADER}` has empty user component"));
    }
    Ok((tenant.to_string(), user.to_string()))
}

fn parse_forwarded_for(value: &str) -> Option<IpAddr> {
    let first = value.split(',').next()?.trim();
    if first.is_empty() {
        return None;
    }
    first.parse::<IpAddr>().ok().or_else(|| {
        first
            .rsplit_once(':')
            .and_then(|(host, _)| host.parse().ok())
    })
}
