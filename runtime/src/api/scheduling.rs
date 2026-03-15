//! pie:core/scheduling - Market prices and program account queries

use crate::api::pie;
use crate::context;
use crate::linker::InstanceState;

use anyhow::Result;

impl pie::core::scheduling::Host for InstanceState {
    // ── Market prices ──────────────────────────────────────────────

    async fn keep(&mut self, device: u32, tier: String) -> Result<f64> {
        let market = context::get_market_state(0).await;
        let dev = device as usize;
        if dev >= market.device_prices.len() {
            return Ok(0.0);
        }
        let price = &market.device_prices[dev];
        match tier.as_str() {
            "gpu" => Ok(price.keep_gpu),
            "cpu" => Ok(price.keep_cpu),
            _ => Ok(price.keep_gpu),
        }
    }

    async fn make(&mut self, device: u32) -> Result<f64> {
        let market = context::get_market_state(0).await;
        let dev = device as usize;
        if dev >= market.device_prices.len() {
            return Ok(0.0);
        }
        Ok(market.device_prices[dev].make)
    }

    async fn interest(&mut self) -> Result<f64> {
        let market = context::get_market_state(0).await;
        Ok(market.interest_rate)
    }

    async fn list_devices(&mut self) -> Result<u32> {
        let market = context::get_market_state(0).await;
        Ok(market.device_prices.len() as u32)
    }

    // ── Account ────────────────────────────────────────────────────

    async fn balance(&mut self) -> Result<f64> {
        let pid = self.id();
        Ok(context::get_balance(0, pid).await)
    }
}
