use ids::{GatewayId, WorkerId};

use crate::state::{Gateway, Worker};

pub type Recovered = (Vec<(WorkerId, Worker)>, Vec<(GatewayId, Gateway)>);

pub trait StateStore: Send + 'static {
    fn put_worker(&mut self, id: WorkerId, worker: Option<&Worker>);

    fn put_gateway(&mut self, id: GatewayId, gateway: Option<&Gateway>);

    fn recover(&mut self) -> Recovered;
}

#[derive(Debug, Default)]
pub struct SoftState;

impl StateStore for SoftState {
    fn put_worker(&mut self, _id: WorkerId, _worker: Option<&Worker>) {}
    fn put_gateway(&mut self, _id: GatewayId, _gateway: Option<&Gateway>) {}
    fn recover(&mut self) -> Recovered {
        (Vec::new(), Vec::new())
    }
}
