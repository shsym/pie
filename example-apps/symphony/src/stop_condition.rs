use crate::{l4m, l4m_async};

pub fn any<SC1, SC2>(sc1: SC1, sc2: SC2) -> impl StopCondition
where
    SC1: StopCondition + 'static,
    SC2: StopCondition + 'static,
{
    StopConditionList::new(vec![Box::new(sc1), Box::new(sc2)])
}

pub trait StopCondition {
    fn should_stop(&mut self, token_ids: &[u32]) -> bool;
}

pub struct Until {
    token_ids: Vec<u32>,
}

impl Until {
    pub fn new(text: &str) -> Self {
        let token_ids = l4m::tokenize(text);
        Self { token_ids }
    }
}

impl StopCondition for Until {
    fn should_stop(&mut self, token_ids: &[u32]) -> bool {
        token_ids.ends_with(&self.token_ids)
    }
}

pub struct Length {
    max_tokens: usize,
}

impl Length {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }
}

impl StopCondition for Length {
    fn should_stop(&mut self, token_ids: &[u32]) -> bool {
        token_ids.len() >= self.max_tokens
    }
}

pub struct StopConditionList {
    conditions: Vec<Box<dyn StopCondition>>,
}

impl StopConditionList {
    pub fn new(conditions: Vec<Box<dyn StopCondition>>) -> Self {
        Self { conditions }
    }
}

impl StopCondition for StopConditionList {
    fn should_stop(&mut self, token_ids: &[u32]) -> bool {
        self.conditions.iter_mut().any(|c| c.should_stop(token_ids))
    }
}
