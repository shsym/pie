use crate::bindings::pie::inferlet::adapter;
use crate::traits::forward::{ForwardPass};
use crate::{Queue, Resource};

pub trait Adapter {
    fn allocate_adapter(&self) -> u32;
    fn deallocate_adapter(&self, ptr: u32);
    fn export_adapter(&self, ptr: u32, name: &str);
    fn import_adapter(&self, name: &str) -> u32;
    fn get_all_exported_adapters(&self) -> Vec<String>;
    fn release_exported_adapter(&self, name: &str);
}

pub trait SetAdapter {
    fn set_adapter(&self, adapter_ptr: u32);
}

impl Adapter for Queue {
    fn allocate_adapter(&self) -> u32 {
        self.allocate_resources(Resource::Adapter, 1)
            .into_iter()
            .next()
            .unwrap()
    }

    fn deallocate_adapter(&self, ptr: u32) {
        self.deallocate_resources(Resource::Adapter, &[ptr])
    }

    fn export_adapter(&self, ptr: u32, name: &str) {
        self.export_resource(Resource::Adapter, &[ptr], name)
    }

    fn import_adapter(&self, name: &str) -> u32 {
        self.import_resource(Resource::Adapter, name)
            .into_iter()
            .next()
            .unwrap()
    }

    fn get_all_exported_adapters(&self) -> Vec<String> {
        self.get_all_exported_resources(Resource::Adapter)
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    fn release_exported_adapter(&self, name: &str) {
        self.release_exported_resources(Resource::Adapter, name)
    }
}

impl SetAdapter for ForwardPass {
    fn set_adapter(&self, adapter_ptr: u32) {
        adapter::set_adapter(&self.inner, adapter_ptr);
    }
}
