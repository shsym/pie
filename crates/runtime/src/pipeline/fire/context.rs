use wasmtime::component::ResourceTable;

pub trait FireContext {
    fn resources(&mut self) -> &mut ResourceTable;

    fn process_id(&self) -> uuid::Uuid;

    async fn settle_pipeline_tail(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}
