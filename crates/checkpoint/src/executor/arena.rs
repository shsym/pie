use std::borrow::Cow;

use crate::error::Error;

pub trait ArenaBacking {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error>;

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error>;

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error>;

    fn finish(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn runs_named_kernels(&self) -> bool {
        false
    }

    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        Err(Error::Contract(format!(
            "this arena backing was offered the kernel `{}` and has no \
             launcher for anything: `runs_named_kernels` said true",
            op.kernel
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSpan {
    pub offset: usize,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub struct TileMapOp<'a> {
    pub kernel: &'a str,
    pub src: ArenaSpan,
    pub dst: ArenaSpan,
    pub dst_scales: Option<ArenaSpan>,
    pub factors: Option<ArenaSpan>,
    pub shape: Option<(u32, u32)>,
}

fn out_of_bounds(what: &str) -> Error {
    Error::Contract(format!("arena {what} is out of bounds"))
}

impl ArenaBacking for &mut [u8] {
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| out_of_bounds("read"))?;
        self.get(offset..end)
            .map(Cow::Borrowed)
            .ok_or_else(|| out_of_bounds("read"))
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| out_of_bounds("write"))?;
        self.get_mut(offset..end)
            .ok_or_else(|| out_of_bounds("write"))?
            .copy_from_slice(bytes);
        Ok(())
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| out_of_bounds("fill"))?;
        self.get_mut(offset..end)
            .ok_or_else(|| out_of_bounds("fill"))?
            .fill(byte);
        Ok(())
    }
}
