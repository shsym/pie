use core::convert::TryInto;
use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReadError {
    UnexpectedEof,
    CountTooLarge(&'static str),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::UnexpectedEof => f.write_str("unexpected end of input"),
            ReadError::CountTooLarge(table) => {
                write!(f, "{table} count exceeds what the input can back")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReadError {}

#[derive(Clone, Copy, Debug)]
pub struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    pub fn take(&mut self, count: usize) -> Result<&'a [u8], ReadError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(ReadError::UnexpectedEof)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ReadError::UnexpectedEof)?;
        self.offset = end;
        Ok(value)
    }

    pub fn u8(&mut self) -> Result<u8, ReadError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, ReadError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, ReadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, ReadError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn bounded_count(
        &self,
        raw_count: u32,
        minimum_record_bytes: usize,
        structural_maximum: usize,
        table: &'static str,
    ) -> Result<usize, ReadError> {
        let count = raw_count as usize;
        let minimum_bytes = count
            .checked_mul(minimum_record_bytes)
            .ok_or(ReadError::CountTooLarge(table))?;
        if minimum_record_bytes == 0
            || count > structural_maximum
            || minimum_bytes > self.remaining()
        {
            return Err(ReadError::CountTooLarge(table));
        }
        Ok(count)
    }

    pub fn length(&self, raw_length: u32, table: &'static str) -> Result<usize, ReadError> {
        let length = raw_length as usize;
        if length > self.remaining() {
            return Err(ReadError::CountTooLarge(table));
        }
        Ok(length)
    }
}
