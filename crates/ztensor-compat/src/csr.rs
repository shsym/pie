use ztensor::vocab::CsrPlan;
use ztensor::{Error, Leaf, Result, Rule, Tensor, Term};

#[derive(Debug, Clone)]
pub struct Csr {
    pub rows: u64,
    pub cols: u64,
    pub values: Vec<u8>,
    pub term: Term,
    pub indices: Vec<u64>,
    pub indptr: Vec<u64>,
}

pub fn read(tensor: &Tensor<'_>) -> Result<Csr> {
    if tensor.layout() != Some("zt.sparse_csr/2") {
        return Err(Error::Unsupported(format!(
            "{:?} has layout {:?}, not zt.sparse_csr/2",
            tensor.name(),
            tensor.layout().unwrap_or("canonical")
        )));
    }
    let name = tensor.name();
    let plan = CsrPlan::of(name, tensor.shape(), tensor.term(), tensor.attributes())?;
    let bytes = tensor.bytes()?;
    if bytes.len() as u64 != plan.size {
        return Err(Error::reject(
            Rule::Size,
            format!(
                "{name:?}: decoded size {} != the {} sparse_csr requires",
                bytes.len(),
                plan.size
            ),
        ));
    }

    let indices = widen(&bytes[plan.indices.range()], plan.index);
    let indptr = widen(&bytes[plan.indptr.range()], plan.index);
    let values_at = plan.values.first().map_or(plan.size, |p| p.offset);
    let values = bytes[values_at as usize..].to_vec();
    let (rows, cols, nnz) = (plan.rows, plan.cols, plan.nnz);

    let bad = |detail: String| Err(Error::reject(Rule::LayoutData, detail));
    if indptr.first() != Some(&0) {
        return bad(format!("{name:?}: indptr must start at 0"));
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return bad(format!("{name:?}: indptr must be non-decreasing"));
    }
    if indptr.last() != Some(&nnz) {
        return bad(format!("{name:?}: indptr must end at nnz ({nnz})"));
    }
    for r in 0..rows as usize {
        let row = &indices[indptr[r] as usize..indptr[r + 1] as usize];
        if row.windows(2).any(|w| w[0] >= w[1]) {
            return bad(format!("{name:?}: row {r} indices not strictly increasing"));
        }
        if row.last().is_some_and(|&c| c >= cols) {
            return bad(format!("{name:?}: row {r} has an index >= cols ({cols})"));
        }
    }

    Ok(Csr {
        rows,
        cols,
        values,
        term: tensor.term().expect("CsrPlan::of requires a type").clone(),
        indices,
        indptr,
    })
}

fn widen(bytes: &[u8], index: Leaf) -> Vec<u64> {
    match index {
        Leaf::U32 => bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as u64)
            .collect(),
        _ => bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}
