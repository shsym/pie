enum ObjectKind {
    KvBlock,
    Emb,
    Dist,
}
type ObjectId = u32;
type CommandId = u32;
struct Allocate {
    object_kind: ObjectKind,
    object_ids: Vec<ObjectId>,
}

struct Deallocate {
    object_kind: ObjectKind,
    object_id: Vec<ObjectId>,
}

struct EmbedText {
    embs: Vec<ObjectId>,
    token_ids: Vec<u32>,
    position_ids: Vec<u32>,
}

struct EmbedImageInner {
    emb_ids: Vec<ObjectId>,
    url: String,
}

struct EmbedImage {
    inner: Vec<EmbedImageInner>,
}

struct FillBlockInner {
    block: ObjectId,
    context_blocks: Vec<ObjectId>,
    input_embs: Vec<ObjectId>,
    output_embs: Option<Vec<ObjectId>>,
}

struct FillBlock {
    inner: Vec<FillBlockInner>,
}

struct MaskBlock {
    block: Vec<ObjectId>,
    mask: Vec<Vec<bool>>,
}

struct CopyBlockInner {
    src_block: ObjectId,
    dst_block: ObjectId,
    src_start: u32,
    dst_start: u32,
    length: u32,
}

struct CopyBlock {
    inner: Vec<CopyBlockInner>,
}

struct DecodeRequest {
    handle: CommandId,
    embs: Vec<ObjectId>,
}

struct DecodeResponse {
    handle: CommandId,
    tokens: Vec<u32>,
}
