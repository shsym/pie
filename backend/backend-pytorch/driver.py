from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

from common import ceil_div
from l4ma import AttentionStorage, VectorStorage
from llama import LlamaForCausalLM
from sdi_pb2 import BatchAllocate, BatchDeallocate, BatchEmbedText, BatchEmbedImage, BatchMaskBlock, BatchCopyBlock, BatchDecodeTokenDistribution, BatchSampleTopKRequest, BatchSampleTopKResponse, \
    ObjectKind, SampleTopKResponse, BatchGetTokenDistributionRequest, BatchGetTokenDistributionResponse, BatchFillBlock

NUM_TOKENS_IN_BLOCK = 16


@dataclass
class TextEmbed:
    token_id: int
    position_id: int


@dataclass
class ImageEmbed:
    vec_id: int
    position_id: (int, int)


@dataclass
class Block:
    position_ids: np.ndarray
    occupancy: np.ndarray


EMPTY_BLOCK = Block(
    position_ids=np.array([0] * NUM_TOKENS_IN_BLOCK),
    occupancy=np.array([False] * NUM_TOKENS_IN_BLOCK)
)

# union type
Embed = Union[TextEmbed, ImageEmbed]


class Driver:
    embeds: dict[int, Embed]
    blocks: dict[int, Block]
    lm: LlamaForCausalLM

    block_storage: AttentionStorage
    embed_storage: VectorStorage
    dist_storage: VectorStorage

    def __init__(self, model, storage: AttentionStorage, embed_storage: VectorStorage, dist_storage: VectorStorage):
        self.embeds = {}
        self.blocks = {}

        self.lm = model
        self.block_storage = storage
        self.embed_storage = embed_storage
        self.dist_storage = dist_storage

    def device(self):
        return self.block_storage.ptr.device

    def dtype(self):
        return self.block_storage.ptr.dtype

    def allocate(self, cmds: BatchAllocate):
        # in current implementation, all allocations are already done in the constructor.
        # but in the future, we may want to allocate more blocks than the GPU capacity, by offloading some of the blocks to the CPU memory.
        # This logic should handle that case.

        for cmd in cmds.items:
            if cmd.kind == ObjectKind.OBJECT_KIND_KV_BLOCK:
                for i in range(cmd.count):
                    self.blocks[cmd.object_id_offset + i] = Block(
                        position_ids=np.array([0] * NUM_TOKENS_IN_BLOCK),
                        occupancy=np.array([False] * NUM_TOKENS_IN_BLOCK)
                    )

            elif cmd.kind == ObjectKind.OBJECT_KIND_EMB:
                # do nothing. Embeds are allocated on the fly.
                ...
            elif cmd.kind == ObjectKind.OBJECT_KIND_DIST:
                # do nothing. Dists are allocated on the fly.
                ...

    def deallocate(self, cmds: BatchDeallocate):
        # in current implementation, all allocations are already done in the constructor.
        # so we don't need to deallocate anything.
        ...

    def embed_text(self, cmds: BatchEmbedText):
        for cmd in cmds.items:
            self.embeds[cmd.embedding_id] = TextEmbed(token_id=cmd.token_id, position_id=cmd.position_id)

    def embed_image(self, cmds: BatchEmbedImage):
        # unimplemented
        ...

    def mask_block(self, cmds: BatchMaskBlock):
        for cmd in cmds.items:
            block = self.blocks[cmd.block_id]
            for i, m in enumerate(cmd.mask):
                block.occupancy[i] = m

    def copy_block(self, cmds: BatchCopyBlock):

        # TODO.
        ...

    def decode_token_distribution(self, cmds: BatchDecodeTokenDistribution):

        # TODO -> make this more efficient by batching.
        for i, cmd in enumerate(cmds.items):
            dist = torch.softmax(self.lm.lm_head(self.embed_storage.ptr[cmd.embedding_id]), dim=-1)
            self.dist_storage.ptr[cmd.distribution_id] = dist

    def sample_top_k_request(self, cmds: BatchSampleTopKRequest) -> BatchSampleTopKResponse:
        res = []
        for i, cmd in enumerate(cmds.items):
            dist = self.dist_storage.ptr[cmd.distribution_id]
            topk_res = torch.topk(dist, k=cmd.k)
            topk_tokens = topk_res.indices.tolist()
            topk_probs = topk_res.values.tolist()

            res.append(SampleTopKResponse(token_ids=topk_tokens))

        return BatchSampleTopKResponse(items=res)

    def get_token_distribution(self, cmds: BatchGetTokenDistributionRequest) -> BatchGetTokenDistributionResponse:

        # get the truncated token distribution. TODO
        ...

    def fill_block(self, cmds: BatchFillBlock):

        ### Step 1.Decide the `chunk size`
        # first estimate the `chunk_size` for the batch (chunk = number of blocks in one batch row)
        # if chunk size is too large -> performance will be nice, but most blocks will be "empty" ones, leading to wasted computation.
        # if chunk size is too small -> there will be only few empty blocks, but the performance could be bad for fill requests with very large contexts.
        # so we need to find a balance between the two, which I use the median of the number of context blocks in the commands.
        # for cmd in cmds.items:
        #     print(cmd.block_id)
        #     print(cmd.context_block_ids)
        #     print(cmd.input_embedding_ids)
        #     print(cmd.output_embedding_ids)

        num_blocks_per_req = [len(cmd.context_block_ids) for cmd in cmds.items]
        NUM_BLOCKS_IN_CHUNK = int(np.max(num_blocks_per_req))
        NUM_TOKENS_IN_CHUNK = NUM_BLOCKS_IN_CHUNK * NUM_TOKENS_IN_BLOCK

        num_chunks_per_req = [ceil_div(n, NUM_BLOCKS_IN_CHUNK) for n in num_blocks_per_req]

        reduce_grps = np.zeros((len(cmds.items), max(num_chunks_per_req)), dtype=np.int32)  # 2d (NUM_CMDS, MAX_NUM_CHUNKS)
        new_q_lut = np.zeros((sum(num_chunks_per_req), 1), dtype=np.int32)
        new_kv_lut = np.zeros((len(cmds.items), 1), dtype=np.int32)
        all_kv_lut = np.zeros((sum(num_chunks_per_req), NUM_BLOCKS_IN_CHUNK), dtype=np.int32)
        masks = np.zeros((sum(num_chunks_per_req), NUM_TOKENS_IN_BLOCK, NUM_TOKENS_IN_CHUNK), dtype=np.bool_)

        new_token_ids = np.zeros((len(cmds.items), NUM_TOKENS_IN_BLOCK), dtype=np.int32)
        POS_DIM = 1
        new_position_ids = np.zeros((len(cmds.items), NUM_TOKENS_IN_BLOCK * POS_DIM), dtype=np.int32)
        input_embed_postproc = []
        output_embed_postproc = []
        k = 0
        for i, cmd in enumerate(cmds.items):

            # update the block metadata

            tgt_block = self.blocks[cmd.block_id]
            for j in range(NUM_TOKENS_IN_BLOCK):

                # process input embeds
                if j < len(cmd.input_embedding_ids) and cmd.input_embedding_ids[j] in self.embeds:
                    embed = self.embeds[cmd.input_embedding_ids[j]]

                    tgt_block.occupancy[j] = True
                    tgt_block.position_ids[j] = embed.position_id

                    if isinstance(embed, TextEmbed):
                        new_token_ids[i, j] = embed.token_id

                    else:
                        # just fill in stubs. It will be filled in later
                        new_token_ids[i, j] = 0

                        # get the image token
                        input_embed_postproc.append({
                            "row": i,
                            "col": j,
                            "vec_id": embed.vec_id,
                            "pos_id": embed.position_id
                        })
                else:
                    new_token_ids[i, j] = 0
                    tgt_block.occupancy[j] = False
                    tgt_block.position_ids[j] = 0

                # process output embeds
                if j < len(cmd.output_embedding_ids) and cmd.output_embedding_ids[j] >= 0:
                    output_embed_postproc.append({
                        "row": i,
                        "col": j,
                        "vec_id": cmd.output_embedding_ids[j],
                    })

            # mask: (len(ctx_ids) * block_size, block_size)
            ctx_block_ids = cmd.context_block_ids
            tgt_block_id = cmd.block_id

            num_chunks = ceil_div(len(ctx_block_ids), NUM_BLOCKS_IN_CHUNK)

            # gather position ids and occupancy masks for all context blocks
            ctx_pos_ids = np.hstack([self.blocks[ctx_id].position_ids for ctx_id in ctx_block_ids])  # int
            ctx_occupancy = np.hstack([self.blocks[ctx_id].occupancy for ctx_id in ctx_block_ids])  # bool
            tgt_pos_ids = np.array(self.blocks[tgt_block_id].position_ids)  # int
            tgt_occupancy = np.array(self.blocks[tgt_block_id].occupancy)  # bool

            # pad them to the NUM_TOKENS_IN_CHUNK
            # print("NUM_TOKENS_IN_CHUNK", NUM_TOKENS_IN_CHUNK, NUM_TOKENS_IN_CHUNK - len(ctx_pos_ids))

            new_kv_lut[i] = tgt_block_id
            new_position_ids[i] = tgt_pos_ids

            for j in range(num_chunks):
                start_b = j * NUM_BLOCKS_IN_CHUNK
                end_b = min(start_b + NUM_BLOCKS_IN_CHUNK, len(ctx_block_ids))

                new_q_lut[k] = i
                new_kv_lut[i] = tgt_block_id
                all_kv_lut[k, :end_b - start_b] = ctx_block_ids[start_b:end_b]

                # ctx_pos_ids[k, :end - start] = ctx_pos_ids[start:end]
                # ctx_occupancy[k, :end - start] = ctx_occupancy[start:end]

                start_t = j * NUM_TOKENS_IN_CHUNK
                end_t = min(start_t + NUM_TOKENS_IN_CHUNK, len(ctx_pos_ids))

                chunk_ctx_pos_ids = ctx_pos_ids[start_t:end_t]
                chunk_ctx_occupancy = ctx_occupancy[start_t:end_t]

                if len(chunk_ctx_pos_ids) < NUM_TOKENS_IN_CHUNK:
                    chunk_ctx_pos_ids = np.pad(chunk_ctx_pos_ids, (0, NUM_TOKENS_IN_CHUNK - len(chunk_ctx_pos_ids)), mode='constant', constant_values=0)
                    chunk_ctx_occupancy = np.pad(chunk_ctx_occupancy, (0, NUM_TOKENS_IN_CHUNK - len(chunk_ctx_occupancy)), mode='constant', constant_values=False)

                casual_mask = chunk_ctx_pos_ids[None, :] <= tgt_pos_ids[:, None]
                valid_mask = np.logical_and(chunk_ctx_occupancy[None, :], tgt_occupancy[:, None])

                # print(ctx_pos_ids.shape)
                # print(casual_mask.shape)
                # print(valid_mask.shape)

                masks[k] = np.logical_and(casual_mask, valid_mask)

                # print("masks[k]", masks[k].astype(int))

                # if all items in the chunk are False, then it will cause NaN in softmax. Check:
                if not masks[k].any():
                    print('Warning: All items in the chunk are False. This may cause NaN in softmax.')

                reduce_grps[i, j] = k

                k += 1

        pt_reduce_grps = torch.as_tensor(reduce_grps, device=self.device(), dtype=torch.int32)
        pt_new_q_lut = torch.as_tensor(new_q_lut, device=self.device(), dtype=torch.int32)
        pt_new_kv_lut = torch.as_tensor(new_kv_lut, device=self.device(), dtype=torch.int32)
        pt_all_kv_lut = torch.as_tensor(all_kv_lut, device=self.device(), dtype=torch.int32)
        pt_masks = torch.as_tensor(masks, device=self.device(), dtype=torch.bool)
        pt_new_token_ids = torch.as_tensor(new_token_ids, device=self.device(), dtype=torch.int32)
        pt_new_position_ids = torch.as_tensor(new_position_ids, device=self.device(), dtype=torch.int32)

        # token ids
        # print("pt_new_q_lut", pt_new_q_lut)
        # print("pt_new_kv_lut", pt_new_kv_lut)
        # print("pt_all_kv_lut", pt_all_kv_lut)
        # print("pt_reduce_grps", pt_reduce_grps)
        # print("new_token_ids", pt_new_token_ids)
        # print("new_position_ids", pt_new_position_ids)
        # print("pt_masks", pt_masks)
        # compute the embeddings...
        input_embeds = self.lm.model.embed_tokens(pt_new_token_ids)

        # inject the image embeddings -> replace with torch.scatter later
        for token_map in input_embed_postproc:
            vec_id = token_map["vec_id"]
            print("img emb")
            input_embeds[token_map["row"], token_map["col"]].copy_(self.embed_storage.ptr[vec_id], non_blocking=True)

        # compute the position embeddings
        cos, sin = self.lm.rotary_emb(input_embeds, pt_new_position_ids.max().item() + 1)

        position_embeds = (cos[pt_new_position_ids].unsqueeze(1), sin[pt_new_position_ids].unsqueeze(1))

        logits = self.lm.model.forward(
            input_embeds=input_embeds,
            kv_ptr=self.block_storage.ptr,
            new_q_lut=pt_new_q_lut,
            new_kv_lut=pt_new_kv_lut,
            all_kv_lut=pt_all_kv_lut,
            mask=pt_masks,
            cmd_groups=pt_reduce_grps,
            rope_cache=position_embeds
        )
        # print(logits.shape)
        # store the logits in the output embeds  -> replace with torch.scatter later
        for token_map in output_embed_postproc:
            vec_id = token_map["vec_id"]

            # print("vec_id", vec_id)
            # print(token_map)

            self.embed_storage.ptr[vec_id].copy_(logits[token_map["row"], token_map["col"]], non_blocking=True)
