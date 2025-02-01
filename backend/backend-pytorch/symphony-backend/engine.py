from __future__ import annotations

import functools
import time
from typing import Any

import numpy as np
import torch

from .block import Block, BlockStorage, BlockLocation
from .command import Command, Transform, Yield, Terminate, Next, Read
from .common import BlockId, ThreadId, cdiv, uid
from .model import Model
from .thread import Thread, ThreadState
from .tokenizer import load_tokenizer


def scheduling_policy(t1: Thread, t2: Thread):
    # if the thread has not reached the minimum ticks
    if not t1.ran_min_ticks() and t1.ran_min_ticks():
        return True

    return t1.ticks < t2.ticks


class TaskBatch:
    tasks: list[Task]
    token_ids: torch.Tensor  # (len(tasks), BLOCK_SIZE)
    position_offsets: torch.Tensor  # (len(tasks), 1)
    kv_drain_addr_lut: torch.Tensor  # (len(tasks), 1)
    kv_lut: torch.Tensor  # (N, BLOCKS_PER_BATCH)
    mask_lut: torch.Tensor  # (N, BLOCKS_PER_BATCH)
    q_lut: torch.Tensor  # (N, 1)
    reduce_grp_lut: torch.Tensor  # (len(tasks), N)

    block_size: int
    blocks_per_batch_item: int

    def __init__(self, tasks: list[Task], block_size: int, blocks_per_batch_item: int):

        self.tasks = tasks
        self.block_size = block_size
        self.blocks_per_batch_item = blocks_per_batch_item

        if len(tasks) > 0:
            self.construct_batch()

    def num_tasks(self):
        return len(self.tasks)

    def batch_size(self):
        return self.kv_lut.shape[0]

    def num_blocks_per_batch(self):
        return self.kv_lut.shape[1]

    def max_grp_size(self):
        return self.reduce_grp_lut.shape[1]

    def to(self, device: torch.device):
        self.token_ids = self.token_ids.to(device)
        self.position_offsets = self.position_offsets.to(device)
        self.kv_drain_addr_lut = self.kv_drain_addr_lut.to(device)
        self.kv_lut = self.kv_lut.to(device)
        self.mask_lut = self.mask_lut.to(device)
        self.q_lut = self.q_lut.to(device)
        self.reduce_grp_lut = self.reduce_grp_lut.to(device)

        return self

    def construct_batch(self):

        token_ids = np.zeros((len(self.tasks), self.block_size), dtype=np.int32)
        position_offsets = np.zeros((len(self.tasks, )), dtype=np.int32)
        kv_drain_addr_lut = np.zeros((len(self.tasks, )), dtype=np.int32)

        sub_batch_list = [cdiv(len(task.kv_addrs), self.blocks_per_batch_item) for task in self.tasks]

        batch_size = sum(sub_batch_list)
        max_grp_size = max(sub_batch_list)

        kv_addr_lut = np.zeros((batch_size, self.blocks_per_batch_item), dtype=np.int32)
        q_addr_lut = np.zeros((batch_size, 1), dtype=np.int32)
        mask_lut = np.zeros((batch_size, self.blocks_per_batch_item), dtype=np.int32)
        reduce_grp_lut = np.zeros((len(self.tasks), max_grp_size), dtype=np.int32)

        offset = 0

        for i, task in enumerate(self.tasks):

            token_ids[i, :len(task.token_ids)] = task.token_ids
            position_offsets[i] = task.pos_offset
            kv_drain_addr_lut[i] = task.kv_new_addr

            sub_size = cdiv(len(task.kv_addrs), self.blocks_per_batch_item)
            for j in range(sub_size):
                sub_kv = task.kv_addrs[j * self.blocks_per_batch_item: (j + 1) * self.blocks_per_batch_item]
                sub_mask = task.mask[j * self.blocks_per_batch_item: (j + 1) * self.blocks_per_batch_item]
                kv_addr_lut[offset + j, :len(sub_kv)] = sub_kv
                mask_lut[offset + j, :len(sub_mask)] = sub_mask

                assert len(sub_kv) == len(sub_mask)

            q_addr_lut[offset: offset + sub_size] = i
            reduce_grp_lut[i, :sub_size] = list(range(offset, offset + sub_size))
            offset += sub_size

        self.token_ids = torch.tensor(token_ids, dtype=torch.int32)
        self.position_offsets = torch.tensor(position_offsets, dtype=torch.int32)
        self.kv_drain_addr_lut = torch.tensor(kv_drain_addr_lut, dtype=torch.int32)
        self.kv_lut = torch.tensor(kv_addr_lut, dtype=torch.int32)
        self.mask_lut = torch.tensor(mask_lut, dtype=torch.int32)
        self.q_lut = torch.tensor(q_addr_lut, dtype=torch.int32)
        self.reduce_grp_lut = torch.tensor(reduce_grp_lut, dtype=torch.int32)


class Task:
    token_ids: list[int]
    pos_offset: int

    new_block_id: int
    block_ids: list[int]

    kv_new_addr: int
    kv_addrs: list[int]
    mask: list[int]

    def __init__(self, token_ids: list[int], pos_offset: int, new_block_id: int, block_ids: list[int], mask: list[int]):
        self.token_ids = token_ids
        self.pos_offset = pos_offset
        self.new_block_id = new_block_id
        self.block_ids = block_ids
        self.mask = mask


class Engine:
    storage_cpu: BlockStorage
    storage_gpu: BlockStorage

    BLOCK_SIZE = 32
    DIST_NUM_VARS = 64

    MAX_THREAD_IDLE_TIME = 100  # after 100 ticks, the thread ctx cache is removed from the CPU storage

    # total # blocks in a single compute cycle = MAX_BATCH_SIZE * BLOCKS_PER_BATCH_ITEM
    MAX_BATCH_SIZE = 128
    BLOCKS_PER_BATCH_ITEM = 32

    threads: dict[ThreadId, Thread]
    blocks: dict[BlockId, Block]

    model: Model

    # debug
    verbose: bool

    last_time: float

    def __init__(self, model: Model,
                 gpu_device: str,
                 gpu_num_blocks: int,
                 cpu_num_blocks: int,
                 block_size: int,
                 dist_num_vars: int = 64,
                 max_thread_idle_time: int = 100,
                 max_batch_size: int = 128,
                 blocks_per_batch_item: int = 32,
                 verbose: bool = False
                 ):

        self.model = model
        self.tokenizer = load_tokenizer(model.model_name)
        num_layers, num_head, block_dim = self.model.kv_shape()

        self.storage_gpu = BlockStorage(num_layers, gpu_num_blocks, num_head, block_size, block_dim, torch.device(gpu_device))
        self.storage_cpu = BlockStorage(num_layers, cpu_num_blocks, num_head, block_size, block_dim, torch.device("cpu"))

        if verbose:
            print(f"GPU storage: {gpu_num_blocks} blocks / {self.storage_gpu.num_bytes() // 1024 // 1024} MB total")
            print(f"CPU storage: {cpu_num_blocks} blocks / {self.storage_cpu.num_bytes() // 1024 // 1024} MB total")

        self.threads = {}
        self.blocks = {}
        self.BLOCK_SIZE = block_size
        self.DIST_NUM_VARS = dist_num_vars
        self.MAX_THREAD_IDLE_TIME = max_thread_idle_time
        self.MAX_BATCH_SIZE = max_batch_size
        self.BLOCKS_PER_BATCH_ITEM = blocks_per_batch_item

        self.verbose = verbose
        self.last_time = time.time()

    def create_block(self, token_ids, prev_bid) -> BlockId:

        block = Block(token_ids, self.blocks[prev_bid] if prev_bid else None)
        b_id = uid("block")

        self.blocks[b_id] = block

        return b_id

    def suspend_block(self, bid: BlockId, recompute: bool = False):

        block = self.blocks[bid]

        if block.location == BlockLocation.GPU:
            if self.storage_cpu.num_free_blocks() > 0 and not recompute:
                print(f"[engine] suspending block {bid}...")

                self.storage_gpu.relocate(bid, self.storage_cpu)
                block.location = BlockLocation.CPU
            else:
                # recomputation
                self.storage_gpu.delete(bid)
                block.location = BlockLocation.REMOVED

        elif block.location == BlockLocation.CPU:

            if recompute:
                self.storage_cpu.delete(bid)
                block.location = BlockLocation.REMOVED

    def destroy_block(self, bid: BlockId):

        if bid not in self.blocks:
            raise ValueError(f"Block {bid} does not exist")

        if len(self.blocks[bid].refs) > 0:
            raise ValueError(f"Block {bid} is still in use")

        if self.storage_gpu.has_block(bid):
            self.storage_gpu.delete(bid)

        if self.storage_cpu.has_block(bid):
            self.storage_cpu.delete(bid)

        del self.blocks[bid]

    def search_block(self, token_ids: list[int], prev_bid: int | None = None) -> BlockId | None:
        prev_block = self.blocks[prev_bid] if prev_bid else None

        # compute the context hash
        if prev_block:
            ctx_hash = str(hash(prev_block.context_hash + str(token_ids)))

        else:
            ctx_hash = str(hash(str(token_ids)))

        # check if the block is already in the storage
        for b_id, block in self.blocks.items():
            if block.token_ids == token_ids and block.location != BlockLocation.REMOVED and block.context_hash == ctx_hash:
                return b_id

        return None

    def create_thread(self, parent_tid: ThreadId | None) -> ThreadId:

        tid = uid("thread")
        thread = Thread()

        # inherit the context from the parent thread
        if parent_tid:
            parent = self.threads[parent_tid]
            thread.block_ids.extend(parent.block_ids)
            thread.command_queue.extend(parent.command_queue)
            thread.ticks = parent.ticks
            thread.min_ticks = parent.min_ticks
            thread.idle_time = parent.idle_time

            for b_id in thread.block_ids:
                self.blocks[b_id].refs.append(tid)

        self.threads[tid] = thread

        return tid

    def suspend_thread(self, tid: ThreadId):

        if tid not in self.threads:
            raise ValueError(f"Thread {tid} does not exist")

        if self.verbose:
            print(f"[engine] Suspending thread {tid}...")

        thread = self.threads[tid]
        thread.state = ThreadState.SUSPENDED

        # this ensures that this thread will get the priority in the next scheduling when it is resumed
        thread.ticks = (thread.ticks // thread.min_ticks) * thread.min_ticks

        for b_id in thread.block_ids:
            # check if all referenced threads are suspended
            if all(self.threads[tid].state == ThreadState.SUSPENDED for tid in self.blocks[b_id].refs):
                self.suspend_block(b_id)

    def destroy_thread(self, tid: ThreadId):

        if tid not in self.threads:
            raise ValueError(f"Thread {tid} does not exist")

        thread = self.threads[tid]

        # resolve all the proceeding commands in the queue
        for cmd in thread.command_queue:
            if cmd.is_blocking():
                cmd.resolve(None)

        for b_id in thread.block_ids:
            self.blocks[b_id].refs.remove(tid)

            if len(self.blocks[b_id].refs) == 0:
                self.destroy_block(b_id)

        del self.threads[tid]

    async def command(self, tid: ThreadId, cmd: Command, immediate: bool = False) -> Any:

        if tid not in self.threads:
            raise ValueError(f"Thread {tid} does not exist")

        thread = self.threads[tid]

        if immediate:
            thread.command_queue.appendleft(cmd)
        else:
            thread.command_queue.append(cmd)

        if cmd.is_blocking():
            return await cmd.response()
        else:
            return None

    def schedule(self, elapsed_time: float = 0.0):

        # first check the validity some obvious things
        assert self.MAX_BATCH_SIZE * self.BLOCKS_PER_BATCH_ITEM <= self.storage_gpu.num_blocks()

        # build inference buffer
        runnables = []
        suspended = []
        terminated = []
        idle = []

        for t_id, thread in self.threads.items():
            # get the threads that are:
            # 1. has a transform command
            while len(thread.command_queue) > 0:

                cmd = thread.command_queue[0]

                if isinstance(cmd, Transform):
                    runnables.append(t_id)
                    break

                elif isinstance(cmd, Next):

                    last_block_id = thread.block_ids[-1]
                    last_block = self.blocks[last_block_id]

                    # TODO: when the n is greater than the number of tokens in the last block
                    l = len(last_block.token_ids)

                    response = {
                        "token_ids": last_block.next_token_id[:l][-cmd.n:].tolist(),
                        "probs": last_block.next_token_probs[:l][-cmd.n:].tolist(),
                        "probs_rem": last_block.next_token_probs_rem[:l][-cmd.n:].tolist()
                    }

                    cmd.resolve(response)
                    thread.command_queue.popleft()

                elif isinstance(cmd, Read):

                    cat_token_ids = []
                    cat_probs = []
                    cat_probs_rem = []

                    for i in cmd.indices:
                        block_idx = i // self.BLOCK_SIZE
                        offset = i % self.BLOCK_SIZE

                        block_id = thread.block_ids[block_idx]
                        block = self.blocks[block_id]

                        cat_token_ids.append(block.next_token_id[offset].tolist())
                        cat_probs.append(block.next_token_probs[offset].tolist())
                        cat_probs_rem.append(block.next_token_probs_rem[offset].tolist())

                    response = {
                        "token_ids": cat_token_ids,
                        "probs": cat_probs,
                        "probs_rem": cat_probs_rem
                    }
                    cmd.resolve(response)
                    thread.command_queue.popleft()

                else:
                    # remove the command from the queue
                    thread.command_queue.popleft()

                    if isinstance(cmd, Yield):
                        suspended.append(t_id)

                    elif isinstance(cmd, Terminate):
                        terminated.append(t_id)

                    break

            else:
                idle.append(t_id)

        # terminate the threads that are marked for termination
        for t_id in terminated:
            self.destroy_thread(t_id)

        # suspend the threads that are marked for suspension
        for t_id in suspended:
            self.suspend_thread(t_id)

        # update the idle ticks
        for t_id in idle:

            thread = self.threads[t_id]
            thread.idle_time += elapsed_time
            # print('idle time', thread.idle_time, thread.state, self.MAX_THREAD_IDLE_TIME)
            if thread.idle_time > self.MAX_THREAD_IDLE_TIME:
                # archive the threads referencing this block

                if thread.state == ThreadState.RUNNING:
                    self.suspend_thread(t_id)

        # total_slots

        # num_slots * num cycles

        # sort the runnables based on the scheduling policy
        runnables_sorted = sorted(runnables, key=functools.cmp_to_key(
            lambda x, y: scheduling_policy(self.threads[x], self.threads[y])
        ))

        tasks = []
        needed_blocks = set()

        # from the highest priority to the lowest
        for t_id in runnables_sorted:

            # fill the slots until it is full
            thread = self.threads[t_id]
            thread.idle_time = 0
            ###### HANDLE RECOMPUTATION ######

            found_removable = False
            removable_idx = -1
            new_cmds = []
            for i, b_id in enumerate(thread.block_ids):
                # if the block is removed, then it has to be recomputed again before current command can be executed
                # note that if a block is REMOVED, then it is guaranteed that all subsequent blocks are REMOVED as well.

                block = self.blocks[b_id]
                if block.location == BlockLocation.REMOVED:
                    if not found_removable:
                        removable_idx = i  # mark the first removable block
                    found_removable = True

                if found_removable:

                    # create a new command that will recompute the block
                    cmd = Transform(self.blocks[b_id].token_ids.copy())
                    new_cmds.append(cmd)

                    # remove the block from the thread
                    block.refs.remove(t_id)

                    if len(block.refs) == 0:
                        self.destroy_block(b_id)

            # remove the blocks that are not needed anymore
            if found_removable:
                del thread.block_ids[removable_idx:]
                thread.command_queue.extendleft(reversed(new_cmds))

            ############## merge transforms. ############################
            token_buffer = []

            while len(thread.command_queue) > 0:
                cmd = thread.command_queue[0]
                if isinstance(cmd, Transform):
                    token_buffer.extend(cmd.token_ids)
                    thread.command_queue.popleft()
                else:
                    break

            new_cmds = []
            for i in range(cdiv(len(token_buffer), self.BLOCK_SIZE)):
                start = i * self.BLOCK_SIZE
                end = min((i + 1) * self.BLOCK_SIZE, len(token_buffer))
                new_cmd = Transform(token_buffer[start:end].copy())
                new_cmds.append(new_cmd)

            # NOTE: reversed() is needed because .extendleft() traverses the list in reverse order
            thread.command_queue.extendleft(reversed(new_cmds))

            #########################################################

            num_batch = 0

            while len(thread.command_queue) > 0 and num_batch < self.MAX_BATCH_SIZE:

                cmd = thread.command_queue[0]
                if not isinstance(cmd, Transform):
                    break

                # print('token buffer', cmd.token_ids)

                # check if the block is already in the storage
                block_needs_compute = False
                b_id = self.search_block(cmd.token_ids, thread.block_ids[-1] if thread.block_ids else None)

                if b_id is None:

                    # first check if this is computable given the number of available slots
                    num_batch_needed = cdiv(len(thread.block_ids) + 1, self.BLOCKS_PER_BATCH_ITEM)

                    if num_batch + num_batch_needed > self.MAX_BATCH_SIZE:
                        break

                    block_needs_compute = True
                    b_id = self.create_block(cmd.token_ids, thread.block_ids[-1] if thread.block_ids else None)

                # remove the command from the queue, since all the checks are done
                thread.command_queue.popleft()

                # add the block to the thread
                block = self.blocks[b_id]
                block.refs.append(t_id)
                thread.block_ids.append(b_id)

                if block_needs_compute:
                    num_slots_needed = cdiv(len(thread.block_ids), self.BLOCKS_PER_BATCH_ITEM)
                    num_batch += num_slots_needed
                    thread.ticks += num_slots_needed
                    num_ctx_blocks = len(thread.block_ids) - 1

                    task = Task(
                        token_ids=cmd.token_ids.copy(),
                        pos_offset=num_ctx_blocks * self.BLOCK_SIZE,
                        new_block_id=b_id,
                        block_ids=thread.block_ids.copy(),
                        mask=[1] * num_ctx_blocks + [2]
                    )
                    tasks.append(task)

            needed_blocks.update(thread.block_ids)
            thread.ticks_idle = 0
            thread.state = ThreadState.RUNNING

            if num_batch == self.MAX_BATCH_SIZE:
                break

        # increase the idle ticks for the blocks that are not needed
        newly_needed = 0
        idle_blocks_in_gpu = []

        for b_id, block in self.blocks.items():
            if b_id in needed_blocks:
                block.idle_ticks = 0
                if block.location != BlockLocation.GPU:
                    newly_needed += 1
            else:
                block.idle_ticks += 1

                if block.location == BlockLocation.GPU:
                    idle_blocks_in_gpu.append(b_id)

        # check if there are enough slots, if not free the ones that are not needed
        num_blocks_to_free = max(newly_needed - self.storage_gpu.num_free_blocks(), 0)

        if num_blocks_to_free > 0:
            # sort the blocks.keys() based on the idle ticks
            idle_blocks_in_gpu = sorted(idle_blocks_in_gpu, key=lambda x: self.blocks[x].idle_ticks, reverse=True)

            # free the top-n blocks
            for i in range(num_blocks_to_free):
                self.suspend_block(idle_blocks_in_gpu[i])

        # prepare the required blocks ready by ensuring that they are in the GPU storage
        for b_id in needed_blocks:
            block = self.blocks[b_id]

            match block.location:
                case BlockLocation.GPU:
                    # nothing to do... just pass
                    pass
                case BlockLocation.CPU:
                    print(f"[engine] Relocating block {b_id} to GPU storage...")
                    # move the block to the GPU storage. it is guaranteed that there is enough space in the GPU storage
                    self.storage_cpu.relocate(b_id, self.storage_gpu)
                    block.location = BlockLocation.GPU

                case BlockLocation.REMOVED:
                    # this is impossible. throw an error
                    raise ValueError(f"Block {b_id} is removed")

                case BlockLocation.SCHEDULED:
                    # create a new block in the GPU storage
                    self.storage_gpu.create(b_id)
                    block.location = BlockLocation.GPU

                case _:
                    raise ValueError(f"Invalid block location {block.location}")

        # now ready to execute the slots

        # slots : slot_targets

        # link block addrs in task
        # print('total tasks', len(tasks))
        for task in tasks:
            task.kv_addrs = self.storage_gpu.translate_addr(task.block_ids)
            task.kv_new_addr = self.storage_gpu.translate_addr(task.new_block_id)

            # print('task', task.kv_addrs, task.mask)

        return tasks

    @torch.inference_mode()
    def step(self) -> bool:

        elapsed_time = time.time() - self.last_time
        self.last_time = time.time()

        tasks = self.schedule(elapsed_time)

        if len(tasks) == 0:
            return False

        # prepare the batch
        task_batch = TaskBatch(tasks, self.BLOCK_SIZE, self.BLOCKS_PER_BATCH_ITEM)
        #
        # for i, task in enumerate(tasks):
        #     print(f'<<<<<<<batch {i}>>>>>>>>')
        #     print(f'target block: {task.new_block_id}, ctx blocks: {task.block_ids}')
        #     print('token ids', len(task.token_ids), task.token_ids)
        #     print('------------------------------------')
        #     print(self.tokenizer.decode(task.token_ids))
        #     print('------------------------------------')
        #     print('pos offset', task.pos_offset)
        #     print('task batch size', task_batch.batch_size())
        #     print("=================================================================================")

        task_batch = task_batch.to(self.storage_gpu.device)

        # execute the batch
        logits = self.model(self.storage_gpu.ptr, task_batch)

        # reduce the logits to top-64
        next_token_probs, next_token_ids = torch.topk(torch.softmax(logits, dim=-1), self.DIST_NUM_VARS, dim=-1)

        next_token_probs_rem = 1 - torch.sum(next_token_probs, dim=-1)  # (len(tasks), BLOCK_SIZE)

        next_token_ids = next_token_ids.numpy(force=True)
        next_token_probs = next_token_probs.numpy(force=True)
        next_token_probs_rem = next_token_probs_rem.numpy(force=True)

        # traverse the target blocks and find the "unfilled" ones.
        for i, task in enumerate(tasks):

            block = self.blocks[task.new_block_id]

            # dispatch the logits
            block.next_token_id = next_token_ids[i]
            block.next_token_probs = next_token_probs[i]
            block.next_token_probs_rem = next_token_probs_rem[i]

            if len(task.token_ids) < self.BLOCK_SIZE:
                # simply drop the block. it will be recomputed along with the remaining tokens in the next iteration
                self.storage_gpu.delete(task.new_block_id)
                block.location = BlockLocation.REMOVED

        return True
