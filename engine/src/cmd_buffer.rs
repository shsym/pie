use rmp_serde::{from_slice, to_vec};
use serde::{Deserialize, Serialize};
use tokio;
use uuid::Uuid;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

/// Run-length encoding for lists of `u32`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RleVec(pub Vec<(u32, u32)>);

/// The high-level commands your system supports.
#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    /// Allocate a certain number of blocks.
    AllocateBlocks(u32),

    /// Allocate a single block.
    AllocateBlock,

    /// Example: Copy from src_block_id to dst_block_id, etc.
    Copy {
        src_block_id: u32,
        dst_block_id: u32,
        src_start: u32,
        dst_start: u32,
        length: u32,
    },

    /// Example: Drop partial data within a block.
    Drop { block_id: u32, start: u32, end: u32 },

    /// Free a single block.
    FreeBlock(u32),

    /// Free multiple blocks (run-length encoded).
    FreeBlocks(RleVec),

    /// Query how many blocks are still available.
    AvailableBlocks,
}

/// A request from client to server, containing:
/// - A `Uuid` to group commands or track them uniquely.
/// - The `Command` to perform.
#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub instance_id: Uuid,
    pub command: Command,
}

/// Possible "success" data in a response (extend as needed).
#[derive(Debug, Serialize, Deserialize)]
pub enum ResponseData {
    /// For an `AllocateBlocks` or `AllocateBlock` command,
    /// server might return the newly allocated block IDs (RLE).
    AllocatedBlocks(RleVec),

    /// For `AvailableBlocks`, server might return the current count.
    AvailableCount(u32),
}

/// A unified response type. On success, the server returns `Ok` with `data`.
/// On error, it returns `Error` with a code/message.
#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Ok {
        instance_id: Uuid,
        data: Option<ResponseData>,
    },
    Error {
        instance_id: Uuid,
        error_code: u32,
        message: String,
    },
}

#[tokio::main]
async fn main() {
    // 1) Create a Dealer socket
    let mut dealer = DealerSocket::new();
    // 2) Connect to the same endpoint as the server
    dealer.connect("tcp://127.0.0.1:5555").await.unwrap();
    println!("Dealer connected to tcp://127.0.0.1:5555");

    // 3) Build a request: instance_id + command
    let request = Request {
        instance_id: Uuid::new_v4(), // generate a new random UUID
        command: Command::AllocateBlocks(5),
    };

    // 4) Serialize to MessagePack
    let packed_request = to_vec(&request).unwrap();

    // 5) Send as a single multi-frame message to the server
    //    The underlying ZMTP protocol will prepend the Dealer identity automatically.
    let outgoing_msg = ZmqMessage::from(packed_request);
    dealer.send(outgoing_msg).await.unwrap();

    println!(
        "Sent AllocateBlocks(5) with instance_id={}",
        request.instance_id
    );

    // 6) Receive the server's reply: [client_id, packed_response]
    let incoming = dealer.recv().await.unwrap();
    println!("Received {} frames from server.", incoming.len());

    if incoming.len() < 2 {
        eprintln!("Expected at least 2 frames in server response.");
        return;
    }

    let packed_response = incoming.get(1).unwrap();

    // 7) Deserialize into our Response enum
    let response: Response = from_slice(packed_response).unwrap();
    match response {
        Response::Ok { instance_id, data } => {
            println!("Server responded OK with instance_id={}", instance_id);
            if let Some(data) = data {
                match data {
                    ResponseData::AllocatedBlocks(rle_vec) => {
                        println!("Server allocated blocks (RLE) = {:?}", rle_vec.0);
                        // If you need the full list:
                        // let expanded = rle_decode(&rle_vec);
                        // println!("Decoded blocks: {:?}", expanded);
                    }
                    ResponseData::AvailableCount(n) => {
                        println!("Available blocks: {}", n);
                    }
                }
            }
        }
        Response::Error {
            instance_id,
            error_code,
            message,
        } => {
            println!(
                "Server returned ERROR instance_id={}, code={}, msg={}",
                instance_id, error_code, message
            );
        }
    }
}
