use rand::Rng;
use rmp_serde::{from_slice, to_vec};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::{self, sync::mpsc};
use uuid::Uuid;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub instance_id: Uuid,
    pub command: Command,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    AllocateBlocks {
        num_blocks: u32,
    },
    AllocateBlock,
    Copy {
        src_block_id: u32,
        dst_block_id: u32,
        src_start: u32,
        dst_start: u32,
        length: u32,
    },
    Drop {
        block_id: u32,
        start: u32,
        end: u32,
    },
    FreeBlock {
        block_id: u32,
    },
    FreeBlocks {
        block_id_offset: u32,
        count: u32,
    },
    AvailableBlocks,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub instance_id: Uuid,
    pub data: ResponseData,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ResponseData {
    AllocatedBlocks { block_id_offset: u32, count: u32 },
    AvailableCount { count: u32 },
    Awk { message: String },
    Error { error_code: u32, message: String },
}

// -----------------------------------------------------------------------------
// The main entry point
// -----------------------------------------------------------------------------
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1) Create a Dealer socket
    let mut socket = DealerSocket::new();
    socket.connect("tcp://127.0.0.1:5555").await?;
    println!("Connected to server at tcp://127.0.0.1:5555");

    // 2) Create an mpsc channel for sending new requests to the socket driver
    let (tx, rx) = mpsc::channel::<Request>(100);

    // 3) Spawn the single I/O driver task that handles all read/write from the socket
    tokio::spawn(socket_driver_task(socket, rx));

    // 4) Spawn a "sender" task that sends random commands every second
    tokio::spawn(sender_task(tx.clone()));

    // Alternatively, you could push *specific* requests:
    // let some_req = Request {
    //     instance_id: Uuid::new_v4(),
    //     command: Command::AllocateBlocks { num_blocks: 5 }
    // };
    // tx.send(some_req).await?;

    // Wait for Ctrl-C
    println!("Press Ctrl-C to quit.");
    tokio::signal::ctrl_c().await?;
    println!("Client shutting down.");

    Ok(())
}

// -----------------------------------------------------------------------------
// The single socket-driver task
// -----------------------------------------------------------------------------
async fn socket_driver_task(mut socket: DealerSocket, mut rx: mpsc::Receiver<Request>) {
    loop {
        tokio::select! {
            // A) Incoming requests from the channel => send to server
            maybe_req = rx.recv() => {
                match maybe_req {
                    Some(req) => {
                        let payload = match to_vec(&req) {
                            Ok(p) => p,
                            Err(e) => {
                                eprintln!("Failed to serialize request: {:?}", e);
                                continue;
                            }
                        };
                        if let Err(e) = socket.send(payload.into()).await {
                            eprintln!("Socket send failed: {:?}", e);
                            // You might choose to break or keep trying
                            break;
                        }
                        println!("*** Sent request: {:?}", req);
                    },
                    None => {
                        // channel closed => no more requests
                        println!("Request channel closed, shutting down driver.");
                        break;
                    }
                }
            },

            // B) Incoming responses from the server
            result = socket.recv() => {
                match result {
                    Ok(msg) => {
                        // Dealer/Router typically has 2 frames: [identity, payload]

                        let payload = msg.get(0).unwrap();
                        match from_slice::<Response>(payload) {
                            Ok(resp) => {
                                println!("---> Received response: {:?}", resp);
                            }
                            Err(err) => {
                                eprintln!("Failed to parse Response from server: {:?}", err);
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("Socket receive error: {:?}", e);
                        // Possibly break or keep going...
                        break;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// A separate "sender" task that pushes random commands
// -----------------------------------------------------------------------------
async fn sender_task(tx: mpsc::Sender<Request>) {
    loop {
        let req = Request {
            instance_id: Uuid::new_v4(),
            command: random_command(),
        };
        if let Err(e) = tx.send(req).await {
            eprintln!("Failed to push request into channel: {:?}", e);
            break;
        }
        // Wait 1 second before sending the next random command
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

// -----------------------------------------------------------------------------
// Helper to generate random commands
// -----------------------------------------------------------------------------
fn random_command() -> Command {
    let mut rng = rand::thread_rng();
    match rng.gen_range(0..7) {
        0 => Command::AllocateBlocks {
            num_blocks: rng.gen_range(1..5),
        },
        1 => Command::AllocateBlock,
        2 => Command::Copy {
            src_block_id: rng.gen_range(1..50),
            dst_block_id: rng.gen_range(1..50),
            src_start: 0,
            dst_start: 0,
            length: 10,
        },
        3 => Command::Drop {
            block_id: rng.gen_range(1..50),
            start: 0,
            end: 10,
        },
        4 => Command::FreeBlock {
            block_id: rng.gen_range(1..50),
        },
        5 => Command::FreeBlocks {
            block_id_offset: rng.gen_range(1..50),
            count: rng.gen_range(1..5),
        },
        6 => Command::AvailableBlocks,
        _ => unreachable!(),
    }
}
