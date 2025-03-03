use crate::lm::{ImageEmbedder, TokenEmb};
use crate::object::VspaceId;
use crate::utils::Stream;
use crate::{backend, driver_l4m};

mod l4m {
    pub mod vision {
        include!(concat!(env!("OUT_DIR"), "/l4m.vision.rs"));
    }
}

//
// pub trait ExecuteCommand: backend::ExecuteCommand<l4m::Request, l4m::Response> {}
// impl<T> crate::driver_l4m::ExecuteCommand for T where T: backend::ExecuteCommand<l4m::Request, l4m::Response> {}

pub trait ExecuteCommand:
    driver_l4m::ExecuteCommand + backend::ExecuteCommand<l4m::vision::Request, l4m::vision::Response>
{
}

impl<T> ExecuteCommand for T where
    T: driver_l4m::ExecuteCommand
        + backend::ExecuteCommand<l4m::vision::Request, l4m::vision::Response>
{
}



// 
// impl<B> ImageEmbedder for Driver<B>
// where
//     B: ExecuteCommand,
// {
//     fn embed_img(
//         &mut self,
//         stream: Stream,
//         space: &VspaceId,
//         addrs: Vec<crate::object::Id<TokenEmb>>,
//         url: String,
//     ) -> Result<(), DriverError> {
//         let addrs = self.lookup_all(space, &addrs)?;
// 
//         let cmd = crate::driver_l4m::Command::EmbedImage(crate::driver_l4m::l4m::EmbedImage {
//             embedding_ids: crate::object::Id::map_to_repr(addrs),
//             url,
//         });
// 
//         self.enqueue_cmd(stream, cmd)
//     }
// }
