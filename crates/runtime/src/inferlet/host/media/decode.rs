use image::imageops::FilterType;
use models::media::{Fault, Rgb8};

const RESAMPLE: FilterType = FilterType::CatmullRom;

pub fn decode(bytes: &[u8]) -> models::media::Result<Rgb8> {
    if bytes.is_empty() {
        return Err(Fault::Decode(
            "no bytes: an empty payload is no image".into(),
        ));
    }
    let img = image::load_from_memory(bytes).map_err(|e| {
        Fault::Decode(format!(
            "the bytes are not an image this front-end reads: {e}"
        ))
    })?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Rgb8::new(h, w, rgb.into_raw())
}

#[must_use]
pub fn resize_exact(src: &Rgb8, th: u32, tw: u32) -> Rgb8 {
    if src.h == th && src.w == tw {
        return src.clone();
    }
    let buf = image::RgbImage::from_raw(src.w, src.h, src.data.clone())
        .expect("an Rgb8 always holds h · w · 3 bytes");
    let out = image::imageops::resize(&buf, tw, th, RESAMPLE);
    Rgb8 {
        h: th,
        w: tw,
        data: out.into_raw(),
    }
}
