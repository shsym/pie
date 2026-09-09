use engine::fire::{Mask, Masking, RsReset, RsVerb};
use model_compiler::CompiledModel;

use crate::error::{Fault, Result};
use crate::record;

use super::{Lane, MROPE_COORDS, Media, PATCH_ROUTE_DROP, Seated, Shell};

struct Ballot {
    slip: Option<crate::device::Buffer>,
}

impl Ballot {
    fn open(device: &crate::device::Context) -> Ballot {
        let grouped = device.ctx().comm("arming ballot").is_ok();
        Ballot {
            slip: grouped
                .then(|| crate::device::Buffer::zeroed(4).ok())
                .flatten(),
        }
    }

    fn any(&mut self, device: &crate::device::Context, mine: bool) -> bool {
        let Some(slip) = self.slip.as_mut() else {
            return mine;
        };
        let vote: f32 = if mine { 1.0 } else { 0.0 };
        let sum = (|| -> Result<f32> {
            slip.write(0, &vote.to_le_bytes())?;
            let mut wire = kernels_cuda::Tensor::new(slip.ptr(), 1, 1, model_ir::Dtype::F32);
            kernels_cuda::collective::all_reduce(device.ctx(), &mut wire)
                .map_err(crate::error::kernel)?;
            device.synchronize()?;
            let mut back = [0u8; 4];
            slip.read(0, &mut back)?;
            Ok(f32::from_le_bytes(back))
        })();
        match sum {
            Ok(total) => total >= 0.5,
            Err(_) => {
                self.slip = None;
                mine
            }
        }
    }
}

#[derive(Clone)]
struct Synthetic {
    word: u64,
    tokens: Vec<u32>,
    mask: Option<Masking>,
    adapter: Option<u32>,
    drafts: bool,
    captures: bool,
    slot: u32,
    pages: Vec<u32>,
    held: Option<u32>,
    media: Option<SyntheticMedia>,
    stream: u8,
}

#[derive(Clone)]
struct SyntheticMedia {
    rows: Vec<u32>,
    patches: Vec<u8>,
    routes: Vec<i32>,
    positions: Vec<i32>,
    embed_rows: Vec<i32>,
    embed_weights: Vec<f32>,
}

#[derive(Debug, Clone)]
enum BodySynth {
    Decode { lanes: u32, class: usize },
    Prefill { class: usize, rows: Vec<u32> },
    Mixed {
        decode: usize,
        class: usize,
        rows: Vec<u32>,
    },
    Fragmented { lanes: Vec<(usize, u32)> },
    Tower {
        class: usize,
        rows: u32,
        images: u32,
        patches: u32,
    },
    Ensemble { lanes: Vec<(usize, u32)> },
    Joint { lanes: Vec<(usize, u32)> },
}

impl BodySynth {
    fn present(&self) -> Vec<usize> {
        let mut classes = match self {
            BodySynth::Decode { class, .. } | BodySynth::Prefill { class, .. } => vec![*class],
            BodySynth::Mixed { decode, class, .. } => vec![*decode, *class],
            BodySynth::Fragmented { lanes }
            | BodySynth::Ensemble { lanes }
            | BodySynth::Joint { lanes } => lanes.iter().map(|(class, _)| *class).collect(),
            BodySynth::Tower { class, .. } => vec![*class],
        };
        classes.sort_unstable();
        classes.dedup();
        classes
    }

    fn kind(&self) -> Kind {
        match self {
            BodySynth::Decode { .. } => Kind::Decode,
            BodySynth::Prefill { .. } => Kind::Prefill,
            BodySynth::Mixed { .. } => Kind::Mixed,
            BodySynth::Fragmented { .. } => Kind::Fragmented,
            BodySynth::Tower { .. } => Kind::Tower,
            BodySynth::Ensemble { .. } => Kind::Ensemble,
            BodySynth::Joint { .. } => Kind::Joint,
        }
    }

    fn skips_on_present_set(&self) -> bool {
        !matches!(self, BodySynth::Tower { .. })
    }

    fn lanes(&self) -> (Vec<(usize, u32)>, Vec<(u32, u32)>) {
        match self {
            BodySynth::Decode { lanes, class } => {
                (vec![(*class, 1u32); *lanes as usize], Vec::new())
            }
            BodySynth::Prefill { class, rows } => (
                rows.iter().map(|rows| (*class, *rows)).collect(),
                Vec::new(),
            ),
            BodySynth::Mixed {
                decode,
                class,
                rows,
            } => (
                core::iter::once((*decode, 1u32))
                    .chain(rows.iter().map(|rows| (*class, *rows)))
                    .collect(),
                Vec::new(),
            ),
            BodySynth::Fragmented { lanes } | BodySynth::Joint { lanes } => {
                (lanes.clone(), Vec::new())
            }
            BodySynth::Tower {
                class,
                rows,
                images,
                patches,
            } => (vec![(*class, *rows)], vec![(*images, *patches)]),
            BodySynth::Ensemble { lanes } => (
                lanes
                    .iter()
                    .flat_map(|(class, count)| vec![(*class, 1u32); *count as usize])
                    .collect(),
                Vec::new(),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Decode,
    Prefill,
    Mixed,
    Fragmented,
    Tower,
    Ensemble,
    Joint,
}

impl Kind {
    pub const COUNT: usize = 7;

    pub const ALL: [Kind; Kind::COUNT] = [
        Kind::Decode,
        Kind::Prefill,
        Kind::Mixed,
        Kind::Fragmented,
        Kind::Tower,
        Kind::Ensemble,
        Kind::Joint,
    ];

    pub fn at(self) -> usize {
        self as usize
    }
}

impl core::fmt::Display for Kind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Kind::Decode => "decode",
            Kind::Prefill => "prefill",
            Kind::Mixed => "mixed",
            Kind::Fragmented => "fragmented",
            Kind::Tower => "tower",
            Kind::Ensemble => "ensemble",
            Kind::Joint => "joint",
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LatticePoint {
    pub bucket: u32,
    pub lanes: u32,
}

struct Deployment {
    points: Vec<LatticePoint>,
    buckets: Vec<u32>,
    patch_points: Vec<u32>,
    decoders: Vec<usize>,
    prefilling: Vec<usize>,
    media: Vec<usize>,
    fragmenting: Vec<Vec<usize>>,
    joining: Vec<Vec<usize>>,
    decoding: model_ir::ClassSet,
    seats: u32,
    context: u32,
    max_lanes: u32,
    patch_fold: u32,
}

#[derive(Default)]
struct Targets {
    targets: Vec<(u32, BodySynth)>,
    unfireable: Vec<String>,
}

fn unfireable_line(deployment: &Deployment, what: &str, at: &str, note: Option<&str>) -> String {
    let (note, comma) = match note {
        Some(note) => (note, ", "),
        None => ("", ""),
    };
    format!(
        "{what} at {at} ({note}{comma}{} seat(s) x {} context, {} lane(s))",
        deployment.seats, deployment.context, deployment.max_lanes,
    )
}

fn decode_keys(deployment: &Deployment, into: &mut Targets) {
    for LatticePoint { bucket, lanes } in deployment.points.iter().copied() {
        for class in deployment.decoders.iter().copied() {
            into.targets
                .push((bucket, BodySynth::Decode { lanes, class }));
        }
    }
}

fn prefill_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        for class in deployment.prefilling.iter().copied() {
            match Shell::spread(
                point,
                deployment.seats.min(deployment.max_lanes),
                deployment.context,
            ) {
                Some(rows) => into
                    .targets
                    .push((point, BodySynth::Prefill { class, rows })),
                None => into.unfireable.push(unfireable_line(
                    deployment,
                    &format!("prefill c{class}"),
                    &format!("bucket {point}"),
                    None,
                )),
            }
        }
    }
}

fn mixed_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        for decode in deployment.decoders.iter().copied() {
            for class in deployment.prefilling.iter().copied() {
                let rows = (point >= 2 && deployment.seats >= 2)
                    .then(|| {
                        Shell::spread(
                            point - 1,
                            (deployment.seats - 1).min(deployment.max_lanes.saturating_sub(1)),
                            deployment.context,
                        )
                    })
                    .flatten();
                match rows {
                    Some(rows) => into.targets.push((
                        point,
                        BodySynth::Mixed {
                            decode,
                            class,
                            rows,
                        },
                    )),
                    None => into.unfireable.push(unfireable_line(
                        deployment,
                        &format!("mixed c{decode}+c{class}"),
                        &format!("bucket {point}"),
                        None,
                    )),
                }
            }
        }
    }
}

fn fragmented_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        for present in &deployment.fragmenting {
            match fragment_rows(deployment, present, point) {
                Some(lanes) => {
                    into.targets.push((point, BodySynth::Fragmented { lanes }));
                }
                None => into.unfireable.push(unfireable_line(
                    deployment,
                    &format!("fragmented {present:?}"),
                    &format!("bucket {point}"),
                    None,
                )),
            }
        }
    }
}

fn tower_keys(deployment: &Deployment, into: &mut Targets) {
    for patches in deployment.patch_points.iter().copied() {
        for class in deployment.media.iter().copied() {
            for point in deployment.buckets.iter().copied() {
                let fold = deployment.patch_fold.max(1);
                let owed = patches.div_ceil(fold).max(1);
                if owed > point || point > deployment.context || deployment.seats == 0 {
                    into.unfireable.push(unfireable_line(
                        deployment,
                        &format!("tower c{class}"),
                        &format!("bucket {point} + patch rung {patches}"),
                        Some(&format!("{owed} placeholder row(s) owed")),
                    ));
                    continue;
                }
                into.targets.push((
                    point,
                    BodySynth::Tower {
                        class,
                        rows: point,
                        images: 1,
                        patches,
                    },
                ));
            }
        }
    }
}

fn ensemble_keys(deployment: &Deployment, into: &mut Targets) {
    let words = deployment.decoders.len() as u32;
    if words < 2 {
        return;
    }
    for LatticePoint { bucket, lanes } in deployment.points.iter().copied() {
        match ensemble_lanes(deployment, lanes) {
            Some(lanes) => {
                into.targets.push((bucket, BodySynth::Ensemble { lanes }));
            }
            None => into.unfireable.push(unfireable_line(
                deployment,
                &format!("ensemble {:?}", deployment.decoders),
                &format!("bucket {bucket}"),
                Some(&format!("{words} decode word(s) in {lanes} lane(s)")),
            )),
        }
    }
}

fn ensemble_lanes(deployment: &Deployment, lanes: u32) -> Option<Vec<(usize, u32)>> {
    let words = deployment.decoders.len() as u32;
    if words < 2 || lanes < words {
        return None;
    }
    let base = lanes / words;
    let over = lanes % words;
    Some(
        deployment
            .decoders
            .iter()
            .copied()
            .enumerate()
            .map(|(at, class)| (class, base + u32::from((at as u32) < over)))
            .collect(),
    )
}

fn joint_keys(deployment: &Deployment, into: &mut Targets) {
    for point in deployment.buckets.iter().copied() {
        for present in &deployment.joining {
            match joint_rows(deployment, present, point) {
                Some(lanes) => into.targets.push((point, BodySynth::Joint { lanes })),
                None => into.unfireable.push(unfireable_line(
                    deployment,
                    &format!("joint {present:?}"),
                    &format!("bucket {point}"),
                    None,
                )),
            }
        }
    }
}

const ARMS: [fn(&Deployment, &mut Targets); Kind::COUNT] = [
    decode_keys,
    prefill_keys,
    mixed_keys,
    fragmented_keys,
    tower_keys,
    ensemble_keys,
    joint_keys,
];

impl core::fmt::Display for BodySynth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BodySynth::Decode { lanes, class } => write!(f, "decode c{class} x{lanes}"),
            BodySynth::Prefill { class, rows } => {
                write!(f, "prefill c{class} {rows:?}")
            }
            BodySynth::Mixed {
                decode,
                class,
                rows,
            } => {
                write!(f, "mixed c{decode}+c{class} {rows:?}")
            }
            BodySynth::Tower {
                class,
                rows,
                images,
                patches,
            } => write!(f, "tower c{class} {rows}r x{images} img {patches}p"),
            BodySynth::Fragmented { lanes } => {
                write!(f, "fragmented ")?;
                for (at, (class, rows)) in lanes.iter().enumerate() {
                    if at > 0 {
                        f.write_str("+")?;
                    }
                    write!(f, "c{class}:{rows}")?;
                }
                Ok(())
            }
            BodySynth::Ensemble { lanes } => {
                write!(f, "ensemble ")?;
                for (at, (class, count)) in lanes.iter().enumerate() {
                    if at > 0 {
                        f.write_str("+")?;
                    }
                    write!(f, "c{class} x{count}")?;
                }
                Ok(())
            }
            BodySynth::Joint { lanes } => {
                write!(f, "joint ")?;
                for (at, (class, rows)) in lanes.iter().enumerate() {
                    if at > 0 {
                        f.write_str("+")?;
                    }
                    write!(f, "c{class}:{rows}")?;
                }
                Ok(())
            }
        }
    }
}

impl Shell {
    fn synthetic_lanes(&self, lanes: &[(usize, u32)]) -> Vec<Synthetic> {
        self.synthetic_lanes_with(lanes, &[])
    }

    fn synthetic_lanes_with(&self, lanes: &[(usize, u32)], media: &[(u32, u32)]) -> Vec<Synthetic> {
        let slots = self.held.len().max(1) as u32;
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let mut next_page = 0u64;
        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes) as usize;
        let taps = self.patch_seat.map_or(0, |seat| seat.embed_taps) as usize;
        let weight_taps = self.patch_seat.map_or(0, |seat| {
            if seat.embed_weights {
                seat.embed_taps
            } else {
                0
            }
        }) as usize;
        let fold = (self.patch_fold as usize).max(1);
        lanes
            .iter()
            .enumerate()
            .map(|(at, &(class, rows))| {
                let wants_media = media.get(at).is_some_and(|(images, _)| *images > 0);
                let request = self.representative(class, rows, wants_media);
                Synthetic {
                    stream: request.stream().code(),
                    word: (self.classify)(&request),
                    tokens: vec![0u32; rows as usize],
                    mask: request
                        .has_custom_mask()
                        .then(|| Masking::Extent(Mask::new(vec![0, rows], u64::from(rows)))),
                    adapter: request.has_adapter().then_some(0),
                    drafts: request.drafts(),
                    captures: request.captures_scores(),
                    slot: (at as u32) % slots,
                    pages: {
                        let pages = u64::from(rows).div_ceil(page_size).max(1);
                        let table: Vec<u32> = (next_page..next_page + pages)
                            .map(|page| u32::try_from(page).unwrap_or(u32::MAX))
                            .collect();
                        next_page += pages;
                        table
                    },
                    held: Some(0),
                    media: media
                        .get(at)
                        .copied()
                        .filter(|(images, patches)| *images > 0 && *patches > 0)
                        .map(|(images, patches)| {
                            let patches = patches as usize;
                            let per = patches / images as usize;
                            let mut per_image: Vec<u32> = vec![per as u32; images as usize];
                            per_image[0] += (patches - per * images as usize) as u32;
                            let live = patches / fold;
                            let mut routes = vec![
                                if self.drops_patch_rows {
                                    PATCH_ROUTE_DROP
                                } else {
                                    0
                                };
                                patches
                            ];
                            for (j, route) in routes.iter_mut().take(live).enumerate() {
                                *route = (j % rows.max(1) as usize) as i32;
                            }
                            let mut embed_weights = vec![0f32; patches * weight_taps];
                            for row in embed_weights.chunks_mut(weight_taps.max(1)) {
                                if let Some(first) = row.first_mut() {
                                    *first = 1.0;
                                }
                            }
                            SyntheticMedia {
                                rows: per_image,
                                patches: vec![0u8; patches * row_bytes],
                                routes,
                                positions: vec![0i32; patches * MROPE_COORDS],
                                embed_rows: vec![0i32; patches * taps],
                                embed_weights,
                            }
                        }),
                }
            })
            .collect()
    }

    fn representative(&self, class: usize, rows: u32, wants_media: bool) -> model_ir::Request {
        let landing = &self.landing[class];
        landing
            .iter()
            .find(|request| {
                request.has_media() == wants_media && (request.query_len() == 1) == (rows == 1)
            })
            .or_else(|| {
                landing
                    .iter()
                    .find(|request| request.has_media() == wants_media)
            })
            .or_else(|| landing.first())
            .copied()
            .unwrap_or_else(|| {
                panic!("the arming pass enumerated class {class}, which no request lands in")
            })
    }

    fn fire_synthetic(&mut self, owned: &[Synthetic]) -> Result<()> {
        self.fire_synthetic_as(owned, crate::serve::Golden::Off)
            .map(|_| ())
    }

    fn fire_synthetic_as(
        &mut self,
        owned: &[Synthetic],
        arm: crate::serve::Golden,
    ) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = owned
            .iter()
            .map(|lane| Seated {
                lane: Lane {
                    slot: lane.slot,
                    word: lane.word,
                    tokens: &lane.tokens,
                },
                pages: &lane.pages,
                held: lane.held,
                kv_less: false,
                translation: &[],
                mask: lane.mask.as_ref(),
                adapter: lane.adapter,
                drafts: lane.drafts,
                captures_scores: lane.captures,
                bidirectional: false,
                self_cond: None,
                readout: None,
                rs: RsVerb::Fold,
                rs_reset: RsReset::Inferred,
                stream: lane.stream,
                group: None,
                peer: None,
                ports: &[],
            })
            .collect();

        let media: Vec<Media<'_>> = owned
            .iter()
            .enumerate()
            .filter_map(|(at, lane)| {
                lane.media.as_ref().map(|shot| Media {
                    lane: at as u32,
                    rows: &shot.rows,
                    patches: &shot.patches,
                    routes: &shot.routes,
                    positions: &shot.positions,
                    token_positions: &[],
                    embed_rows: &shot.embed_rows,
                    embed_weights: &shot.embed_weights,
                })
            })
            .collect();

        self.arming = true;
        self.golden_arm = arm;
        let armed = self.fire_media(&seated, &[], &media, &mut Vec::new());
        self.golden_arm = crate::serve::Golden::Off;
        self.arming = false;
        armed
    }

    fn golden(&mut self, key: &record::BodyKey, owned: &[Synthetic]) -> Result<()> {
        use crate::serve::Golden;

        let refused = |why: String| Fault::Golden {
            key: key.to_string(),
            why: format!(
                "lanes=[{}] {why}",
                owned
                    .iter()
                    .map(|lane| format!("{:#x}/{}r{}", lane.word, lane.tokens.len(), lane.slot))
                    .collect::<Vec<String>>()
                    .join(" "),
            ),
        };
        let slots: Vec<u32> = owned.iter().map(|lane| lane.slot).collect();
        let fire = |shell: &mut Shell, arm: Golden| -> Result<Vec<Vec<f32>>> {
            for &slot in &slots {
                shell.open(slot)?;
            }
            let out = shell.fire_synthetic_as(owned, arm)?;
            shell.device.synchronize()?;
            Ok(out)
        };
        if crate::record::ptr_traced(key) {
            crate::record::PTR_TAG.store(2, std::sync::atomic::Ordering::Relaxed);
        }
        let walked = fire(self, Golden::Eager);
        crate::record::PTR_TAG.store(0, std::sync::atomic::Ordering::Relaxed);
        let walked =
            walked.map_err(|fault| refused(format!("the control arm would not fire: {fault}")))?;
        let replayed = fire(self, Golden::Body)
            .map_err(|fault| refused(format!("the body arm would not fire: {fault}")))?;
        match evidence(&walked, &replayed) {
            None => Ok(()),
            Some(why) => {
                if super::diag::on().golden_probe {
                    let walked_again = fire(self, Golden::Eager);
                    let replayed_again = fire(self, Golden::Body);
                    let same = |a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>| {
                        evidence(a, b).map_or("identical".to_string(), |why| {
                            why.split("  lanes=").next().unwrap_or(&why).to_string()
                        })
                    };
                    if let Ok(w2) = &walked_again {
                        eprintln!("[golden-probe] {key} walk vs walk: {}", same(&walked, w2));
                    }
                    if let Ok(b2) = &replayed_again {
                        eprintln!("[golden-probe] {key} body vs body: {}", same(&replayed, b2));
                    }
                    let script = self.cache.body_script(key);
                    let execs = script.iter().filter(|(island, ..)| !island).count();
                    let mut culprit = None;
                    for k in 1..=execs {
                        crate::record::REPLAY_UPTO.store(k, std::sync::atomic::Ordering::Relaxed);
                        let partial = fire(self, Golden::Body);
                        crate::record::REPLAY_UPTO
                            .store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
                        match partial {
                            Ok(out) => {
                                let agrees = evidence(&walked, &out).is_none();
                                eprintln!(
                                    "[golden-probe] {key} replay first {k} exec(s): {}",
                                    if agrees {
                                        "agrees with the walk"
                                    } else {
                                        "DIFFERS"
                                    }
                                );
                                if !agrees {
                                    culprit = Some(k);
                                    break;
                                }
                            }
                            Err(fault) => {
                                eprintln!(
                                    "[golden-probe] {key} replay first {k} exec(s): would not fire: {fault}"
                                );
                                break;
                            }
                        }
                    }
                    if let Some(k) = culprit {
                        let mut seen = 0usize;
                        for (at_step, (island, from, upto)) in script.iter().enumerate() {
                            if *island {
                                continue;
                            }
                            seen += 1;
                            if seen == k {
                                eprintln!(
                                    "[golden-probe] {key} first disagreeing exec is step {at_step}: regions {from}..{upto}"
                                );
                                for region in *from..*upto {
                                    if let Some(template) =
                                        self.compiled.template().get(region as usize)
                                    {
                                        eprintln!(
                                            "[golden-probe]   region {region}: nodes {:?} phase {:?} stream {} lowering {:?}",
                                            template.nodes,
                                            template.phase,
                                            template.stream,
                                            template.lowering
                                        );
                                        for node in template.nodes.clone() {
                                            if let Some(held) = self.trace.nodes.get(node as usize)
                                            {
                                                let op = format!("{:?}", held.op);
                                                let head: String = op.chars().take(90).collect();
                                                eprintln!(
                                                    "[golden-probe]     node {node} layer {:?}: {head}",
                                                    held.layer
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let mut nth = 0usize;
                    for (at_step, (island, from, upto)) in script.iter().enumerate() {
                        if *island {
                            continue;
                        }
                        let j = nth;
                        nth += 1;
                        crate::record::REPLAY_FROM.store(j, std::sync::atomic::Ordering::Relaxed);
                        crate::record::REPLAY_UPTO
                            .store(j + 1, std::sync::atomic::Ordering::Relaxed);
                        let alone = fire(self, Golden::Body);
                        crate::record::REPLAY_FROM.store(0, std::sync::atomic::Ordering::Relaxed);
                        crate::record::REPLAY_UPTO
                            .store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
                        let verdict = match &alone {
                            Ok(out) => match evidence(&walked, out) {
                                None => "agrees".to_string(),
                                Some(why) => format!(
                                    "DIFFERS ({})",
                                    why.chars().take(120).collect::<String>()
                                ),
                            },
                            Err(fault) => format!("would not fire: {fault}"),
                        };
                        let nodes = self
                            .compiled
                            .template()
                            .get(*from as usize)
                            .map(|t| t.nodes.start)
                            .unwrap_or(0);
                        let nodes_end = self
                            .compiled
                            .template()
                            .get((*upto as usize).saturating_sub(1))
                            .map(|t| t.nodes.end)
                            .unwrap_or(0);
                        let layers: Vec<Option<u32>> = {
                            let mut seen = Vec::new();
                            for n in nodes..nodes_end {
                                if let Some(h) = self.trace.nodes.get(n as usize) {
                                    if seen.last() != Some(&h.layer) {
                                        seen.push(h.layer);
                                    }
                                }
                            }
                            seen
                        };
                        eprintln!(
                            "[golden-probe] {key} exec {j} alone (step {at_step}, regions {from}..{upto}, nodes {nodes}..{nodes_end}, layers {layers:?}): {verdict}"
                        );
                    }
                    for (i, body_lane) in replayed.iter().enumerate() {
                        let best = walked
                            .iter()
                            .enumerate()
                            .map(|(j, walk_lane)| {
                                let agree = body_lane
                                    .iter()
                                    .zip(walk_lane)
                                    .filter(|(x, y)| x.to_bits() == y.to_bits())
                                    .count();
                                (agree, j)
                            })
                            .max()
                            .unwrap_or((0, 0));
                        eprintln!(
                            "[golden-probe] {key} body lane {i} best matches walk lane {} ({} of {} cells bit-exact); head body={:?} walk[{}]={:?} walk[{i}]={:?}",
                            best.1,
                            best.0,
                            body_lane.len(),
                            &body_lane[..body_lane.len().min(4)],
                            best.1,
                            &walked[best.1][..walked[best.1].len().min(4)],
                            &walked[i][..walked[i].len().min(4)],
                        );
                        if i >= 7 {
                            break;
                        }
                    }
                }
                Err(refused(why))
            }
        }
    }

    fn golden_real(&mut self, key: &record::BodyKey, owned: &[Synthetic]) -> Result<()> {
        use crate::serve::Golden;

        let context = self.pools.paging().context();
        if owned.is_empty()
            || owned.len() > self.held.len()
            || owned
                .iter()
                .any(|lane| lane.tokens.is_empty() || 2 * lane.tokens.len() > context as usize)
        {
            return Ok(());
        }
        let row = u32::from(self.adapters().slots().seats() >= 2);
        let real: Vec<Synthetic> = owned
            .iter()
            .map(|lane| {
                let rows = lane.tokens.len() as u32;
                let have = rows.max(1);
                let extent = have + rows;
                Synthetic {
                    tokens: (0..rows).map(|at| 1 + at % 16).collect(),
                    mask: lane.mask.as_ref().map(|_| {
                        Masking::Rows(
                            (0..rows)
                                .map(|at| {
                                    let allowed = have + at + 1;
                                    let mut runs = vec![0, allowed];
                                    if extent > allowed {
                                        runs.push(extent - allowed);
                                    }
                                    Mask::new(runs, u64::from(extent))
                                })
                                .collect(),
                        )
                    }),
                    adapter: lane.adapter.map(|_| row),
                    held: Some(have),
                    ..lane.clone()
                }
            })
            .collect();
        let refused = |why: String| Fault::Golden {
            key: key.to_string(),
            why: format!(
                "realistic pair: lanes=[{}] {why}",
                real.iter()
                    .map(|lane| format!("{:#x}/{}r{}", lane.word, lane.tokens.len(), lane.slot))
                    .collect::<Vec<String>>()
                    .join(" "),
            ),
        };
        let slots: Vec<u32> = owned.iter().map(|lane| lane.slot).collect();
        let fire = |shell: &mut Shell, arm: Golden| -> Result<Vec<Vec<f32>>> {
            for &slot in &slots {
                shell.open(slot)?;
            }
            shell.fire_synthetic_as(owned, Golden::Eager)?;
            let out = shell.fire_synthetic_as(&real, arm)?;
            shell.device.synchronize()?;
            Ok(out)
        };
        let Ok(walked) = fire(self, Golden::Eager) else {
            return Ok(());
        };
        let replayed = fire(self, Golden::Body)
            .map_err(|fault| refused(format!("the body arm would not fire: {fault}")))?;
        match evidence(&walked, &replayed) {
            None => Ok(()),
            Some(why) => Err(refused(why)),
        }
    }

    pub(super) fn arm_bodies(&mut self) -> Result<()> {
        if !self.records_bodies() || !Self::keyable_units(&self.compiled) {
            return Ok(());
        }

        let ceiling = self.lane_ceiling();
        if ceiling == 0 {
            return Ok(());
        }
        if super::diag::on().arm_trace {
            for (class, requests) in self.landing.iter().enumerate() {
                eprintln!("[arm-trace] class c{class}: {requests:?}");
            }
        }
        let points: Vec<LatticePoint> = if self.budget.buckets.is_empty() {
            (1..=ceiling)
                .map(|n| LatticePoint {
                    bucket: n,
                    lanes: n,
                })
                .collect()
        } else {
            let mut points = Vec::new();
            for point in self.budget.buckets.iter().copied() {
                if point <= ceiling {
                    points.push(LatticePoint {
                        bucket: point,
                        lanes: point,
                    });
                } else {
                    if ceiling > points.last().map_or(0, |point| point.bucket) {
                        points.push(LatticePoint {
                            bucket: point,
                            lanes: ceiling,
                        });
                    }
                    break;
                }
            }
            points
        };
        let seats = self.held.len() as u32;
        let context = if self.spaces == 0 {
            self.budget.max_tokens
        } else {
            self.pools.paging().context()
        };
        let max_lanes = self.budget.max_lanes;
        let classes = self.compiled.classes.classes.len();
        let textual = |class: usize| !self.media.contains(class);
        let staged = self
            .towered
            .then_some(self.patch_seat)
            .flatten()
            .is_some_and(|seat| seat.row_bytes > 0);
        let patch_points: Vec<u32> = match (staged, self.budgets.patches.as_ref()) {
            (true, Some(ladder)) => ladder.buckets.clone(),
            _ => Vec::new(),
        };
        let deployment = Deployment {
            points,
            buckets: self.budget.buckets.clone(),
            patch_points,
            decoders: self
                .decoding
                .iter()
                .filter(|class| textual(*class))
                .collect(),
            prefilling: (0..classes)
                .filter(|class| {
                    !self.decoding.contains(*class)
                        && textual(*class)
                        && !self.landing[*class].is_empty()
                })
                .collect(),
            media: self.media.iter().collect(),
            fragmenting: self.fragmenting(),
            joining: self.joining(),
            decoding: self.decoding.clone(),
            seats,
            context,
            max_lanes,
            patch_fold: self.patch_fold,
        };
        let mut found = Targets::default();
        for enumerate in ARMS {
            enumerate(&deployment, &mut found);
        }
        let Targets {
            mut targets,
            mut unfireable,
        } = found;
        let top = targets.iter().map(|(bucket, _)| *bucket).max();
        let rank = |kind: Kind| match kind {
            Kind::Decode => 0u8,
            Kind::Ensemble => 1,
            Kind::Mixed => 2,
            Kind::Prefill => 3,
            Kind::Joint => 4,
            Kind::Fragmented => 5,
            Kind::Tower => 6,
        };
        targets
            .sort_by_key(|(bucket, target)| (Some(*bucket) != top, rank(target.kind()), *bucket));
        if targets.is_empty() {
            return Ok(());
        }

        let mut armed = 0usize;
        let mut wanted = 0usize;
        let mut tally = [(0usize, 0usize); Kind::COUNT];
        let mut refused: Option<String> = None;
        let mut unadmitted: Vec<Vec<usize>> = Vec::new();

        let mut never = 0usize;
        let mut never_from = 0u32;
        let mut belted = false;
        let mut starved: Option<u64> = None;
        let mut peer_stopped = false;
        let unit_margin = self.pools.map_unit_bytes().saturating_mul(16);
        let mut ballot = Ballot::open(&self.device);
        for (bucket, target) in targets {
            let spent = self.cache.body_stats();
            let spare = self.pools.spare_bytes().unwrap_or(u64::MAX);
            let price = (2 * spent.census.bytes / spent.census.bodies.max(1)) as u64;
            let margin = price.max(unit_margin);
            let headroom = spare < margin;
            let mine = spent.census.bodies >= record::MAX_BODIES
                || spent.census.bytes >= self.bodies_mem
                || headroom;
            if ballot.any(&self.device, mine) {
                if never == 0 {
                    never_from = bucket;
                    belted = spent.census.bodies >= record::MAX_BODIES;
                    if headroom && !belted && spent.census.bytes < self.bodies_mem {
                        starved = Some(spare);
                    }
                    peer_stopped = !mine;
                }
                never += 1;
                continue;
            }
            let present = target.present();
            if target.skips_on_present_set() && unadmitted.contains(&present) {
                unfireable.push(format!(
                    "bucket {bucket}, {target}: inadmissible present set"
                ));
                continue;
            }
            wanted += 1;
            let at = target.kind().at();
            let (lanes, media) = target.lanes();
            tally[at].1 += 1;
            let owned = if media.is_empty() {
                self.synthetic_lanes(&lanes)
            } else {
                self.synthetic_lanes_with(&lanes, &media)
            };
            self.armed_body = None;
            let mut faulted = false;
            for _ in 0..record::WARM_FIRES {
                let fired = self.fire_synthetic(&owned);
                let landed = self.device.synchronize();
                if let Err(why) = fired.and(landed) {
                    if super::diag::on().arm_trace {
                        eprintln!("[arm-trace] refused bucket {bucket}, {target}: {why}");
                    }
                    refused = Some(format!("bucket {bucket}, {target}: {why}"));
                    faulted = true;
                    break;
                }
            }
            let admitted = self.armed_body.is_some();
            let key = self.armed_body.take();
            if self.golden
                && let Some(key) = key.as_ref()
                && self.cache.holds_body(key)
            {
                let verdict = self
                    .golden(key, &owned)
                    .and_then(|()| self.golden_real(key, &owned));
                if let Err(fault) = verdict {
                    if super::diag::on().golden_skip {
                        eprintln!("[arm-trace] golden refused {key}: {fault}");
                        self.cache.body_drop(key);
                    } else {
                        return Err(fault);
                    }
                }
            }
            if key.as_ref().is_some_and(|key| self.cache.body_armed(key)) {
                armed += 1;
                tally[at].0 += 1;
                if super::diag::on().arm_trace
                    && let Some(key) = key.as_ref()
                {
                    eprintln!("[arm-trace] armed {key}");
                }
            } else if !admitted && !faulted && target.skips_on_present_set() {
                unadmitted.push(present);
            }
        }

        let stats = self.cache.body_stats();
        let seal = if armed == 0 {
            Seal::Open
        } else if never == 0 && refused.is_none() {
            Seal::Complete
        } else {
            Seal::Partial { never }
        };
        let pool_line = format!(
            "pool declared {} MiB, high water {} MiB, committed {} MiB, spare under the ceiling {} MiB",
            self.pools.declared_bytes() >> 20,
            self.pools.high_water_bytes() >> 20,
            self.pools.committed_bytes() >> 20,
            self.pools.spare_bytes().map_or(0, |bytes| bytes >> 20),
        );
        let report = Armed {
            pool_line,
            wanted,
            armed,
            kinds: tally,
            segmented: stats.census.segmented,
            bytes: stats.census.bytes,
            unweighed: stats.census.unweighed,
            bodies_mem: self.bodies_mem,
            bodies: stats.census.bodies,
            last_refusal: refused,
            unfireable,
            never,
            never_from,
            belted,
            starved,
            peer_stopped,
            declines: stats.tally.declines,
            refusals: stats.tally.refusals,
            seal,
        };
        eprintln!("engine-cuda: {report}");
        if !matches!(seal, Seal::Open) {
            self.cache.seal_bodies();
        }
        self.armed = Some(report);
        Ok(())
    }

    fn fragmenting(&self) -> Vec<Vec<usize>> {
        let classes = self.compiled.classes.classes.len();
        let mut seen: Vec<&model_ir::ClassSet> = Vec::new();
        let mut found: Vec<Vec<usize>> = Vec::new();
        for region in self.compiled.template() {
            if region.mask.len() < 2 || seen.contains(&&region.mask) {
                continue;
            }
            seen.push(&region.mask);
            for separator in 0..classes {
                if region.mask.contains(separator) {
                    continue;
                }
                let Some(present) = Self::witness(&self.compiled, &region.mask, separator) else {
                    continue;
                };
                if present.iter().any(|class| self.media.contains(*class)) {
                    continue;
                }
                if Self::breaks(&self.compiled, &region.mask, &present) && !found.contains(&present)
                {
                    found.push(present);
                }
            }
        }
        found
    }

    fn joining(&self) -> Vec<Vec<usize>> {
        let mut found: Vec<Vec<usize>> = Vec::new();
        for region in self.compiled.template() {
            if region.mask.len() < 2 {
                continue;
            }
            let present: Vec<usize> = region.mask.iter().collect();
            if present
                .iter()
                .any(|class| self.decoding.contains(*class) || self.media.contains(*class))
                || found.contains(&present)
            {
                continue;
            }
            found.push(present);
        }
        found
    }

    fn witness(
        compiled: &CompiledModel,
        mask: &model_ir::ClassSet,
        separator: usize,
    ) -> Option<Vec<usize>> {
        let mut whole: Vec<usize> = mask.iter().collect();
        whole.push(separator);
        let order = compiled
            .order
            .class_order(&model_ir::ClassSet::of(whole.iter().copied()));
        let mut before: Option<usize> = None;
        let mut after: Option<usize> = None;
        let mut passed = false;
        for class in order {
            let class = class as usize;
            if class == separator {
                passed = true;
                continue;
            }
            if !mask.contains(class) {
                continue;
            }
            if passed {
                after = Some(class);
                break;
            }
            before = Some(class);
        }
        let mut present = vec![before?, separator, after?];
        present.sort_unstable();
        Some(present)
    }

    fn breaks(compiled: &CompiledModel, mask: &model_ir::ClassSet, present: &[usize]) -> bool {
        let order = compiled
            .order
            .class_order(&model_ir::ClassSet::of(present.iter().copied()));
        let mut runs = 0usize;
        let mut inside = false;
        for class in order {
            if mask.contains(class as usize) {
                runs += usize::from(!inside);
                inside = true;
            } else {
                inside = false;
            }
        }
        runs > 1
    }

    fn spread(rows: u32, lanes: u32, context: u32) -> Option<Vec<u32>> {
        let lanes = lanes.min(rows);
        if lanes == 0 || u64::from(context) * u64::from(lanes) < u64::from(rows) {
            return None;
        }
        let base = rows / lanes;
        let over = rows % lanes;
        Some((0..lanes).map(|at| base + u32::from(at < over)).collect())
    }
}

fn joint_rows(
    deployment: &Deployment,
    present: &[usize],
    bucket: u32,
) -> Option<Vec<(usize, u32)>> {
    let width = present.len() as u32;
    let seats = deployment.seats.min(deployment.max_lanes);
    if width < 2 || seats < width || bucket < width {
        return None;
    }
    let lanes_per_class = (seats / width).max(1);
    let mut lanes = Vec::new();
    let base = bucket / width;
    let over = bucket % width;
    for (at, class) in present.iter().enumerate() {
        let rows = base + u32::from((at as u32) < over);
        for share in Shell::spread(rows, lanes_per_class, deployment.context)? {
            lanes.push((*class, share));
        }
    }
    Some(lanes)
}

fn fragment_rows(
    deployment: &Deployment,
    present: &[usize],
    bucket: u32,
) -> Option<Vec<(usize, u32)>> {
    let width = present.len() as u32;
    if width < 2 || deployment.seats < width || deployment.max_lanes < width || bucket < width {
        return None;
    }
    let decodes = present
        .iter()
        .filter(|class| deployment.decoding.contains(**class))
        .count() as u32;
    let prefilling: Vec<usize> = present
        .iter()
        .copied()
        .filter(|class| !deployment.decoding.contains(*class))
        .collect();
    if prefilling.is_empty() {
        return None;
    }
    let rows = Shell::spread(
        bucket - decodes,
        prefilling.len() as u32,
        deployment.context,
    )?;
    if rows.len() != prefilling.len() {
        return None;
    }
    let mut taken = rows.into_iter();
    Some(
        present
            .iter()
            .map(|class| {
                if deployment.decoding.contains(*class) {
                    (*class, 1u32)
                } else {
                    (*class, taken.next().unwrap_or(1))
                }
            })
            .collect(),
    )
}

fn evidence(walked: &[Vec<f32>], replayed: &[Vec<f32>]) -> Option<String> {
    fn ordered(bits: u32) -> u32 {
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        }
    }

    if walked.len() != replayed.len() {
        return Some(format!(
            "the two arms answered {} and {} lanes  class=structural",
            walked.len(),
            replayed.len(),
        ));
    }
    let mut first: Option<(usize, f32, f32)> = None;
    let mut differing = 0usize;
    let mut total = 0usize;
    let mut worst = 0u32;
    let mut worst_bf16 = 0u32;
    let mut worst_bf16_at: Option<(usize, usize, f32, f32)> = None;
    let mut beyond_one_bf16 = 0usize;
    let mut beyond_eight_bf16 = 0usize;
    let mut non_finite = (0usize, 0usize);
    let mut per_lane: Vec<usize> = Vec::with_capacity(walked.len());
    for (lane, (a, b)) in walked.iter().zip(replayed).enumerate() {
        per_lane.push(0);
        if a.len() != b.len() {
            return Some(format!(
                "lane {lane} answered {} and {} elements  class=structural",
                a.len(),
                b.len(),
            ));
        }
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            total += 1;
            if x.to_bits() == y.to_bits() {
                continue;
            }
            differing += 1;
            per_lane[lane] += 1;
            non_finite.0 += usize::from(!x.is_finite());
            non_finite.1 += usize::from(!y.is_finite());
            worst = worst.max(ordered(x.to_bits()).abs_diff(ordered(y.to_bits())));
            let steps = ordered(x.to_bits() & 0xffff_0000)
                .abs_diff(ordered(y.to_bits() & 0xffff_0000))
                >> 16;
            beyond_one_bf16 += usize::from(steps > 1);
            beyond_eight_bf16 += usize::from(steps > 8);
            if steps > worst_bf16 {
                worst_bf16 = steps;
                worst_bf16_at = Some((lane, at, *x, *y));
            }
            if first.is_none() {
                first = Some((at, *x, *y));
            }
        }
    }
    let (at, x, y) = first?;
    let class = if worst <= 2 { "numeric" } else { "structural" };
    let lane_runs = {
        let mut runs: Vec<String> = Vec::new();
        let mut start = 0usize;
        while start < per_lane.len() {
            let differs = per_lane[start] > 0;
            let mut end = start;
            while end < per_lane.len() && (per_lane[end] > 0) == differs {
                end += 1;
            }
            let width = walked[start].len().max(1);
            let mean = per_lane[start..end].iter().sum::<usize>() as f64
                / ((end - start) as f64 * width as f64);
            runs.push(if differs {
                format!("{start}..{end} differ ({:.0}% of cells)", mean * 100.0)
            } else {
                format!("{start}..{end} agree")
            });
            start = end;
        }
        runs.join(", ")
    };
    let worst_bf16_line = worst_bf16_at.map_or(String::new(), |(lane, at, x, y)| {
        format!("  worst_bf16=lane {lane} #{at} (walk={x}, body={y}, {worst_bf16} bf16 step(s))")
    });
    Some(format!(
        "differs at #{at} (walk={x} {:#010x}, body={y} {:#010x})  n_diff={differing}/{total}  \
         max_ulp={worst}  class={class}  beyond_1_bf16={beyond_one_bf16}  \
         beyond_8_bf16={beyond_eight_bf16}  non_finite(walk,body)={non_finite:?}  \
         lanes=[{lane_runs}]{worst_bf16_line}",
        x.to_bits(),
        y.to_bits(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{BodySynth, Deployment, LatticePoint, Targets, ensemble_keys};

    fn deployment(decoders: usize, point: LatticePoint) -> Deployment {
        Deployment {
            points: vec![point],
            buckets: Vec::new(),
            patch_points: Vec::new(),
            decoders: (0..decoders).collect(),
            prefilling: Vec::new(),
            media: Vec::new(),
            fragmenting: Vec::new(),
            joining: Vec::new(),
            decoding: model_ir::ClassSet::of(0..decoders),
            seats: point.lanes,
            context: 512,
            max_lanes: point.lanes,
            patch_fold: 1,
        }
    }

    fn lanes(target: &BodySynth) -> Vec<(usize, u32)> {
        match target {
            BodySynth::Ensemble { lanes } => lanes.clone(),
            other => panic!("the ensemble arm produced {other}"),
        }
    }

    #[test]
    fn two_decode_words_arm_the_pair_at_the_rungs_lane_count() {
        let mut found = Targets::default();
        ensemble_keys(
            &deployment(
                2,
                LatticePoint {
                    bucket: 256,
                    lanes: 256,
                },
            ),
            &mut found,
        );
        assert_eq!(
            found.targets.len(),
            1,
            "one key per rung, and there is one rung"
        );
        let (bucket, target) = &found.targets[0];
        assert_eq!(*bucket, 256);
        assert_eq!(
            lanes(target),
            vec![(0, 128), (1, 128)],
            "the rung's lanes are split across the two decode words"
        );
        assert_eq!(target.present(), vec![0, 1]);
        let (rows, media) = target.lanes();
        assert!(media.is_empty(), "an ensemble lane submits no image");
        assert_eq!(rows.len(), 256, "one lane per row");
        assert!(
            rows.iter().all(|(_, rows)| *rows == 1),
            "a decode lane is one row"
        );
        assert_eq!(rows.iter().map(|(_, rows)| *rows).sum::<u32>(), 256);
    }
}

#[derive(Debug, Clone)]
pub struct Armed {
    pub pool_line: String,
    pub wanted: usize,
    pub armed: usize,
    pub kinds: [(usize, usize); Kind::COUNT],
    pub segmented: usize,
    pub bytes: usize,
    pub unweighed: usize,
    pub bodies_mem: usize,
    pub bodies: usize,
    pub last_refusal: Option<String>,
    pub unfireable: Vec<String>,
    pub never: usize,
    pub never_from: u32,
    pub belted: bool,
    pub starved: Option<u64>,
    pub peer_stopped: bool,
    pub declines: u64,
    pub refusals: u64,
    pub seal: Seal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seal {
    Complete,
    Partial {
        never: usize,
    },
    Open,
}

impl core::fmt::Display for Armed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let columns = Kind::ALL
            .iter()
            .map(|kind| {
                format!(
                    "{kind} {}/{}",
                    self.kinds[kind.at()].0,
                    self.kinds[kind.at()].1
                )
            })
            .collect::<Vec<String>>()
            .join(", ");
        write!(
            f,
            "bodies armed {} of {} compositions at load ({columns}; {} segmented; {} MiB of {} \
             MiB, {} of {} seats)",
            self.armed,
            self.wanted,
            self.segmented,
            self.bytes >> 20,
            self.bodies_mem >> 20,
            self.bodies,
            record::MAX_BODIES,
        )?;
        if self.unweighed > 0 {
            write!(f, " ({} unweighed)", self.unweighed)?;
        }
        write!(f, " ({})", self.pool_line)?;
        if let Some(why) = &self.last_refusal {
            write!(f, " (last refusal: {why})")?;
        }
        if !self.unfireable.is_empty() {
            write!(
                f,
                " [{} key(s) never fired — this deployment cannot synthesize them, or their \
                 present set was already refused admission; e.g. {}]",
                self.unfireable.len(),
                self.unfireable[0],
            )?;
        }
        match self.seal {
            Seal::Complete => write!(f, " [sealed: every key attempted]")?,
            Seal::Partial { never } if never > 0 => write!(
                f,
                " [sealed partial: {never} key(s) never attempted, {} at bucket {}, and they \
                 walk eagerly for the life of this load]",
                match self.starved {
                    _ if self.peer_stopped =>
                        "a peer rank's bound (the stop is agreed across the group)".to_string(),
                    Some(spare) => format!(
                        "the ceiling's spare ran out ({} MiB left beyond the elastic pool's growth)",
                        spare >> 20
                    ),
                    None if self.belted => "record::MAX_BODIES".to_string(),
                    None => "[engine] bodies_mem".to_string(),
                },
                self.never_from,
            )?,
            Seal::Partial { .. } => write!(f, " [sealed partial: a key refused]")?,
            Seal::Open => write!(f, " [not sealed: nothing armed]")?,
        }
        if self.declines != 0 || self.refusals != 0 {
            write!(
                f,
                " [{} declined a workspace grant, {} inadmissible]",
                self.declines, self.refusals
            )?;
        }
        Ok(())
    }
}
