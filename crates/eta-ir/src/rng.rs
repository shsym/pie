#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64Round {
    pub xor_shift: u32,
    pub multiplier: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RngFormula {
    pub splitmix64_rounds: [SplitMix64Round; 3],
    pub lane_stride: u64,
    pub lane_index_bias: u64,
    pub uniform_mantissa_shift: u32,
    pub uniform_mantissa_bits: u32,
    pub uniform_midpoint: f32,
    pub ambient_seed_xor: u64,
    pub keyed_word_bits: u32,
}

pub const RNG_FORMULA: RngFormula = RngFormula {
    splitmix64_rounds: [
        SplitMix64Round {
            xor_shift: 27,
            multiplier: Some(0x3C79_AC49_2BA7_B653),
        },
        SplitMix64Round {
            xor_shift: 33,
            multiplier: Some(0x1C69_B3F7_4AC4_AE35),
        },
        SplitMix64Round {
            xor_shift: 27,
            multiplier: None,
        },
    ],
    lane_stride: 0x9E37_79B9_7F4A_7C15,
    lane_index_bias: 1,
    uniform_mantissa_shift: 40,
    uniform_mantissa_bits: 24,
    uniform_midpoint: 0.5,
    ambient_seed_xor: 0xA5A5_A5A5,
    keyed_word_bits: 32,
};

#[inline]
pub fn splitmix64(mut value: u64) -> u64 {
    for round in RNG_FORMULA.splitmix64_rounds {
        value ^= value >> round.xor_shift;
        if let Some(multiplier) = round.multiplier {
            value = value.wrapping_mul(multiplier);
        }
    }
    value
}

#[inline]
pub fn seed_eff(seed: u32) -> u64 {
    seed as u64 ^ RNG_FORMULA.ambient_seed_xor
}

#[inline]
pub fn stream_salt(stream: u32) -> u64 {
    splitmix64((stream as u64).wrapping_mul(RNG_FORMULA.lane_stride))
}

#[inline]
pub fn seed_eff_stream(seed: u32, stream: u32) -> u64 {
    seed_eff(seed) ^ stream_salt(stream)
}

#[inline]
pub fn keyed_seed(key: u32, counter: u32) -> u64 {
    splitmix64(((key as u64) << RNG_FORMULA.keyed_word_bits) | counter as u64)
}

pub const UNIFORM_MAX: f32 = 1.0 - f32::EPSILON / 2.0;

#[inline]
pub fn hash_uniform(seed_eff: u64, index: u32) -> f32 {
    let x = seed_eff.wrapping_add(
        RNG_FORMULA
            .lane_stride
            .wrapping_mul(index as u64 + RNG_FORMULA.lane_index_bias),
    );
    let bits = (splitmix64(x) >> RNG_FORMULA.uniform_mantissa_shift) as u32;
    let denominator = (1u32 << RNG_FORMULA.uniform_mantissa_bits) as f32;
    let raw = (bits as f32 + RNG_FORMULA.uniform_midpoint) * (1.0 / denominator);
    if raw < UNIFORM_MAX { raw } else { UNIFORM_MAX }
}

pub const NORMAL_PAIR_STRIDE: u32 = 2;

pub const NORMAL_TWO_PI: f32 = 6.283_185_5;

#[cfg(feature = "std")]
#[inline]
pub fn hash_normal(seed_eff: u64, index: u32) -> f32 {
    let lane = index.wrapping_mul(NORMAL_PAIR_STRIDE);
    let u0 = hash_uniform(seed_eff, lane);
    let u1 = hash_uniform(seed_eff, lane.wrapping_add(1));
    let radius = (-2.0f32 * u0.ln()).sqrt();
    radius * (NORMAL_TWO_PI * u1).cos()
}
