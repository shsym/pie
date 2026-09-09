#[macro_export]
macro_rules! declare_tagged_enum {
    (
        $(#[$enum_meta:meta])*
        $vis:vis enum $enum_name:ident {
            $($(#[$variant_meta:meta])* $variant:ident = $tag:literal, $spelling:literal;)*
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(u8)]
        $vis enum $enum_name {
            $($(#[$variant_meta])* $variant = $tag,)*
        }

        impl $enum_name {
            pub const ALL: &'static [$enum_name] = &[$($enum_name::$variant,)*];

            pub fn from_u8(tag: u8) -> Option<Self> {
                Some(match tag {
                    $($tag => $enum_name::$variant,)*
                    _ => return None,
                })
            }

            pub fn name(self) -> &'static str {
                match self {
                    $($enum_name::$variant => $spelling,)*
                }
            }
        }
    };
}
