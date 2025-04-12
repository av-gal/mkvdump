use crate::{Error, IResult};

/// A Variable Size Integer, as defined in the [EBML spec](https://datatracker.ietf.org/doc/html/rfc8794#name-variable-size-integer)
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Serialize)]
pub struct VarInt {
    val: u64,
    len: u8,
}

impl VarInt {
    /// Construct a VarInt with the minimum length for its value
    pub fn new(val: u64) -> Self {
        Self {
            val,
            len: (val.checked_ilog2().unwrap_or(0) as u8 / 7) + 1,
        }
    }

    /// Construct a VarInt with the given number of bytes.
    ///
    /// Panics if `val` cannot be contained within the given `len`.
    pub fn new_with_length(val: u64, len: u8) -> Self {
        // TODO find something more efficient for this
        assert!(len > val.checked_ilog2().unwrap_or(0) as u8 / 7);

        Self { val, len }
    }

    /// Returns the maximum VarInt with a given length.
    pub fn max_with_size(len: u8) -> Self {
        Self {
            val: 2u64.pow(7 * len as u32) - 1,
            len,
        }
    }

    /// Returns `true` if this `VarInt` is the maximum for its length
    pub fn is_max(&self) -> bool {
        *self == Self::max_with_size(self.len)
    }

    pub fn len(&self) -> u8 {
        self.len
    }

    pub fn as_u64(&self) -> u64 {
        self.val
    }
}

pub fn parse_varint(first_input: &[u8]) -> IResult<VarInt, &[u8]> {
    let (first_byte, _) = first_input.split_first().ok_or(Error::NeedData)?;

    let vint_prefix_size = first_byte.leading_zeros() as usize + 1;

    // Maximum 8 bytes, i.e. first byte can't be 0
    // TODO: Use EBMLMaxSizeLength instead of hardcoding 8 bytes
    if vint_prefix_size > 8 {
        return Err(Error::InvalidVarint);
    }

    let (varint_bytes, input) = first_input
        .split_at_checked(vint_prefix_size)
        .ok_or(Error::NeedData)?;
    // any efficient way to avoid this copy here?
    let mut value_buffer = [0u8; 8];
    value_buffer[(8 - vint_prefix_size)..].copy_from_slice(varint_bytes);
    let mut value = u64::from_be_bytes(value_buffer);

    // discard varint prefix (zeros + market bit)
    let num_bits_in_value = 7 * vint_prefix_size;
    let bitmask = (1 << num_bits_in_value) - 1;
    value &= bitmask;

    // If all VINT_DATA bits are set to 1, it's an unkown size/value
    // https://github.com/ietf-wg-cellar/ebml-specification/blob/master/specification.markdown#unknown-data-size
    //
    // In 32-bit plaforms, the conversion from u64 to usize will fail if the value
    // is bigger than u32::MAX.
    // let result = (value != bitmask).then(|| value.try_into()).transpose()?;

    Ok((
        VarInt {
            val: value,
            len: vint_prefix_size as u8,
        },
        input,
    ))
}

macro_rules! ebml_elements {
    ($($(#[doc = $doc:literal])* name = $element_name:ident, original_name = $original_name:expr, id = $id:expr, variant = $variant:ident, unknownsizeallowed = $unknownsizeallowed:literal;)+) => {
        use serde::{Serialize, Serializer};

        /// Matroska Element Type.
        #[derive(Debug, PartialEq)]
        pub enum Type {
            /// Unsigned
            Unsigned,
            /// Signed
            Signed,
            /// Float
            Float,
            /// String
            String,
            /// Utf8
            Utf8,
            /// Date
            Date,
            /// Master
            Master,
            /// Binary
            Binary,
        }

        /// Matroska Element ID.
        #[derive(Debug, PartialEq, Eq, Clone)]
        pub enum Id {
            /// Unknown ID containing the value parsed.
            Unknown(u32),
            /// Corrupted element. Used when there is a parsing error and a portion of the input is skipped.
            Corrupted,
            $(
                $(#[doc = $doc])*
                $element_name,
            )+
        }

        impl Id {
            /// Build a new ID from an u32. If the value does not represent a known element,
            /// an Unknown ID will be created.
            pub fn new(id: u32) -> Self {
                match id {
                    $($id => Self::$element_name,)+
                    _ => Self::Unknown(id)
                }
            }

            /// Build a special corrupted ID
            pub fn corrupted() -> Self {
                Self::Corrupted
            }

            /// Get type of element for this ID
            pub fn get_type(&self) -> Type {
                match self {
                    $(Id::$element_name => Type::$variant,)+
                    Id::Unknown(_) | Id::Corrupted => Type::Binary
                }
            }

            /// Get underlying integer value
            pub fn get_value(&self) -> Option<u32> {
                match self {
                    $(Id::$element_name => Some($id),)+
                    Id::Unknown(value) => Some(*value),
                    Id::Corrupted => None
                }
            }

            /// Whether this element is allowed to have an unknown size
            pub fn unknown_size_allowed(&self) -> Option<bool> {
                match self {
                    $(Id::$element_name => Some($unknownsizeallowed),)+
                    Id::Unknown(_) => None,
                    Id::Corrupted => None
                }
            }

            /// Returns the length of this ID, in bytes
            ///
            /// Returns zero if this ID is `Id::Corrupted`
            pub fn size(&self) -> u8 {
                if *self == Id::Corrupted {
                    0
                } else {
                    (self.get_value().unwrap() + 1).ilog2() as u8
                }
            }
        }
        }

        impl Serialize for Id {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                match *self {
                    $(Id::$element_name => s.serialize_str($original_name),)+
                    Id::Unknown(value) => s.serialize_str(&format!("0x{:X}", value)),
                    Id::Corrupted => s.serialize_str("Corrupted")
                }
            }
        }
    };
}

macro_rules! ebml_enumerations {
    ($($(#[doc = $enum_doc:expr])* $id:ident { $($(#[doc = $variant_doc:expr])* $variant:ident = $value:expr, original_label = $original_label:expr;)+ };)+) => {
        use crate::elements::Id;
        use serde::Serialize;

        $(
            $(#[doc = $enum_doc])*
            #[derive(Debug, PartialEq, Eq, Clone, Serialize)]
            pub enum $id {
                $(
                    $(#[doc = $variant_doc])*
                    #[serde(rename = $original_label)]
                    $variant,
                )+
            }

            impl $id {
                /// Create a new instance
                pub fn new(value: u64) -> Option<Self> {
                    match value {
                        $($value => Some(Self::$variant),)+
                        _ => None,
                    }
                }
            }
        )+

        /// Enumeration of values for a given Matroska Element.
        #[derive(Debug, PartialEq, Eq, Clone, Serialize)]
        #[serde(untagged)]
        pub enum Enumeration {
            $(
                $(#[doc = $enum_doc])*
                $id($id),
            )+
        }

        impl Enumeration {
            /// Create new enumeration
            pub fn new(id: &Id, value: u64) -> Option<Self> {
                match id {
                    $(
                        Id::$id => $id::new(value).map(Self::$id),
                    )+
                    _ => None
                }
            }
        }
    };
}

pub(crate) use ebml_elements;
pub(crate) use ebml_enumerations;
use serde::Serialize;
