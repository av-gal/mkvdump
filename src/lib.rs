#![forbid(missing_docs)]

//! Parse MKV and WebM content
//!
//! Provides a set of Matroska structures and
//! functions to parse Matroska elements.

use std::ops::Not;

use ebml::{parse_varint, VarInt};
use serde::{Serialize, Serializer};
use serde_with::skip_serializing_none;

mod ebml;
/// Matroska elements
pub mod elements;
/// Matroska enumerations
pub mod enumerations;
mod error;
/// The tree module contains helpers for building tree
/// structures from parsed elements
pub mod tree;

use crate::elements::{Id, Type};
use crate::enumerations::Enumeration;
pub use error::Error;

/// Result type helper
pub type Result<T> = std::result::Result<T, Error>;
type IResult<T, O> = Result<(T, O)>;

/// Trait to serialize this object into a canonical representation
/// 
/// "Canonical" means both that these bytes should be the object's only possible serialization,
/// and that there should be no other object that produces these bytes.
pub trait Generate {
    /// Attempt to parse this item out of the buffer.
    ///
    /// Returns the item and a buffer of the remaining data, or an error.
    fn generate(&self) -> Result<&[u8]>
    where
        Self: std::marker::Sized;
}

/// Trait to deserialize an object from bytes.
///
/// Part of this trait's contract is that the deserialised object should have a one-to-one
/// relationship with the underlying bytes: no other sequence of bytes should produce exactly
/// the same struct.
pub trait Parse {
    /// Attempt to parse this item out of the buffer.
    ///
    /// Returns the item and a buffer of the remaining data, or an error.
    fn parse(input: &[u8]) -> IResult<Self, &[u8]>
    where
        Self: std::marker::Sized;
}

impl Parse for Id {
    fn parse(input: &[u8]) -> IResult<Self, &[u8]> {
        let (first_byte, _) = input.split_first().ok_or(Error::NeedData)?;
        // SAFETY: A byte will never have more than eight leading zeroes, which will always fit into usize
        let num_bytes = first_byte.leading_zeros() as usize + 1;

        // IDs can only have up to 4 bytes in Matroska
        // TODO: Use EBMLMaxIDLength instead of hardcoding 4 bytes
        if num_bytes > 4 {
            return Err(Error::InvalidId);
        }

        let (varint_bytes, input) = input.split_at_checked(num_bytes).ok_or(Error::NeedData)?;
        let mut value_buffer = [0u8; 4];
        value_buffer[(4 - varint_bytes.len())..].copy_from_slice(varint_bytes);
        let id = u32::from_be_bytes(value_buffer);

        Ok((Id::new(id), input))
    }
}

/// Represents an [EBML Header](https://github.com/ietf-wg-cellar/ebml-specification/blob/master/specification.markdown#ebml-header)
#[skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Header {
    /// The Element ID
    pub id: Id,
    /// Size of the Element Body
    #[serde(serialize_with = "serialize_size")]
    body_size: VarInt,
    /// Position in the input
    pub position: Option<usize>,
}

//todo fix
fn serialize_size<S: Serializer>(size: &VarInt, s: S) -> std::result::Result<S::Ok, S::Error> {
    if size.is_max() {
        s.serialize_str("Unknown")
    } else {
        s.serialize_u64(size.as_u64())
    }
}

impl Header {
    /// Create a new Header
    pub fn new(id: Id, body_size: VarInt) -> Self {
        Self {
            id,
            body_size,
            position: None,
        }
    }

    /// Returns the length of the header itself in bytes.
    ///
    /// The length of the header is zero if the element is corrupted.
    pub fn size(&self) -> u8 {
        if self.id == Id::Corrupted {
            0
        } else {
            self.id.size() + self.body_size.len()
        }
    }

    // fn with_unknown_size(id: Id, header_size: usize) -> Self {
    //     Self {
    //         id,
    //         header_size,
    //         body_size: None,
    //         position: None,
    //     }
    // }
}

impl Parse for Header {
    fn parse(input: &[u8]) -> IResult<Self, &[u8]>
    where
        Self: std::marker::Sized {
            let (id, input) = Id::parse(input)?;
            let (body_size, input) = parse_varint(input)?;
        
            // Only Segment and Cluster have unknownsizeallowed="1" in ebml_matroska.xml.
            // Also mentioned in https://www.w3.org/TR/mse-byte-stream-format-webm/
            if body_size == VarInt::max_with_size(body_size.len())
                && !id.unknown_size_allowed().unwrap_or(true)
            {
                return Err(Error::ForbiddenUnknownSize);
            }
        
            Ok((Header::new(id, body_size), input))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
enum Lacing {
    Xiph,
    Ebml,
    FixedSize,
}

/// A Matroska [Block](https://www.matroska.org/technical/basics.html#block-structure)
#[skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Block {
    track_number: VarInt,
    timestamp: i16,
    #[serde(skip_serializing_if = "Not::not")]
    invisible: bool,
    lacing: Option<Lacing>,
    num_frames: Option<u8>,
}

/// A Matroska [SimpleBlock](https://www.matroska.org/technical/basics.html#simpleblock-structure)
#[skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SimpleBlock {
    track_number: VarInt,
    timestamp: i16,
    #[serde(skip_serializing_if = "Not::not")]
    keyframe: bool,
    #[serde(skip_serializing_if = "Not::not")]
    invisible: bool,
    lacing: Option<Lacing>,
    #[serde(skip_serializing_if = "Not::not")]
    discardable: bool,
    num_frames: Option<u8>,
}

/// Enumeration with possible binary value payloads
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Binary {
    /// A standard binary payload that will not be parsed further
    Standard(String),
    /// A SeekId payload
    SeekId(Id),
    /// A SimpleBlock
    SimpleBlock(SimpleBlock),
    /// A Block
    Block(Block),
    /// Void
    Void,
    /// Represents the payload of a corrupted region of the file
    Corrupted,
}

// impl Parse for Binary {
//     fn parse(input: &[u8]) -> IResult<Self, &[u8]>
//     where
//         Self: std::marker::Sized {
//             let (binary, input) = peek_binary(header, input)?;
//             // Actually consume the bytes from the body
//             let (_, input) = input
//                 .split_at_checked(
//                     header
//                         .body_size
//                         .as_u64()
//                         .try_into()
//                         .expect("value of u64 > usize!"),
//                 )
//                 .ok_or(Error::NeedData)?;
//             Ok((binary, input))
//     }
// }

fn parse_binary<'a>(header: &Header, input: &'a [u8]) -> IResult<Binary, &'a [u8]> {
    let (binary, input) = peek_binary(header, input)?;
    // Actually consume the bytes from the body
    let (_, input) = input
        .split_at_checked(
            header
                .body_size
                .as_u64()
                .try_into()
                .expect("value of u64 > usize!"),
        )
        .ok_or(Error::NeedData)?;
    Ok((binary, input))
}

/// Peek into Binary body without advancing the buffer.
///
/// It may be useful to parse just the first bytes of the binary body
/// without requiring the whole binary to be loaded into memory.
pub fn peek_binary<'a>(header: &Header, input: &'a [u8]) -> IResult<Binary, &'a [u8]> {
    let binary = match header.id {
        Id::SeekId => Binary::SeekId(Id::parse(input)?.0),
        Id::SimpleBlock => Binary::SimpleBlock(parse_simple_block(input)?.0),
        Id::Block => Binary::Block(parse_block(input)?.0),
        Id::Void => Binary::Void,
        _ => Binary::Standard(peek_standard_binary(
            input,
            header
                .body_size
                .as_u64()
                .try_into()
                .expect("value in u64 > usize::MAX on this system!"),
        )?),
    };

    Ok((binary, input))
}

fn peek_standard_binary(input: &[u8], size: usize) -> Result<String> {
    const MAX_LENGTH: usize = 64;
    if size <= MAX_LENGTH {
        let (bytes, _) = input.split_at_checked(size).ok_or(Error::NeedData)?;
        let string_values = bytes
            .iter()
            .map(|n| format!("{:02x}", n))
            .fold("".to_owned(), |acc, s| acc + &s + " ")
            .trim_end()
            .to_owned();
        Ok(format!("[{}]", string_values))
    } else {
        Ok(format!("{} bytes", size))
    }
}

/// An unsigned value that may contain an enumeration
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Unsigned {
    /// An standard value
    Standard(u64),
    /// An enumerated value
    Enumeration(Enumeration),
}

impl Unsigned {
    fn new(id: &Id, value: u64) -> Self {
        Enumeration::new(id, value).map_or(Self::Standard(value), Self::Enumeration)
    }
}

/// A date-and-timestamp stored as an i64, representing the number of nanoseconds from 2001-01-01T00:00:00 UTC
#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize)]
pub struct EbmlDate(i64);

#[cfg(feature = "chrono")]
impl From<EbmlDate> for chrono::DateTime<chrono::Utc> {
    fn from(value: EbmlDate) -> Self {
        use chrono::prelude::*;

        let nanos_2001 = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).unwrap();

        nanos_2001 + chrono::TimeDelta::nanoseconds(value.0)
    }
}

#[cfg(feature = "time")]
impl From<EbmlDate> for time::UtcDateTime {
    fn from(value: EbmlDate) -> Self {
        use time::{Date, Duration, Month, Time, UtcDateTime};

        UtcDateTime::new(
            Date::from_calendar_date(2001, Month::January, 1).unwrap(),
            Time::from_hms(0, 0, 0).unwrap(),
        ) + Duration::nanoseconds(value.0)
    }
}

#[cfg(feature = "jiff")]
impl From<EbmlDate> for jiff::Zoned {
    fn from(value: EbmlDate) -> Self {
        use jiff::{civil::date, tz::TimeZone, ToSpan};

        &date(2001, 1, 1)
            .at(0, 0, 0, 0)
            .to_zoned(TimeZone::UTC)
            .unwrap()
            + (value.0).nanoseconds()
    }
}

/// An [EBML Body](https://github.com/ietf-wg-cellar/ebml-specification/blob/master/specification.markdown#ebml-body)
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Body {
    /// A Master Body contains no data, but will contain zero or more elements
    /// that come after it.
    Master,
    /// An Unsigned Integer that may contain a known Enumeration
    Unsigned(Unsigned),
    /// A Signed Integer
    Signed(i64),
    /// A Float
    Float(f64),
    /// A String
    String(String),
    /// An UTF-8 String
    Utf8(String),
    /// A Date
    Date(EbmlDate),
    /// A Binary
    Binary(Binary),
}

/// Represents an EBML Element
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Element {
    /// The Header
    #[serde(flatten)]
    pub header: Header,
    /// The Body
    #[serde(rename = "value")]
    pub body: Body,
}

impl Element {
    /// Returns the length of this element header and body, or none if the element body size is undefined.
    pub fn size(&self) -> Option<u64> {
        if self.header.body_size.is_max() {
            None
        } else {
            Some(self.header.body_size.as_u64() + self.header.size() as u64)
        }
    }
}

const SYNC_ELEMENT_IDS: &[Id] = &[
    Id::Cluster,
    Id::Ebml,
    Id::Segment,
    Id::SeekHead,
    Id::Info,
    Id::Tracks,
    Id::Cues,
    Id::Attachments,
    Id::Chapters,
    Id::Tags,
];

/// Parse corrupt area
///
/// If we ever hit a damaged element, we can try to recover by finding
/// one of those IDs next to start clean. Those are the 4-bytes IDs,
/// which according to the EBML spec:
/// "Four-octet Element IDs are somewhat special in that they are useful
/// for resynchronizing to major structures in the event of data corruption or loss."
///
/// This parser either stops once a valid sync id or consumes the whole buffer.
/// It returns NeedData if the input is an empty slice.
pub fn parse_corrupt(input: &[u8]) -> IResult<Element, &[u8]> {
    const SYNC_ID_LEN: usize = 4;

    if input.is_empty() {
        return Err(Error::NeedData);
    }

    for (offset, window) in input.windows(SYNC_ID_LEN).enumerate() {
        for sync_id in SYNC_ELEMENT_IDS {
            let id_value = sync_id.get_value().unwrap();
            let id_bytes = id_value.to_be_bytes();
            if window == id_bytes {
                // TODO: we might want to try and parse the element here, because if the
                // the sync element header itself is corrupt (e.g. invalid varint), then
                // the consuming side might step into an infinite loop.
                return Ok((
                    Element {
                        header: Header::new(Id::corrupted(), VarInt::new(offset as u64)),
                        body: Body::Binary(Binary::Corrupted),
                    },
                    &input[offset..],
                ));
            }
        }
    }
    Ok((
        Element {
            header: Header::new(Id::corrupted(), VarInt::new(input.len() as u64)),
            body: Body::Binary(Binary::Corrupted),
        },
        &[],
    ))
}

impl Parse for Element {
    fn parse(original_input: &[u8]) -> IResult<Element, &[u8]> {
        let (header, input) = Header::parse(original_input)?;

        let (body, input) = parse_body(&header, input)?;
    
        let element = Element { header, body };
        Ok((element, input))
    }
}

/// Parse element body
pub fn parse_body<'a>(header: &Header, input: &'a [u8]) -> IResult<Body, &'a [u8]> {
    let element_type = header.id.get_type();
    let (input, body) = match element_type {
        Type::Master => (input, Body::Master),
        Type::Unsigned => {
            let (value, input) = parse_int(header, input)?;
            (input, Body::Unsigned(Unsigned::new(&header.id, value)))
        }
        Type::Signed => {
            let (value, input) = parse_int(header, input)?;
            (input, Body::Signed(value))
        }
        Type::Float => {
            let (value, input) = parse_float(header, input)?;
            (input, Body::Float(value))
        }
        Type::String => {
            let (value, input) = parse_string(header, input)?;
            (input, Body::String(value))
        }
        Type::Utf8 => {
            let (value, input) = parse_string(header, input)?;
            (input, Body::Utf8(value))
        }
        Type::Date => {
            let (value, input) = parse_date(header, input)?;
            (input, Body::Date(value))
        }
        Type::Binary => {
            let (value, input) = parse_binary(header, input)?;
            (input, Body::Binary(value))
        }
    };
    Ok((body, input))
}

fn parse_string<'a>(header: &Header, input: &'a [u8]) -> IResult<String, &'a [u8]> {
    let body_size = header.body_size.as_u64();
    let (string_bytes, input) = input
        .split_at_checked(
            body_size
                .try_into()
                .expect("value of u64 > usize::MAX on this system!"),
        )
        .ok_or(Error::NeedData)?;
    let value = String::from_utf8(string_bytes.to_vec())?;

    // Remove trimming null characters
    let value = value.trim_end_matches('\0').to_string();

    Ok((value, input))
}

fn parse_date<'a>(header: &Header, input: &'a [u8]) -> IResult<EbmlDate, &'a [u8]> {
    let (timestamp_nanos_to_2001, input) = parse_int::<i64>(header, input)?;
    Ok((EbmlDate(timestamp_nanos_to_2001), input))
}

trait Integer64FromBigEndianBytes {
    fn from_be_bytes(input: [u8; 8]) -> Self;
}

impl Integer64FromBigEndianBytes for u64 {
    fn from_be_bytes(input: [u8; 8]) -> Self {
        u64::from_be_bytes(input)
    }
}

impl Integer64FromBigEndianBytes for i64 {
    fn from_be_bytes(input: [u8; 8]) -> Self {
        i64::from_be_bytes(input)
    }
}

fn parse_int<'a, T: Integer64FromBigEndianBytes>(
    header: &Header,
    input: &'a [u8],
) -> IResult<T, &'a [u8]> {
    let body_size = header.body_size.as_u64();
    if body_size > 8 {
        return Err(Error::ForbiddenIntegerSize);
    }

    let (int_bytes, input) = input
        .split_at_checked(
            body_size
                .try_into()
                .expect("value of u64 > usize::MAX on this system"),
        )
        .ok_or(Error::NeedData)?;

    let mut value_buffer = [0u8; 8];
    value_buffer[(8 - int_bytes.len())..].copy_from_slice(int_bytes);
    let value = T::from_be_bytes(value_buffer);

    Ok((value, input))
}

fn parse_float<'a>(header: &Header, input: &'a [u8]) -> IResult<f64, &'a [u8]> {
    let body_size = header.body_size.as_u64();

    if body_size == 4 {
        let (float_bytes, input) = input
            .split_at_checked(
                body_size
                    .try_into()
                    .expect("value of u64 > usize::MAX on this system!"),
            )
            .ok_or(Error::NeedData)?;
        let value = f32::from_be_bytes(float_bytes.try_into().unwrap()) as f64;
        Ok((value, input))
    } else if body_size == 8 {
        let (float_bytes, input) = input
            .split_at_checked(
                body_size
                    .try_into()
                    .expect("value of u64 > usize::MAX on this system!"),
            )
            .ok_or(Error::NeedData)?;
        let value = f64::from_be_bytes(float_bytes.try_into().unwrap());
        Ok((value, input))
    } else if body_size == 0 {
        Ok((0f64, input))
    } else {
        Err(Error::ForbiddenFloatSize)
    }
}

fn parse_i16(input: &[u8]) -> IResult<i16, &[u8]> {
    let (bytes, input) = input.split_at_checked(2).ok_or(Error::NeedData)?;
    let value = i16::from_be_bytes(bytes.try_into().unwrap());
    Ok((value, input))
}

fn is_invisible(flags: u8) -> bool {
    (flags & (1 << 3)) != 0
}

fn get_lacing(flags: u8) -> Option<Lacing> {
    match (flags & (0b110)) >> 1 {
        0b01 => Some(Lacing::Xiph),
        0b11 => Some(Lacing::Ebml),
        0b10 => Some(Lacing::FixedSize),
        _ => None,
    }
}

fn parse_block(input: &[u8]) -> IResult<Block, &[u8]> {
    let (track_number, input) = parse_varint(input)?;
    // Track number can only be 56 bits in length
    if track_number.len() > 7 {
        return Err(Error::InvalidTrackNumber);
    }
    let (timestamp, input) = parse_i16(input)?;
    let (&flags, input) = input.split_first().ok_or(Error::NeedData)?;

    let invisible = is_invisible(flags);
    let lacing = get_lacing(flags);
    let (num_frames, input) = if lacing.is_some() {
        let (num_frames, input) = input.split_first().ok_or(Error::NeedData)?;
        (Some(num_frames + 1), input)
    } else {
        (None, input)
    };

    Ok((
        Block {
            track_number,
            timestamp,
            invisible,
            lacing,
            num_frames,
        },
        input,
    ))
}

fn parse_simple_block(input: &[u8]) -> IResult<SimpleBlock, &[u8]> {
    let (track_number, input) = parse_varint(input)?;
    // Track number can only be 56 bits in length
    if track_number.len() > 7 {
        return Err(Error::InvalidTrackNumber);
    }
    let (timestamp, input) = parse_i16(input)?;
    let (&flags, input) = input.split_first().ok_or(Error::NeedData)?;

    let keyframe = (flags & (1 << 7)) != 0;
    let invisible = is_invisible(flags);
    let lacing = get_lacing(flags);
    let discardable = (flags & 0b1) != 0;
    let (num_frames, input) = if lacing.is_some() {
        let (num_frames, input) = input.split_first().ok_or(Error::NeedData)?;
        (Some(num_frames + 1), input)
    } else {
        (None, input)
    };

    Ok((
        SimpleBlock {
            track_number,
            timestamp,
            keyframe,
            invisible,
            lacing,
            discardable,
            num_frames,
        },
        input,
    ))
}

/// Helper to add resiliency to corrupt inputs
pub fn parse_element_or_corrupted(input: &[u8]) -> IResult<Element, &[u8]> {
    Element::parse(input).or_else(|_| parse_corrupt(input))
}

#[cfg(test)]
mod tests {
    use crate::enumerations::TrackType;

    use super::*;

    const EMPTY: &[u8] = &[];
    const UNKNOWN_VARINT: &[u8] = &[0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];

    #[test]
    fn test_parse_id() {
        assert_eq!(Id::parse(&[0x1A, 0x45, 0xDF, 0xA3]), Ok((Id::Ebml, EMPTY)));
        assert_eq!(Id::parse(&[0x42, 0x86]), Ok((Id::EbmlVersion, EMPTY)));
        assert_eq!(Id::parse(&[0x23, 0x83, 0xE3]), Ok((Id::FrameRate, EMPTY)));

        // 1 byte missing from FrameRate (3-bytes long)
        assert_eq!(Id::parse(&[0x23, 0x83]), Err(Error::NeedData));

        // Longer than 4 bytes
        const FAILURE_INPUT: &[u8] = &[0x08, 0x45, 0xDF, 0xA3];
        assert_eq!(Id::parse(FAILURE_INPUT), Err(Error::InvalidId));

        // Unknown ID
        let (id, remaining) = Id::parse(&[0x19, 0xAB, 0xCD, 0xEF]).unwrap();
        assert_eq!((&id, remaining), (&Id::Unknown(0x19ABCDEF), EMPTY));
        assert_eq!(serde_yaml::to_string(&id).unwrap().trim(), "'0x19ABCDEF'");
        assert_eq!(id.get_value().unwrap(), 0x19ABCDEF);
    }

    #[test]
    fn test_parse_varint() {
        assert_eq!(parse_varint(&[0x9F]), Ok((VarInt::new(31), EMPTY)));
        assert_eq!(parse_varint(&[0x81]), Ok((VarInt::new(1), EMPTY)));
        assert_eq!(parse_varint(&[0x53, 0xAC]), Ok((VarInt::new(5036), EMPTY)));

        const INVALID_VARINT: &[u8] = &[0x00, 0xAC];
        assert_eq!(parse_varint(INVALID_VARINT), Err(Error::InvalidVarint));

        assert_eq!(
            parse_varint(UNKNOWN_VARINT),
            Ok((VarInt::max_with_size(8), EMPTY))
        );
    }

    #[test]
    fn test_parse_element_header() {
        const INPUT: &[u8] = &[0x1A, 0x45, 0xDF, 0xA3, 0x9F];
        assert_eq!(
            Header::parse(INPUT),
            Ok((Header::new(Id::Ebml, VarInt::new(31)), EMPTY))
        );
    }

    #[test]
    fn test_parse_string() {
        assert_eq!(
            parse_string(
                &Header::new(Id::DocType, VarInt::new(4)),
                &[0x77, 0x65, 0x62, 0x6D]
            ),
            Ok(("webm".to_string(), EMPTY))
        );

        assert_eq!(
            parse_string(
                &Header::new(Id::DocType, VarInt::new(6)),
                &[0x77, 0x65, 0x62, 0x6D, 0x00, 0x00]
            ),
            Ok(("webm".to_string(), EMPTY))
        );

        // TODO Forbid this by construction
        // assert_eq!(
        //     parse_string(&Header::with_unknown_size(Id::DocType, 3), EMPTY),
        //     Err(Error::ForbiddenUnknownSize)
        // );
    }

    #[test]
    fn test_parse_corrupted() {
        // This integer would have more than 8 bytes.
        // It needs to find a valid 4-byte Element ID, but can't
        // so we get an incomplete.
        assert_eq!(
            Element::parse(&[0x42, 0x87, 0x90, 0x01]),
            Err(Error::ForbiddenIntegerSize)
        );

        // Now it finds a Segment.
        const SEGMENT_ID: &[u8] = &[0x18, 0x53, 0x80, 0x67];
        let (element, remaining) =
            parse_element_or_corrupted(&[0x42, 0x87, 0x90, 0x01, 0x18, 0x53, 0x80, 0x67]).unwrap();
        assert_eq!(
            (remaining, &element),
            (
                SEGMENT_ID,
                &Element {
                    header: Header::new(Id::corrupted(), VarInt::new(4)),
                    body: Body::Binary(Binary::Corrupted),
                },
            )
        );
        assert!(element.header.id.get_value().is_none());
    }

    #[test]
    fn test_parse_corrupted_unknown_size() {
        // String
        assert_eq!(
            Element::parse(&[0x86, 0xFF, 0x56, 0x5F, 0x54]),
            Err(Error::ForbiddenUnknownSize)
        );

        // Binary
        assert_eq!(
            Element::parse(&[0x63, 0xA2, 0xFF]),
            Err(Error::ForbiddenUnknownSize)
        );

        // Integer
        assert_eq!(
            Element::parse(&[0x42, 0x87, 0xFF, 0x01]),
            Err(Error::ForbiddenUnknownSize)
        );

        // Float
        assert_eq!(
            Element::parse(&[0x44, 0x89, 0xFF, 0x01]),
            Err(Error::ForbiddenUnknownSize)
        );
    }

    #[test]
    fn test_parse_int() {
        assert_eq!(
            parse_int(&Header::new(Id::EbmlVersion, VarInt::new(1)), &[0x01]),
            Ok((1u64, EMPTY))
        );
        // TODO forbid these headers by construction
        // assert_eq!(
        //     parse_int::<u64>(&Header::new(Id::EbmlVersion, 3,  ), EMPTY),
        //     Err(Error::ForbiddenUnknownSize)
        // );
        // assert_eq!(
        //     parse_int::<i64>(&Header::with_unknown_size(Id::EbmlVersion, 3), EMPTY),
        //     Err(Error::ForbiddenUnknownSize)
        // );
    }

    #[test]
    fn test_parse_float() {
        assert_eq!(
            parse_float(
                &Header::new(Id::Duration, VarInt::new(4)),
                &[0x45, 0x7A, 0x30, 0x00]
            ),
            Ok((4003., EMPTY))
        );
        assert_eq!(
            parse_float(
                &Header::new(Id::Duration, VarInt::new(8)),
                &[0x40, 0xAF, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00]
            ),
            Ok((4003., EMPTY))
        );
        assert_eq!(
            parse_float(&Header::new(Id::Duration, VarInt::new(0)), EMPTY),
            Ok((0., EMPTY))
        );
        assert_eq!(
            parse_float(&Header::new(Id::Duration, VarInt::new(7)), EMPTY),
            Err(Error::ForbiddenFloatSize)
        );
        // TODO forbid these headers by construction
        // assert_eq!(
        //     parse_float(&Header::with_unknown_size(Id::Duration, 3), EMPTY),
        //     Err(Error::ForbiddenUnknownSize)
        // );
    }

    #[test]
    fn test_parse_binary() {
        const BODY: &[u8] = &[0x15, 0x49, 0xA9, 0x66];
        assert_eq!(
            parse_binary(&Header::new(Id::SeekId, VarInt::new(4)), BODY),
            Ok((Binary::SeekId(Id::Info), EMPTY))
        );
        // TODO forbid these headers by construction
        // assert_eq!(
        //     parse_binary(&Header::with_unknown_size(Id::SeekId, 3), EMPTY),
        //     Err(Error::ForbiddenUnknownSize)
        // );
    }

    #[test]
    fn test_parse_date() {
        let expected_datetime = EbmlDate(681899235000000000);

        assert_eq!(
            parse_date(
                &Header::new(Id::DateUtc, VarInt::new(8)),
                &[0x09, 0x76, 0x97, 0xbd, 0xca, 0xc9, 0x1e, 0x00]
            ),
            Ok((expected_datetime, EMPTY))
        )
    }

    #[cfg(feature = "chrono")]
    #[test]
    fn test_date_chrono() {
        use chrono::prelude::*;
        let expected_datetime = Utc.from_utc_datetime(
            &NaiveDate::from_ymd_opt(2022, 8, 11)
                .unwrap()
                .and_hms_opt(8, 27, 15)
                .unwrap(),
        );

        assert_eq!(
            DateTime::<Utc>::from(EbmlDate(681899235000000000)),
            expected_datetime
        )
    }

    #[cfg(feature = "time")]
    #[test]
    fn test_date_time() {
        use time::{Date, Month, Time, UtcDateTime};
        let expected_datetime = UtcDateTime::new(
            Date::from_calendar_date(2022, Month::August, 11).unwrap(),
            Time::from_hms(8, 27, 15).unwrap(),
        );

        assert_eq!(
            UtcDateTime::from(EbmlDate(681899235000000000)),
            expected_datetime
        )
    }

    #[cfg(feature = "jiff")]
    #[test]
    fn test_date_jiff() {
        use jiff::{civil::date, tz::TimeZone, Zoned};
        let expected_datetime = date(2022, 8, 11)
            .at(8, 27, 15, 0)
            .to_zoned(TimeZone::UTC)
            .unwrap();

        assert_eq!(Zoned::from(EbmlDate(681899235000000000)), expected_datetime)
    }

    #[test]
    fn test_parse_master_element() {
        const INPUT: &[u8] = &[
            0x1A, 0x45, 0xDF, 0xA3, 0x9F, 0x42, 0x86, 0x81, 0x01, 0x42, 0xF7, 0x81, 0x01, 0x42,
            0xF2, 0x81, 0x04, 0x42, 0xF3, 0x81, 0x08, 0x42, 0x82, 0x84, 0x77, 0x65, 0x62, 0x6D,
            0x42, 0x87, 0x81, 0x04, 0x42, 0x85, 0x81, 0x02,
        ];

        let result = Element::parse(INPUT);
        assert_eq!(
            result,
            Ok((
                Element {
                    header: Header::new(Id::Ebml, VarInt::new(31)),
                    body: Body::Master
                },
                &INPUT[5..],
            ))
        );
    }

    #[test]
    fn test_parse_enumeration() {
        const INPUT: &[u8] = &[0x83, 0x81, 0x01];
        assert_eq!(
            Element::parse(INPUT),
            Ok((
                Element {
                    header: Header::new(Id::TrackType, VarInt::new(1)),
                    body: Body::Unsigned(Unsigned::Enumeration(Enumeration::TrackType(
                        TrackType::Video
                    )))
                },
                EMPTY,
            ))
        );

        const INPUT_UNKNOWN_ENUMERATION: &[u8] = &[0x83, 0x81, 0xFF];
        let (element, remaining) = Element::parse(INPUT_UNKNOWN_ENUMERATION).unwrap();
        assert_eq!(
            (remaining, &element),
            (
                EMPTY,
                &Element {
                    header: Header::new(Id::TrackType, VarInt::new(1)),
                    body: Body::Unsigned(Unsigned::Standard(255))
                }
            )
        );
        assert_eq!(
            serde_yaml::to_string(&element).unwrap().trim(),
            "id: TrackType\nheader_size: 2\nbody_size: 1\nvalue: 255"
        );
    }

    #[test]
    fn test_parse_seek_id() {
        assert_eq!(
            Element::parse(&[0x53, 0xAB, 0x84, 0x15, 0x49, 0xA9, 0x66]),
            Ok((
                Element {
                    header: Header::new(Id::SeekId, VarInt::new(4)),
                    body: Body::Binary(Binary::SeekId(Id::Info))
                },
                EMPTY,
            ))
        );
    }

    #[test]
    fn test_parse_crc32() {
        assert_eq!(
            Element::parse(&[0xBF, 0x84, 0xAF, 0x93, 0x97, 0x18]),
            Ok((
                Element {
                    header: Header::new(Id::Crc32, VarInt::new(4)),
                    body: Body::Binary(Binary::Standard("[af 93 97 18]".into()))
                },
                EMPTY,
            ))
        );
    }

    #[test]
    fn test_parse_empty() {
        assert_eq!(
            Element::parse(&[0x63, 0xC0, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            Ok((
                Element {
                    header: Header::new(Id::Targets, VarInt::new_with_length(0, 8)),
                    body: Body::Master
                },
                EMPTY,
            ))
        );
    }

    #[test]
    fn test_parse_block() {
        assert_eq!(
            parse_block(&[0x81, 0x0F, 0x7A, 0x00]),
            Ok((
                Block {
                    track_number: VarInt::new(1),
                    timestamp: 3962,
                    invisible: false,
                    lacing: None,
                    num_frames: None
                },
                EMPTY,
            ))
        );

        assert_eq!(parse_block(UNKNOWN_VARINT), Err(Error::InvalidTrackNumber));
    }

    #[test]
    fn test_parse_simple_block() {
        assert_eq!(
            parse_simple_block(&[0x81, 0x00, 0x53, 0x00]),
            Ok((
                SimpleBlock {
                    track_number: VarInt::new(1),
                    timestamp: 83,
                    keyframe: false,
                    invisible: false,
                    lacing: None,
                    discardable: false,
                    num_frames: None,
                },
                EMPTY,
            ))
        );

        assert_eq!(
            parse_simple_block(UNKNOWN_VARINT),
            Err(Error::InvalidTrackNumber)
        );
    }

    #[test]
    fn test_peek_standard_binary() -> Result<()> {
        let input = &[1, 2, 3];
        assert_eq!(peek_standard_binary(input, 3)?, "[01 02 03]");

        let input = &[0; 5];
        assert_eq!(peek_standard_binary(input, 65)?, "65 bytes");
        Ok(())
    }

    #[test]
    fn test_serialize_enumeration() {
        assert_eq!(
            serde_yaml::to_string(&Enumeration::TrackType(TrackType::Video))
                .unwrap()
                .trim(),
            "video"
        );
        assert_eq!(
            serde_yaml::to_string(&Unsigned::Standard(5))
                .unwrap()
                .trim(),
            "5"
        );
    }

    #[test]
    fn test_parse_corrupt() {
        // can not find a valid sync id in  a bonkers array, so it should consume the
        // entire buffer
        assert_eq!(
            parse_corrupt(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            Ok((
                Element {
                    header: Header::new(Id::corrupted(), VarInt::new(10)),
                    body: Body::Binary(Binary::Corrupted)
                },
                EMPTY,
            ))
        );
    }
}
