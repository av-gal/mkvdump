use std::{num::TryFromIntError, string::FromUtf8Error};

/// An Error while parsing Matroska/WebM files
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Error {
    /// Need data
    #[error("need data")]
    NeedData,
    /// Invalid ID
    #[error("invalid id")]
    InvalidId,
    /// Invalid Varint
    #[error("invalid varint")]
    InvalidVarint,
    /// Unknown Element size where it's forbidden
    #[error("forbidden unknown size")]
    ForbiddenUnknownSize,
    /// Error building UTF-8 string
    #[error("{0}")]
    Utf8(#[from] FromUtf8Error),
    /// Forbidden Integer size
    #[error("forbidden integer size")]
    ForbiddenIntegerSize,
    /// Forbidden Float size
    #[error("forbidden float size")]
    ForbiddenFloatSize,
    /// No valid element found
    #[error("valid element not found")]
    ValidElementNotFound,
    /// Missing track number
    #[error("missing track number")]
    InvalidTrackNumber,
    /// Overflow
    #[error("overflow")]
    Overflow(#[from] TryFromIntError),
    /// Invalid Date
    #[error("invalid date")]
    InvalidDate,
}
