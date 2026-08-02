//! Error types for the OpenVINO backend.

/// A recoverable error originating from the OpenVINO runtime.
#[derive(thiserror::Error, Debug)]
pub enum OpenVinoError {
    #[error("{0}")]
    Message(String),

    #[error("OpenVINO runtime error: {0}")]
    Runtime(String),
}

/// Convenience alias.
pub type OpenVinoResult<T> = std::result::Result<T, OpenVinoError>;

impl From<String> for OpenVinoError {
    fn from(s: String) -> Self {
        OpenVinoError::Message(s)
    }
}

impl From<&str> for OpenVinoError {
    fn from(s: &str) -> Self {
        OpenVinoError::Message(s.to_string())
    }
}

/// Convert from any openvino crate error type by formatting its Display impl.
pub fn from_ov_error(e: impl std::fmt::Display) -> OpenVinoError {
    OpenVinoError::Runtime(e.to_string())
}
