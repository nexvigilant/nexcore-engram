// Copyright (c) 2026 Matthew Campion, PharmD; NexVigilant
// All Rights Reserved. See LICENSE file for details.

//! # Engram — Unit of Persistent Knowledge
//!
//! An engram is the fundamental unit of knowledge in the unified memory system.
//! Each engram has a source layer, tags, temporal metadata, and access tracking.
//!
//! ## Primitive Grounding
//!
//! `Engram` is T2-C (π + κ + μ + ν):
//! - π Persistence: knowledge that survives across sessions
//! - κ Comparison: searchable and comparable
//! - μ Mapping: title→content mapping
//! - ν Frequency: access frequency tracking

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};

/// Unique identifier for an engram.
pub type EngramId = u64;

/// Source layer from which an engram was ingested.
///
/// ## Tier: T2-P (λ + ∂)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngramSource {
    /// MEMORY.md — loaded into system prompt every session.
    Memory,
    /// Brain artifacts — session-scoped deep reference.
    Brain,
    /// Lessons catalog — searchable by keyword/tag.
    Lesson,
    /// Implicit learning — key-value preferences.
    Implicit,
    /// Current session — ephemeral until persisted.
    Session,
}

/// A unit of persistent knowledge.
///
/// ## Tier: T2-C (π + κ + μ + ν)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engram {
    /// Unique identifier (assigned by store on insert).
    pub id: EngramId,
    /// Short descriptive title.
    pub title: String,
    /// Full knowledge content.
    pub content: String,
    /// Which memory layer this came from.
    pub source: EngramSource,
    /// Classification tags.
    pub tags: Vec<String>,
    /// When this knowledge was created.
    pub created_at: DateTime,
    /// When this knowledge was last accessed.
    pub last_accessed: DateTime,
    /// How many times this has been accessed.
    pub access_count: u64,
    /// Current relevance score (decays over time).
    pub relevance: f64,
}

impl Engram {
    /// Create a new engram with default temporal metadata.
    #[must_use]
    pub fn new(
        id: EngramId,
        title: impl Into<String>,
        content: impl Into<String>,
        source: EngramSource,
    ) -> Self {
        let now = DateTime::now();
        Self {
            id,
            title: title.into(),
            content: content.into(),
            source,
            tags: Vec::new(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            relevance: 1.0,
        }
    }

    /// Add classification tags.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Record an access event (updates last_accessed and access_count).
    pub fn record_access(&mut self) {
        self.last_accessed = DateTime::now();
        self.access_count += 1;
    }

    /// All searchable text combined (title + content + tags).
    #[must_use]
    pub fn searchable_text(&self) -> String {
        format!("{} {} {}", self.title, self.content, self.tags.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engram_new() {
        let e = Engram::new(0, "Test", "Content", EngramSource::Memory);
        assert_eq!(e.title, "Test");
        assert_eq!(e.content, "Content");
        assert_eq!(e.source, EngramSource::Memory);
        assert_eq!(e.access_count, 0);
        assert!((e.relevance - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engram_with_tags() {
        let e = Engram::new(0, "Test", "Content", EngramSource::Lesson)
            .with_tags(vec!["rust".to_string(), "model-checking".to_string()]);
        assert_eq!(e.tags.len(), 2);
        assert_eq!(e.tags[0], "rust");
    }

    #[test]
    fn test_record_access() {
        let mut e = Engram::new(0, "Test", "Content", EngramSource::Memory);
        let before = e.last_accessed;
        e.record_access();
        assert_eq!(e.access_count, 1);
        assert!(e.last_accessed >= before);
    }

    #[test]
    fn test_searchable_text() {
        let e = Engram::new(0, "Title", "Body text", EngramSource::Memory)
            .with_tags(vec!["tag1".to_string()]);
        let text = e.searchable_text();
        assert!(text.contains("Title"));
        assert!(text.contains("Body text"));
        assert!(text.contains("tag1"));
    }

    #[test]
    fn test_source_equality() {
        assert_eq!(EngramSource::Memory, EngramSource::Memory);
        assert_ne!(EngramSource::Memory, EngramSource::Brain);
    }
}
