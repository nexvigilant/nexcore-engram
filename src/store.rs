// Copyright (c) 2026 Matthew Campion, PharmD; NexVigilant
// All Rights Reserved. See LICENSE file for details.

//! # Engram Store — Unified Knowledge Repository
//!
//! In-memory store with inverted index search, temporal decay, and disk persistence.
//! Consolidates all memory layers into a single queryable interface.
//!
//! ## Primitive Grounding
//!
//! `EngramStore` is T3 (π + κ + μ + λ + Σ + ν):
//! - π Persistence: durable storage with save/load
//! - κ Comparison: TF-IDF search and ranking
//! - μ Mapping: id→engram mapping
//! - λ Location: cross-layer location awareness
//! - Σ Sum: aggregation across sources
//! - ν Frequency: access tracking and decay

use std::collections::HashMap;
use std::path::Path;

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};

use crate::consolidate::{DuplicatePair, find_duplicates};
use crate::decay::{DecayConfig, decay_score};
use crate::engram::{Engram, EngramId, EngramSource};
use crate::search::{SearchIndex, SearchResult};

/// Unified knowledge store with search and decay.
///
/// ## Tier: T3 (π + κ + μ + λ + Σ + ν)
pub struct EngramStore {
    engrams: HashMap<EngramId, Engram>,
    index: SearchIndex,
    next_id: EngramId,
    decay_config: DecayConfig,
}

/// Serializable snapshot of the store for disk persistence.
#[derive(Serialize, Deserialize)]
struct StoreSnapshot {
    engrams: Vec<Engram>,
    next_id: EngramId,
}

impl EngramStore {
    /// Create an empty store with default decay configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            engrams: HashMap::new(),
            index: SearchIndex::new(),
            next_id: 1,
            decay_config: DecayConfig::default(),
        }
    }

    /// Set custom decay configuration.
    #[must_use]
    pub fn with_decay_config(mut self, config: DecayConfig) -> Self {
        self.decay_config = config;
        self
    }

    /// Insert an engram, returning its assigned ID.
    pub fn insert(&mut self, mut engram: Engram) -> EngramId {
        let id = self.next_id;
        self.next_id += 1;
        engram.id = id;

        let text = engram.searchable_text();
        self.index.index(id, &text);
        self.engrams.insert(id, engram);
        id
    }

    /// Get an engram by ID (records an access event).
    pub fn get(&mut self, id: EngramId) -> Option<&Engram> {
        if let Some(engram) = self.engrams.get_mut(&id) {
            engram.record_access();
        }
        self.engrams.get(&id)
    }

    /// Get an engram by ID without recording access.
    #[must_use]
    pub fn peek(&self, id: EngramId) -> Option<&Engram> {
        self.engrams.get(&id)
    }

    /// Remove an engram by ID.
    pub fn remove(&mut self, id: EngramId) -> Option<Engram> {
        self.index.remove(id);
        self.engrams.remove(&id)
    }

    /// Search across all engrams with TF-IDF scoring.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        self.index.search(query)
    }

    /// Search with decay-adjusted scoring (recent knowledge ranks higher).
    #[must_use]
    pub fn search_with_decay(&self, query: &str) -> Vec<SearchResult> {
        let now = DateTime::now();
        let mut results = self.index.search(query);

        for result in &mut results {
            if let Some(engram) = self.engrams.get(&result.id) {
                let decay = decay_score(engram, now, &self.decay_config);
                result.score *= decay;
            }
        }

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results
    }

    /// Find near-duplicate engrams above the similarity threshold.
    #[must_use]
    pub fn find_duplicates(&self, threshold: f64) -> Vec<DuplicatePair> {
        let engrams: Vec<&Engram> = self.engrams.values().collect();
        find_duplicates(&engrams, threshold)
    }

    /// Get all stale engrams (below decay threshold).
    #[must_use]
    pub fn stale_engrams(&self) -> Vec<EngramId> {
        let now = DateTime::now();
        self.engrams
            .values()
            .filter(|e| crate::decay::is_stale(e, now, &self.decay_config))
            .map(|e| e.id)
            .collect()
    }

    /// Number of engrams in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.engrams.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.engrams.is_empty()
    }

    /// Get engrams from a specific source layer.
    #[must_use]
    pub fn by_source(&self, source: &EngramSource) -> Vec<&Engram> {
        self.engrams
            .values()
            .filter(|e| &e.source == source)
            .collect()
    }

    /// Save store to disk as JSON.
    pub fn save(&self, path: &Path) -> Result<(), EngramError> {
        let snapshot = StoreSnapshot {
            engrams: self.engrams.values().cloned().collect(),
            next_id: self.next_id,
        };
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| EngramError::Serialization(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| EngramError::Io(e.to_string()))?;
        Ok(())
    }

    /// Load store from disk.
    pub fn load(path: &Path) -> Result<Self, EngramError> {
        let json = std::fs::read_to_string(path).map_err(|e| EngramError::Io(e.to_string()))?;
        let snapshot: StoreSnapshot =
            serde_json::from_str(&json).map_err(|e| EngramError::Serialization(e.to_string()))?;

        let mut store = Self::new();
        store.next_id = snapshot.next_id;
        for engram in snapshot.engrams {
            let text = engram.searchable_text();
            let id = engram.id;
            store.index.index(id, &text);
            store.engrams.insert(id, engram);
        }
        Ok(store)
    }

    /// Summary statistics across all layers.
    #[must_use]
    pub fn stats(&self) -> StoreStats {
        let total = self.engrams.len();
        let stale = self.stale_engrams().len();
        StoreStats {
            total,
            active: total - stale,
            stale,
            memory_count: self.by_source(&EngramSource::Memory).len(),
            brain_count: self.by_source(&EngramSource::Brain).len(),
            lesson_count: self.by_source(&EngramSource::Lesson).len(),
            implicit_count: self.by_source(&EngramSource::Implicit).len(),
            session_count: self.by_source(&EngramSource::Session).len(),
        }
    }
}

impl Default for EngramStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Store statistics by layer.
#[derive(Debug, Clone)]
pub struct StoreStats {
    /// Total engrams.
    pub total: usize,
    /// Active (non-stale) engrams.
    pub active: usize,
    /// Stale engrams (below decay threshold).
    pub stale: usize,
    /// Engrams from MEMORY.md.
    pub memory_count: usize,
    /// Engrams from Brain artifacts.
    pub brain_count: usize,
    /// Engrams from Lessons catalog.
    pub lesson_count: usize,
    /// Engrams from Implicit learning.
    pub implicit_count: usize,
    /// Engrams from current session.
    pub session_count: usize,
}

/// Errors from engram operations.
#[derive(Debug, Clone)]
pub enum EngramError {
    /// File I/O error.
    Io(String),
    /// Serialization/deserialization error.
    Serialization(String),
    /// Engram not found.
    NotFound(EngramId),
    /// Error during ingestion.
    IngestError(String),
}

impl std::fmt::Display for EngramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Serialization(e) => write!(f, "Serialization error: {e}"),
            Self::NotFound(id) => write!(f, "Engram not found: {id}"),
            Self::IngestError(e) => write!(f, "Ingest error: {e}"),
        }
    }
}

impl std::error::Error for EngramError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut store = EngramStore::new();
        let e = Engram::new(0, "Test", "Content here", EngramSource::Memory);
        let id = store.insert(e);

        let retrieved = store.get(id);
        assert!(retrieved.is_some());
        if let Some(e) = retrieved {
            assert_eq!(e.title, "Test");
            assert_eq!(e.access_count, 1); // get records access
        }
    }

    #[test]
    fn test_peek_no_access() {
        let mut store = EngramStore::new();
        let id = store.insert(Engram::new(0, "Test", "Content", EngramSource::Memory));

        let peeked = store.peek(id);
        assert!(peeked.is_some());
        if let Some(e) = peeked {
            assert_eq!(e.access_count, 0); // peek does not record
        }
    }

    #[test]
    fn test_search() {
        let mut store = EngramStore::new();
        store.insert(Engram::new(
            0,
            "Rust",
            "Systems programming language",
            EngramSource::Memory,
        ));
        store.insert(Engram::new(
            0,
            "Python",
            "Scripting language",
            EngramSource::Memory,
        ));

        let results = store.search("rust systems");
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_remove() {
        let mut store = EngramStore::new();
        let id = store.insert(Engram::new(0, "Test", "Remove me", EngramSource::Session));
        assert_eq!(store.len(), 1);
        store.remove(id);
        assert_eq!(store.len(), 0);
        assert!(store.peek(id).is_none());
    }

    #[test]
    fn test_by_source() {
        let mut store = EngramStore::new();
        store.insert(Engram::new(0, "A", "Memory content", EngramSource::Memory));
        store.insert(Engram::new(0, "B", "Brain content", EngramSource::Brain));
        store.insert(Engram::new(0, "C", "Memory again", EngramSource::Memory));

        assert_eq!(store.by_source(&EngramSource::Memory).len(), 2);
        assert_eq!(store.by_source(&EngramSource::Brain).len(), 1);
        assert_eq!(store.by_source(&EngramSource::Lesson).len(), 0);
    }

    #[test]
    fn test_save_and_load() {
        let mut store = EngramStore::new();
        store.insert(Engram::new(
            0,
            "Persistent",
            "Knowledge one",
            EngramSource::Memory,
        ));
        store.insert(Engram::new(
            0,
            "Also Persistent",
            "Knowledge two",
            EngramSource::Brain,
        ));

        let dir = tempfile::tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("engrams.json");

        let save_result = store.save(&path);
        assert!(save_result.is_ok());

        let loaded = EngramStore::load(&path);
        assert!(loaded.is_ok());
        if let Ok(loaded_store) = loaded {
            assert_eq!(loaded_store.len(), 2);
            // Search should work after reload (index rebuilt)
            let results = loaded_store.search("persistent");
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_stats() {
        let mut store = EngramStore::new();
        store.insert(Engram::new(0, "A", "Content", EngramSource::Memory));
        store.insert(Engram::new(0, "B", "Content", EngramSource::Brain));
        store.insert(Engram::new(0, "C", "Content", EngramSource::Lesson));

        let stats = store.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.memory_count, 1);
        assert_eq!(stats.brain_count, 1);
        assert_eq!(stats.lesson_count, 1);
    }

    #[test]
    fn test_default() {
        let store = EngramStore::default();
        assert!(store.is_empty());
    }
}
