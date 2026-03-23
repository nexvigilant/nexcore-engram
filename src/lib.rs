// Copyright (c) 2026 Matthew Campion, PharmD; NexVigilant
// All Rights Reserved. See LICENSE file for details.

//! # nexcore-engram
//!
//! Unified knowledge daemon — persistent memory with semantic search and temporal decay.
//!
//! Consolidates four memory layers (MEMORY.md, Brain artifacts, Lessons, Implicit)
//! into a single searchable store with TF-IDF ranking and time-based decay.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           EngramStore (π)                │
//! │  ┌──────────┐  ┌───────────┐            │
//! │  │  Engrams  │  │ SearchIdx │ TF-IDF(κ) │
//! │  │ (HashMap) │  │ (InvIdx)  │            │
//! │  └──────────┘  └───────────┘            │
//! │  ┌──────────┐  ┌───────────┐            │
//! │  │  Decay    │  │Consolidate│            │
//! │  │  (ν + ∝)  │  │  (κ + Σ)  │            │
//! │  └──────────┘  └───────────┘            │
//! └────────────────────┬────────────────────┘
//!                      │ ingest
//!       ┌──────┬───────┼───────┬──────┐
//!       │      │       │       │      │
//!     Memory  Brain  Lessons Implicit Session
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use nexcore_engram::prelude::*;
//!
//! let mut store = EngramStore::new();
//!
//! // Insert knowledge from different layers
//! store.insert(Engram::new(0, "LTL Model Checking",
//!     "Evaluate formula at position 0 on maximal-depth paths",
//!     EngramSource::Lesson));
//!
//! // Search across all layers
//! let results = store.search("model checking");
//! assert!(!results.is_empty());
//! ```
//!
//! ## Primitive Grounding
//!
//! | Type | Tier | Dominant | Primitives |
//! |------|------|----------|------------|
//! | `Engram` | T2-C | π | π κ μ ν |
//! | `EngramStore` | T3 | π | π κ μ λ Σ ν |
//! | `SearchIndex` | T2-C | κ | κ μ Σ N |
//! | `DecayConfig` | T2-P | ν | ν ∝ |
//! | `DuplicatePair` | T2-P | κ | κ N |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod consolidate;
pub mod decay;
pub mod engram;
pub mod grounding;
pub mod ingest;
pub mod search;
pub mod store;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::consolidate::{DuplicatePair, find_duplicates};
    pub use crate::decay::{DecayConfig, decay_score, is_stale};
    pub use crate::engram::{Engram, EngramId, EngramSource};
    pub use crate::search::{SearchIndex, SearchResult};
    pub use crate::store::{EngramError, EngramStore, StoreStats};
}

#[cfg(test)]
mod integration_tests {
    use crate::prelude::*;

    #[test]
    fn test_cross_layer_search() {
        let mut store = EngramStore::new();

        // Insert knowledge from different sources
        store.insert(
            Engram::new(
                0,
                "LTL Model Checking",
                "Evaluate formula at position 0 on maximal-depth paths",
                EngramSource::Lesson,
            )
            .with_tags(vec!["model-checking".to_string(), "LTL".to_string()]),
        );
        store.insert(Engram::new(
            0,
            "Kripke Structure",
            "M equals S S0 R L with successors and predecessors",
            EngramSource::Brain,
        ));
        store.insert(Engram::new(
            0,
            "Workspace Conventions",
            "Edition 2024 forbid unsafe deny unwrap",
            EngramSource::Memory,
        ));

        // Cross-layer search finds results from any source
        let results = store.search("model checking");
        assert!(!results.is_empty());

        // "kripke" finds brain artifact
        let results = store.search("kripke structure");
        assert!(!results.is_empty());
        let top = store.peek(results[0].id);
        assert!(top.is_some());
        if let Some(e) = top {
            assert_eq!(e.source, EngramSource::Brain);
        }
    }

    #[test]
    fn test_decay_affects_ranking() {
        let mut store = EngramStore::new();

        // Insert two engrams with same content but different ages
        let mut old = Engram::new(
            0,
            "Old Knowledge",
            "temporal logic verification algorithms",
            EngramSource::Lesson,
        );
        old.created_at = nexcore_chrono::DateTime::now() - nexcore_chrono::Duration::days(60);
        old.last_accessed = old.created_at;

        let fresh = Engram::new(
            0,
            "Fresh Knowledge",
            "temporal logic verification algorithms",
            EngramSource::Session,
        );

        let _old_id = store.insert(old);
        let fresh_id = store.insert(fresh);

        // With decay, fresh should rank higher
        let decayed_results = store.search_with_decay("temporal logic");
        assert_eq!(decayed_results.len(), 2);
        assert_eq!(decayed_results[0].id, fresh_id);
    }

    #[test]
    fn test_consolidation_detects_near_duplicates() {
        let mut store = EngramStore::new();

        store.insert(Engram::new(
            0,
            "LTL Evaluation",
            "Evaluate LTL formula at position zero on complete paths not at each extension",
            EngramSource::Memory,
        ));
        store.insert(Engram::new(
            0,
            "LTL Evaluation Position",
            "Evaluate LTL formula at position zero on maximal paths not at each step",
            EngramSource::Lesson,
        ));
        store.insert(Engram::new(
            0,
            "Rust Closures",
            "Edition 2024 pattern matching closures binding references",
            EngramSource::Memory,
        ));

        let dupes = store.find_duplicates(0.3);
        assert!(
            !dupes.is_empty(),
            "Should detect near-duplicate LTL lessons"
        );
        // The two LTL lessons should be the most similar pair
        let pair = &dupes[0];
        assert!(
            pair.similarity > 0.3,
            "LTL lessons should have high similarity: {}",
            pair.similarity
        );
    }

    #[test]
    fn test_full_lifecycle_save_reload_search() {
        let mut store = EngramStore::new();
        store.insert(Engram::new(
            0,
            "Persistent Fact",
            "Model checkers verify temporal properties",
            EngramSource::Brain,
        ));

        let dir = tempfile::tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("lifecycle.json");

        // Save
        assert!(store.save(&path).is_ok());

        // Reload
        let loaded = EngramStore::load(&path);
        assert!(loaded.is_ok());
        if let Ok(reloaded) = loaded {
            // Search works after reload
            let results = reloaded.search("temporal properties");
            assert!(!results.is_empty());
            assert_eq!(results[0].id, 1);
        }
    }

    #[test]
    fn test_ingest_and_search_memory_md() {
        let dir = tempfile::tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("MEMORY.md");
        std::fs::write(
            &path,
            "# Memory\n\n## Model Checking\nCTL and LTL verification\n\n## Rust Patterns\nEdition 2024 conventions\n",
        )
        .ok();

        let mut store = EngramStore::new();
        let count = crate::ingest::ingest_memory_md(&mut store, &path);
        assert!(count.is_ok());
        assert_eq!(count.ok().unwrap_or(0), 2);

        // Cross-check: searching finds ingested content
        let results = store.search("CTL LTL verification");
        assert!(!results.is_empty());
    }
}
