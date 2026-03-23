// Copyright (c) 2026 Matthew Campion, PharmD; NexVigilant
// All Rights Reserved. See LICENSE file for details.

//! # Lex Primitiva Grounding for Engram Types
//!
//! Maps each engram type to its T1 primitive composition.

use nexcore_lex_primitiva::grounding::GroundsTo;
use nexcore_lex_primitiva::primitiva::{LexPrimitiva, PrimitiveComposition};
use nexcore_lex_primitiva::state_mode::StateMode;

use crate::consolidate::DuplicatePair;
use crate::decay::DecayConfig;
use crate::engram::Engram;
use crate::search::SearchIndex;
use crate::store::EngramStore;

// ── Engram: T2-C (π + κ + μ + ν) ───────────────────────────

impl GroundsTo for Engram {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Persistence, // π — knowledge that persists (dominant)
            LexPrimitiva::Comparison,  // κ — searchable/comparable
            LexPrimitiva::Mapping,     // μ — title→content mapping
            LexPrimitiva::Frequency,   // ν — access frequency tracking
        ])
        .with_dominant(LexPrimitiva::Persistence, 0.90)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Accumulated) // knowledge accumulates, never lost
    }
}

// ── EngramStore: T3 (π + κ + μ + λ + Σ + ν) ────────────────

impl GroundsTo for EngramStore {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Persistence, // π — durable storage (dominant)
            LexPrimitiva::Comparison,  // κ — search/ranking
            LexPrimitiva::Mapping,     // μ — id→engram mapping
            LexPrimitiva::Location,    // λ — cross-layer location
            LexPrimitiva::Sum,         // Σ — aggregation across sources
            LexPrimitiva::Frequency,   // ν — access tracking
        ])
        .with_dominant(LexPrimitiva::Persistence, 0.88)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Accumulated)
    }
}

// ── SearchIndex: T2-C (κ + μ + Σ + N) ──────────────────────

impl GroundsTo for SearchIndex {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Comparison, // κ — query matching (dominant)
            LexPrimitiva::Mapping,    // μ — term→postings mapping
            LexPrimitiva::Sum,        // Σ — TF-IDF aggregation
            LexPrimitiva::Quantity,   // N — frequency counts
        ])
        .with_dominant(LexPrimitiva::Comparison, 0.92)
    }
}

// ── DecayConfig: T2-P (ν + ∝) ──────────────────────────────

impl GroundsTo for DecayConfig {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Frequency,       // ν — temporal frequency (dominant)
            LexPrimitiva::Irreversibility, // ∝ — aging is one-way
        ])
        .with_dominant(LexPrimitiva::Frequency, 0.90)
    }
}

// ── DuplicatePair: T2-P (κ + N) ────────────────────────────

impl GroundsTo for DuplicatePair {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Comparison, // κ — similarity comparison (dominant)
            LexPrimitiva::Quantity,   // N — similarity score
        ])
        .with_dominant(LexPrimitiva::Comparison, 0.95)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engram_grounding() {
        let comp = Engram::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Persistence));
        assert_eq!(comp.primitives.len(), 4);
    }

    #[test]
    fn test_store_grounding() {
        let comp = EngramStore::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Persistence));
        assert_eq!(comp.primitives.len(), 6);
    }

    #[test]
    fn test_search_index_grounding() {
        let comp = SearchIndex::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Comparison));
        assert_eq!(comp.primitives.len(), 4);
    }

    #[test]
    fn test_decay_config_grounding() {
        let comp = DecayConfig::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Frequency));
        assert_eq!(comp.primitives.len(), 2);
    }

    #[test]
    fn test_duplicate_pair_grounding() {
        let comp = DuplicatePair::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Comparison));
        assert_eq!(comp.primitives.len(), 2);
    }

    #[test]
    fn test_engram_state_mode() {
        assert_eq!(Engram::state_mode(), Some(StateMode::Accumulated));
    }

    #[test]
    fn test_store_state_mode() {
        assert_eq!(EngramStore::state_mode(), Some(StateMode::Accumulated));
    }
}
