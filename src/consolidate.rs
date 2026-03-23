// Copyright (c) 2026 Matthew Campion, PharmD; NexVigilant
// All Rights Reserved. See LICENSE file for details.

//! # Knowledge Consolidation
//!
//! Detects near-duplicate engrams across memory layers using Jaccard similarity.
//! Enables deduplication when the same knowledge exists in multiple sources.
//!
//! ## Primitive Grounding
//!
//! `DuplicatePair` is T2-P (κ + N):
//! - κ Comparison: similarity comparison (dominant)
//! - N Quantity: similarity score

use std::collections::HashSet;

use crate::engram::{Engram, EngramId};

/// A pair of potentially duplicate engrams with their similarity score.
///
/// ## Tier: T2-P (κ + N)
#[derive(Debug, Clone)]
pub struct DuplicatePair {
    /// First engram ID.
    pub id_a: EngramId,
    /// Second engram ID.
    pub id_b: EngramId,
    /// Jaccard similarity score [0.0, 1.0].
    pub similarity: f64,
}

/// Compute Jaccard similarity between two texts based on word sets.
///
/// J(A, B) = |A ∩ B| / |A ∪ B|
#[must_use]
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let words_a: HashSet<&str> = a
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 2)
        .collect();
    let words_b: HashSet<&str> = b
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 2)
        .collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }

    let intersection = words_a.intersection(&words_b).count() as f64;
    let union = words_a.union(&words_b).count() as f64;

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Find duplicate pairs among engrams above the similarity threshold.
///
/// Returns pairs sorted by descending similarity.
#[must_use]
pub fn find_duplicates(engrams: &[&Engram], threshold: f64) -> Vec<DuplicatePair> {
    let mut pairs = Vec::new();

    for i in 0..engrams.len() {
        for j in (i + 1)..engrams.len() {
            let text_a = engrams[i].searchable_text();
            let text_b = engrams[j].searchable_text();
            let sim = jaccard_similarity(&text_a, &text_b);
            if sim >= threshold {
                pairs.push(DuplicatePair {
                    id_a: engrams[i].id,
                    id_b: engrams[j].id,
                    similarity: sim,
                });
            }
        }
    }

    pairs.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::EngramSource;

    #[test]
    fn test_jaccard_identical() {
        let sim = jaccard_similarity("hello world test", "hello world test");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let sim = jaccard_similarity("hello world", "foo bar baz");
        assert!(sim.abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let sim = jaccard_similarity("hello world test", "hello world different");
        // Intersection: {hello, world} = 2, Union: {hello, world, test, different} = 4
        assert!((sim - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_empty() {
        let sim = jaccard_similarity("", "");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_duplicates_detects_similar() {
        let e1 = Engram::new(
            1,
            "LTL model checking",
            "bounded path exploration temporal",
            EngramSource::Lesson,
        );
        let e2 = Engram::new(
            2,
            "LTL model checking",
            "bounded path enumeration temporal",
            EngramSource::Brain,
        );
        let e3 = Engram::new(
            3,
            "Rust closures",
            "pattern matching edition 2024",
            EngramSource::Memory,
        );

        let engrams = vec![&e1, &e2, &e3];
        let dupes = find_duplicates(&engrams, 0.3);
        assert!(!dupes.is_empty());
        // e1 and e2 are most similar
        assert_eq!(dupes[0].id_a, 1);
        assert_eq!(dupes[0].id_b, 2);
    }

    #[test]
    fn test_find_duplicates_respects_threshold() {
        let e1 = Engram::new(
            1,
            "Alpha",
            "completely different content here",
            EngramSource::Memory,
        );
        let e2 = Engram::new(
            2,
            "Beta",
            "nothing in common with the above",
            EngramSource::Brain,
        );

        let engrams = vec![&e1, &e2];
        let dupes = find_duplicates(&engrams, 0.8);
        assert!(dupes.is_empty(), "Dissimilar engrams should not be flagged");
    }
}
