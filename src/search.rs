// Copyright (c) 2026 Matthew Campion, PharmD; NexVigilant
// All Rights Reserved. See LICENSE file for details.

//! # Search Engine — Inverted Index with TF-IDF
//!
//! Full-text search over engrams using an inverted index with TF-IDF scoring.
//!
//! ## Primitive Grounding
//!
//! `SearchIndex` is T2-C (κ + μ + Σ + N):
//! - κ Comparison: query matching (dominant)
//! - μ Mapping: term→postings mapping
//! - Σ Sum: TF-IDF score aggregation
//! - N Quantity: frequency counts

use std::collections::HashMap;

use crate::engram::EngramId;

/// Search result with relevance score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// ID of the matching engram.
    pub id: EngramId,
    /// TF-IDF relevance score.
    pub score: f64,
}

/// Inverted index for full-text search with TF-IDF scoring.
///
/// ## Tier: T2-C (κ + μ + Σ + N)
pub struct SearchIndex {
    /// term → Vec<(engram_id, term_frequency)>
    postings: HashMap<String, Vec<(EngramId, f64)>>,
    /// engram_id → document length (in terms)
    doc_lengths: HashMap<EngramId, usize>,
    /// Total number of indexed documents.
    doc_count: usize,
}

impl SearchIndex {
    /// Create an empty search index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Tokenize text into lowercase terms (minimum 2 chars).
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| s.len() >= 2)
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Index a document's text under the given engram ID.
    pub fn index(&mut self, id: EngramId, text: &str) {
        let tokens = Self::tokenize(text);
        let doc_len = tokens.len();
        if doc_len == 0 {
            return;
        }

        // Count term frequencies
        let mut tf_counts: HashMap<&str, usize> = HashMap::new();
        for token in &tokens {
            *tf_counts.entry(token.as_str()).or_insert(0) += 1;
        }

        // Add to postings list
        for (term, count) in tf_counts {
            let tf = count as f64 / doc_len as f64;
            self.postings
                .entry(term.to_string())
                .or_default()
                .push((id, tf));
        }

        self.doc_lengths.insert(id, doc_len);
        self.doc_count += 1;
    }

    /// Remove a document from the index.
    pub fn remove(&mut self, id: EngramId) {
        for postings in self.postings.values_mut() {
            postings.retain(|&(eid, _)| eid != id);
        }
        // Clean up empty posting lists
        self.postings.retain(|_, v| !v.is_empty());
        if self.doc_lengths.remove(&id).is_some() {
            self.doc_count = self.doc_count.saturating_sub(1);
        }
    }

    /// Search with TF-IDF scoring. Returns results sorted by descending score.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let terms = Self::tokenize(query);
        if terms.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<EngramId, f64> = HashMap::new();

        for term in &terms {
            if let Some(postings) = self.postings.get(term.as_str()) {
                let df = postings.len() as f64;
                let idf = ((self.doc_count as f64) / df).ln_1p();
                for &(id, tf) in postings {
                    *scores.entry(id).or_insert(0.0) += tf * idf;
                }
            }
        }

        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect();

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.doc_count
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = SearchIndex::tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // "a" filtered out (< 2 chars), "is" kept (== 2 chars)
        assert!(tokens.contains(&"is".to_string()));
    }

    #[test]
    fn test_tokenize_underscore() {
        let tokens = SearchIndex::tokenize("model_checking is_good");
        assert!(tokens.contains(&"model_checking".to_string()));
        assert!(tokens.contains(&"is_good".to_string()));
    }

    #[test]
    fn test_index_and_search() {
        let mut idx = SearchIndex::new();
        idx.index(1, "Rust programming language");
        idx.index(2, "Python programming language");
        idx.index(3, "Rust is fast and safe");

        let results = idx.search("rust");
        assert!(!results.is_empty());
        let ids: Vec<EngramId> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_multi_term_search_ranks_correctly() {
        let mut idx = SearchIndex::new();
        idx.index(1, "model checking temporal logic");
        idx.index(2, "signal detection algorithms");
        idx.index(3, "temporal signal processing pipeline");

        let results = idx.search("temporal signal");
        assert!(!results.is_empty());
        // Engram 3 has both terms, should rank highest
        assert_eq!(results[0].id, 3);
    }

    #[test]
    fn test_remove_from_index() {
        let mut idx = SearchIndex::new();
        idx.index(1, "hello world");
        idx.index(2, "hello rust");
        assert_eq!(idx.len(), 2);

        idx.remove(1);
        assert_eq!(idx.len(), 1);

        let results = idx.search("hello");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 2);
    }

    #[test]
    fn test_empty_search() {
        let idx = SearchIndex::new();
        let results = idx.search("anything");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_no_match() {
        let mut idx = SearchIndex::new();
        idx.index(1, "rust programming");
        let results = idx.search("python");
        assert!(results.is_empty());
    }

    #[test]
    fn test_idf_weighting() {
        let mut idx = SearchIndex::new();
        // "the" appears in all docs (low IDF), "kripke" in one (high IDF)
        idx.index(1, "the kripke structure model");
        idx.index(2, "the model checking algorithm");
        idx.index(3, "the temporal logic formula");

        let results = idx.search("kripke");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
        assert!(results[0].score > 0.0);
    }
}
