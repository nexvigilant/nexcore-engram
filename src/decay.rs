// Copyright (c) 2026 Matthew Campion, PharmD; NexVigilant
// All Rights Reserved. See LICENSE file for details.

//! # Temporal Decay Engine
//!
//! Knowledge loses relevance over time unless reinforced by access.
//! Uses exponential decay with access-frequency resistance.
//!
//! ## Primitive Grounding
//!
//! `DecayConfig` is T2-P (ν + ∝):
//! - ν Frequency: temporal frequency drives decay rate
//! - ∝ Irreversibility: aging is one-directional

use nexcore_chrono::DateTime;

use crate::engram::Engram;

/// Configuration for knowledge decay.
///
/// ## Tier: T2-P (ν + ∝)
#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// Half-life in days — knowledge loses half its relevance after this period.
    pub half_life_days: f64,
    /// Minimum relevance threshold before an engram is considered stale.
    pub stale_threshold: f64,
    /// Weight of access frequency in resisting decay.
    pub access_weight: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            half_life_days: 14.0,
            stale_threshold: 0.1,
            access_weight: 0.1,
        }
    }
}

/// Compute decayed relevance score for an engram.
///
/// Formula: `base_decay + (access_boost × recency_boost)`, clamped to [0, 1].
///
/// - `base_decay = 0.5^(age / half_life)` — exponential decay
/// - `access_boost = ln(1 + access_count) × access_weight` — frequency resistance
/// - `recency_boost = 0.5^(since_last_access / (2 × half_life))` — recent access bonus
#[must_use]
pub fn decay_score(engram: &Engram, now: DateTime, config: &DecayConfig) -> f64 {
    let age_days = (now - engram.created_at).num_seconds() as f64 / 86400.0;

    // Exponential decay based on age
    let base_decay = 0.5_f64.powf(age_days / config.half_life_days);

    // Access frequency boosts resistance to decay
    let access_boost = (engram.access_count as f64).ln_1p() * config.access_weight;

    // Recent access provides recency bonus
    let since_last_access = (now - engram.last_accessed).num_seconds() as f64 / 86400.0;
    let recency_boost = 0.5_f64.powf(since_last_access / (config.half_life_days * 2.0));

    // Combined score, clamped to [0, 1]
    (base_decay + access_boost * recency_boost).clamp(0.0, 1.0)
}

/// Check if an engram is stale (below the configured threshold).
#[must_use]
pub fn is_stale(engram: &Engram, now: DateTime, config: &DecayConfig) -> bool {
    decay_score(engram, now, config) < config.stale_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::EngramSource;
    use nexcore_chrono::Duration;

    #[test]
    fn test_fresh_engram_high_relevance() {
        let e = Engram::new(1, "Fresh", "Content", EngramSource::Memory);
        let score = decay_score(&e, DateTime::now(), &DecayConfig::default());
        assert!(
            score > 0.9,
            "Fresh engram should have high relevance: {score}"
        );
    }

    #[test]
    fn test_old_engram_decays() {
        let mut e = Engram::new(1, "Old", "Content", EngramSource::Memory);
        e.created_at = DateTime::now() - Duration::days(30);
        e.last_accessed = e.created_at;
        let score = decay_score(&e, DateTime::now(), &DecayConfig::default());
        assert!(
            score < 0.5,
            "30-day-old engram should have decayed: {score}"
        );
    }

    #[test]
    fn test_accessed_engram_resists_decay() {
        let mut accessed = Engram::new(1, "Accessed", "Content", EngramSource::Memory);
        accessed.created_at = DateTime::now() - Duration::days(30);
        accessed.last_accessed = DateTime::now();
        accessed.access_count = 10;

        let mut unaccessed = accessed.clone();
        unaccessed.access_count = 0;
        unaccessed.last_accessed = unaccessed.created_at;

        let config = DecayConfig::default();
        let accessed_score = decay_score(&accessed, DateTime::now(), &config);
        let unaccessed_score = decay_score(&unaccessed, DateTime::now(), &config);

        assert!(
            accessed_score > unaccessed_score,
            "Accessed engram should resist decay: {accessed_score} > {unaccessed_score}"
        );
    }

    #[test]
    fn test_stale_detection() {
        let mut e = Engram::new(1, "Ancient", "Content", EngramSource::Memory);
        e.created_at = DateTime::now() - Duration::days(365);
        e.last_accessed = e.created_at;
        assert!(is_stale(&e, DateTime::now(), &DecayConfig::default()));
    }

    #[test]
    fn test_fresh_not_stale() {
        let e = Engram::new(1, "Fresh", "Content", EngramSource::Memory);
        assert!(!is_stale(&e, DateTime::now(), &DecayConfig::default()));
    }

    #[test]
    fn test_decay_score_clamped() {
        let mut e = Engram::new(1, "Boosted", "Content", EngramSource::Memory);
        e.access_count = 1_000_000; // extreme access count
        let score = decay_score(&e, DateTime::now(), &DecayConfig::default());
        assert!(score <= 1.0, "Score should be clamped to 1.0: {score}");
    }
}
