// Copyright (c) 2026 Matthew Campion, PharmD; NexVigilant
// All Rights Reserved. See LICENSE file for details.

//! # Source Ingestion
//!
//! Ingest engrams from all four memory layers:
//! - MEMORY.md (sections split by `## ` headers)
//! - Brain artifacts (markdown files in a directory)
//! - Lessons (JSONL with title/content/tags)
//! - Implicit (JSON key-value preferences)
//!
//! ## Primitive Grounding
//!
//! Ingestion is T2-C (σ + ∃ + μ):
//! - σ Sequence: sequential file processing
//! - ∃ Existence: validating source existence
//! - μ Mapping: transforming file content to engrams

use std::path::Path;

use crate::engram::{Engram, EngramSource};
use crate::store::{EngramError, EngramStore};

/// Ingest engrams from MEMORY.md, splitting on `## ` section headers.
///
/// Each section becomes a separate engram with the header as title.
pub fn ingest_memory_md(store: &mut EngramStore, path: &Path) -> Result<usize, EngramError> {
    let content = std::fs::read_to_string(path).map_err(|e| EngramError::Io(e.to_string()))?;

    let mut count = 0;
    let mut current_section = String::new();
    let mut current_content = String::new();

    for line in content.lines() {
        if line.starts_with("## ") {
            // Save previous section
            if !current_section.is_empty() && !current_content.trim().is_empty() {
                let engram = Engram::new(
                    0,
                    &current_section,
                    current_content.trim(),
                    EngramSource::Memory,
                );
                store.insert(engram);
                count += 1;
            }
            current_section = line.trim_start_matches("## ").to_string();
            current_content = String::new();
        } else {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }

    // Save last section
    if !current_section.is_empty() && !current_content.trim().is_empty() {
        let engram = Engram::new(
            0,
            &current_section,
            current_content.trim(),
            EngramSource::Memory,
        );
        store.insert(engram);
        count += 1;
    }

    Ok(count)
}

/// Ingest engrams from a directory of brain artifact markdown files.
///
/// Each `.md` file becomes one engram with the filename as title.
pub fn ingest_brain_dir(store: &mut EngramStore, dir: &Path) -> Result<usize, EngramError> {
    if !dir.is_dir() {
        return Ok(0);
    }

    let mut count = 0;
    let entries = std::fs::read_dir(dir).map_err(|e| EngramError::Io(e.to_string()))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "md") {
            let content =
                std::fs::read_to_string(&path).map_err(|e| EngramError::Io(e.to_string()))?;
            let title = path
                .file_stem()
                .map_or("untitled".to_string(), |s| s.to_string_lossy().to_string());
            let engram = Engram::new(0, title, content, EngramSource::Brain);
            store.insert(engram);
            count += 1;
        }
    }

    Ok(count)
}

/// Ingest from a lessons JSONL file (one JSON object per line).
///
/// Expected format: `{"title": "...", "content": "...", "tags": [...]}`
pub fn ingest_lessons_jsonl(store: &mut EngramStore, path: &Path) -> Result<usize, EngramError> {
    let content = std::fs::read_to_string(path).map_err(|e| EngramError::Io(e.to_string()))?;

    let mut count = 0;

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            let title = value
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("untitled");
            let body = value.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let tags: Vec<String> = value
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let engram = Engram::new(0, title, body, EngramSource::Lesson).with_tags(tags);
            store.insert(engram);
            count += 1;
        }
    }

    Ok(count)
}

/// Ingest from an implicit learning JSON file (key-value object).
///
/// Each key becomes an engram title, value becomes content.
pub fn ingest_implicit_json(store: &mut EngramStore, path: &Path) -> Result<usize, EngramError> {
    let content = std::fs::read_to_string(path).map_err(|e| EngramError::Io(e.to_string()))?;

    let value: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| EngramError::Serialization(e.to_string()))?;

    let mut count = 0;

    if let Some(obj) = value.as_object() {
        for (key, val) in obj {
            let content_str = match val {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let engram = Engram::new(0, key, content_str, EngramSource::Implicit)
                .with_tags(vec!["implicit".to_string()]);
            store.insert(engram);
            count += 1;
        }
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_ingest_memory_md() {
        let dir = tempfile::tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("MEMORY.md");
        {
            let mut f = std::fs::File::create(&path)
                .ok()
                .unwrap_or_else(|| unreachable!());
            write!(
                f,
                "# Auto Memory\n\n## Section One\nContent for section one.\n\n## Section Two\nContent for section two.\nMore content.\n"
            )
            .ok();
        }

        let mut store = EngramStore::new();
        let result = ingest_memory_md(&mut store, &path);
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or(0), 2);
        assert_eq!(store.len(), 2);

        // Search should find content
        let results = store.search("section one");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ingest_brain_dir() {
        let dir = tempfile::tempdir().ok().unwrap_or_else(|| unreachable!());

        // Create two markdown files
        std::fs::write(
            dir.path().join("lesson1.md"),
            "Model checking lesson content",
        )
        .ok();
        std::fs::write(dir.path().join("lesson2.md"), "Kripke structure details").ok();
        std::fs::write(dir.path().join("not_md.txt"), "Should be ignored").ok();

        let mut store = EngramStore::new();
        let result = ingest_brain_dir(&mut store, dir.path());
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or(0), 2); // Only .md files
    }

    #[test]
    fn test_ingest_lessons_jsonl() {
        let dir = tempfile::tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("lessons.jsonl");
        std::fs::write(
            &path,
            r#"{"title":"LTL Lesson","content":"Evaluate at position 0","tags":["ltl","model-checking"]}
{"title":"Rust Lesson","content":"Edition 2024 patterns","tags":["rust"]}
"#,
        )
        .ok();

        let mut store = EngramStore::new();
        let result = ingest_lessons_jsonl(&mut store, &path);
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or(0), 2);

        // Verify tags were ingested
        let results = store.search("ltl");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ingest_implicit_json() {
        let dir = tempfile::tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("implicit.json");
        std::fs::write(
            &path,
            r#"{"preferred_style":"concise","rust_edition":"2024","test_pattern":"assert_ok"}"#,
        )
        .ok();

        let mut store = EngramStore::new();
        let result = ingest_implicit_json(&mut store, &path);
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or(0), 3);

        let results = store.search("rust_edition");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ingest_missing_file() {
        let mut store = EngramStore::new();
        let result = ingest_memory_md(&mut store, Path::new("/nonexistent/MEMORY.md"));
        assert!(result.is_err());
    }

    #[test]
    fn test_ingest_empty_brain_dir() {
        let mut store = EngramStore::new();
        let result = ingest_brain_dir(&mut store, Path::new("/nonexistent/dir"));
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or(99), 0);
    }
}
