use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// All `.hddl` files next to the domain, sorted, excluding the domain itself
/// and any file literally named `domain.hddl`.
pub fn discover_sibling_problems(domain_path: &Path) -> io::Result<Vec<PathBuf>> {
    let directory = domain_path.parent().unwrap_or_else(|| Path::new(""));
    let domain_file = domain_path.file_name();

    let mut problems = Vec::new();
    for file in list_hddl_files(directory)? {
        let name = file.file_name();
        if name == domain_file {
            continue;
        }
        let is_domain = name
            .map(|name| name.eq_ignore_ascii_case("domain.hddl"))
            .unwrap_or(false);
        if !is_domain {
            problems.push(file);
        }
    }
    Ok(problems)
}

/// A problem argument may be a single file or a directory of files.
pub fn expand_problem_arg(path: &Path) -> io::Result<Vec<PathBuf>> {
    if path.is_dir() {
        list_hddl_files(path)
    } else {
        Ok(vec![path.to_path_buf()])
    }
}

fn list_hddl_files(directory: &Path) -> io::Result<Vec<PathBuf>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("hddl"))
            .unwrap_or(false)
        {
            entries.push(path);
        }
    }
    entries.sort();
    Ok(entries)
}
