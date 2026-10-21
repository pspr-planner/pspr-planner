use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use crate::parsing::json::{Program, decode_program};

/// Configuration for invoking the external HDDL parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserConfig {
    pub executable: PathBuf,
}

/// Locate `hddl_analyzer`. An explicit path wins over `$HDDL_PARSER`, which
/// wins over `PATH`.
pub fn resolve_parser_config(explicit: Option<&Path>) -> Result<ParserConfig, String> {
    let from_env = env::var_os("HDDL_PARSER").filter(|value| !value.is_empty());
    let requested = match explicit {
        Some(path) => Some(path.to_path_buf()),
        None => from_env.map(PathBuf::from),
    };

    let candidate = match requested {
        Some(path) => Some(path),
        None => find_executable(Path::new("hddl_analyzer")),
    };

    match candidate {
        None => Err("cannot find the HDDL-Parser executable (hddl_analyzer). \
             Build it from https://github.com/koala-planner/HDDL-Parser \
             and pass --parser PATH or set HDDL_PARSER."
            .to_string()),
        Some(path) => find_executable(&path)
            .map(|executable| ParserConfig { executable })
            .ok_or_else(|| format!("HDDL-Parser executable is not runnable: {}", path.display())),
    }
}

/// Whether `path` names a JSON export (case-insensitive `.json` extension).
pub fn is_json_file(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

/// Parse a domain, optionally paired with a problem. For HDDL inputs this runs
/// the external parser once and decodes the resulting JSON document.
pub fn parse_program(
    config: &ParserConfig,
    domain_path: &Path,
    problem_path: Option<&Path>,
) -> Result<Program, String> {
    if is_json_file(domain_path) {
        return match problem_path {
            None => read_program_json(domain_path),
            Some(_) => Err("a .json input already contains its problem; \
                 pass an .hddl domain to pair it with problem files"
                .to_string()),
        };
    }

    let mut arguments: Vec<OsString> = vec!["convert".into(), domain_path.into()];
    if let Some(problem) = problem_path {
        arguments.push("-p".into());
        arguments.push(problem.into());
    }
    arguments.push("--to".into());
    arguments.push("json".into());

    let stdout = run_external(config, &arguments)?;
    let json = extract_json_object(&stdout).ok_or_else(|| {
        format!(
            "the HDDL parser printed no JSON for {}",
            describe_input(domain_path, problem_path)
        )
    })?;

    decode_program(json).map_err(|err| {
        format!(
            "failed to decode the HDDL parser JSON for {}: {}",
            describe_input(domain_path, problem_path),
            err
        )
    })
}

/// Read a complete program that the HDDL parser already exported as JSON.
/// Files produced with `convert -o program.json` are pure JSON, but accepting
/// the `[Ok]` status line as well makes a redirected stdout usable directly.
pub fn read_program_json(path: &Path) -> Result<Program, String> {
    let bytes = fs::read(path).map_err(|err| format!("cannot read {}: {}", path.display(), err))?;
    let json = extract_json_object(&bytes)
        .ok_or_else(|| format!("no JSON object found in {}", path.display()))?;
    decode_program(json).map_err(|err| format!("failed to decode {}: {}", path.display(), err))
}

fn describe_input(domain_path: &Path, problem_path: Option<&Path>) -> String {
    match problem_path {
        None => domain_path.display().to_string(),
        Some(problem) => format!("{} + {}", domain_path.display(), problem.display()),
    }
}

/// Run the parser, returning stdout on success. `convert` reports parse errors
/// on stderr (and, historically, still exits 0), so we also check that the
/// output actually contains JSON.
fn run_external(config: &ParserConfig, arguments: &[OsString]) -> Result<Vec<u8>, String> {
    let output = Command::new(&config.executable)
        .args(arguments)
        .output()
        .map_err(|err| format!("could not run {}: {}", config.executable.display(), err))?;

    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        return Err(format!(
            "the HDDL parser exited with status {}{}",
            status_description(&output.status),
            append_stderr(&stderr)
        ));
    }

    if extract_json_object(&output.stdout).is_none() {
        return Err(format!(
            "the HDDL parser did not produce JSON{}",
            append_stderr(&stderr)
        ));
    }

    Ok(output.stdout)
}

fn status_description(status: &ExitStatus) -> String {
    match status.code() {
        Some(code) => code.to_string(),
        None => status.to_string(),
    }
}

/// The `convert` command prefixes the JSON with a coloured `[Ok]` line when
/// writing to stdout; everything before the first `{` is discarded.
fn extract_json_object(bytes: &[u8]) -> Option<&[u8]> {
    bytes
        .iter()
        .position(|&byte| byte == b'{')
        .map(|start| &bytes[start..])
}

fn append_stderr(stderr: &str) -> String {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        format!(": {}", trimmed)
    }
}

/// Resolve `name` the way `findExecutable` does: a path with a directory
/// component is used as-is (if executable), otherwise `PATH` is searched.
fn find_executable(name: &Path) -> Option<PathBuf> {
    let has_directory = name.is_absolute() || name.components().count() > 1;
    if has_directory {
        return is_executable(name).then(|| name.to_path_buf());
    }

    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| is_executable(candidate))
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    match fs::metadata(path) {
        Ok(metadata) => metadata.is_file() && metadata.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    match fs::metadata(path) {
        Ok(metadata) => metadata.is_file(),
        Err(_) => false,
    }
}
