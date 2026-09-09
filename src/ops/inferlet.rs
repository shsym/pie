use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand};
use serde::Deserialize;

use runtime::inferlet::program::{Manifest, ProgramName, Repository};

use crate::ui::{self, Align, Answer, Mark, Palette, Row, Table};

#[derive(Subcommand, Debug)]
pub enum InferletCmd {
    List,

    Info(InfoArgs),

    Download(TargetArgs),

    Remove(TargetArgs),
}

#[derive(Args, Debug)]
pub struct TargetArgs {
    pub inferlet: String,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    pub inferlet: String,
}

pub async fn run(cmd: InferletCmd, global: &bootstrap::GlobalArgs) -> Result<Answer> {
    match cmd {
        InferletCmd::List => list(),
        InferletCmd::Info(args) => info(args, global).await,
        InferletCmd::Download(args) => download(args, global).await,
        InferletCmd::Remove(args) => remove(args).await,
    }
}

fn programs_dir() -> std::path::PathBuf {
    bootstrap::paths::pie_home().join("programs")
}

pub(crate) fn cached_version(name: &str) -> Option<ProgramName> {
    open(String::new())
        .cached()
        .into_iter()
        .map(|(program, _, _)| program)
        .filter(|program| program.name == name)
        .max_by(|a, b| version_order(&a.version).cmp(&version_order(&b.version)))
}

fn version_order(version: &str) -> (u64, u64, u64, String) {
    let mut parts = version.split('.').map(|p| p.parse::<u64>().ok());
    match (parts.next(), parts.next(), parts.next()) {
        (Some(Some(major)), Some(Some(minor)), Some(Some(patch))) => {
            (major, minor, patch, String::new())
        }
        _ => (0, 0, 0, version.to_string()),
    }
}

fn open(registry_url: String) -> Repository {
    let mut repo = Repository::new(registry_url, programs_dir());
    repo.load_program_cache();
    repo
}

#[derive(serde::Serialize)]
#[serde(transparent)]
pub struct InferletList {
    inferlets: Vec<CachedInferlet>,
}

#[derive(serde::Serialize)]
struct CachedInferlet {
    name: String,
    version: String,
    description: Option<String>,
    bytes: u64,
}

impl ui::Report for InferletList {
    fn render(&self, palette: &Palette) {
        if self.inferlets.is_empty() {
            println!("nothing downloaded yet");
            println!("  inferlets arrive on first use, or with `pie inferlet download <name>`");
            return;
        }
        let mut table = Table::new([Align::Left, Align::Right, Align::Left], 2);
        for inferlet in &self.inferlets {
            let description = inferlet
                .description
                .as_deref()
                .unwrap_or("")
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            table.push(Row::new(
                Mark::Plain,
                [
                    format!("{}@{}", inferlet.name, inferlet.version),
                    ui::bytes(inferlet.bytes),
                    description,
                ],
            ));
        }
        table.print(palette);
    }
}

fn list() -> Result<Answer> {
    let repo = open(String::new());
    Ok(Answer::report(InferletList {
        inferlets: repo
            .cached()
            .into_iter()
            .map(|(name, manifest, bytes)| CachedInferlet {
                name: name.name,
                version: name.version,
                description: manifest.package.description,
                bytes,
            })
            .collect(),
    }))
}

async fn download(args: TargetArgs, global: &bootstrap::GlobalArgs) -> Result<Answer> {
    let (cfg_path, _) = bootstrap::cli_config_path(global);
    let cfg = crate::derive::load_worker_config(&cfg_path)?;
    let name = resolve_inferlet_id(&args.inferlet, &cfg.server.registry).await?;
    let mut repo = open(cfg.server.registry.clone());
    if repo.exists(&name) {
        return Ok(Answer::noop(format!(
            "{}@{} was already downloaded",
            name.name, name.version
        )));
    }
    repo.add_from_registry(&name, false).await?;
    Ok(Answer::did(format!(
        "downloaded {}@{}",
        name.name, name.version
    )))
}

async fn remove(args: TargetArgs) -> Result<Answer> {
    let mut repo = open(String::new());
    let name = match args.inferlet.split_once('@') {
        Some(_) => ProgramName::parse(&args.inferlet)?,
        None => {
            let matching: Vec<ProgramName> = repo
                .cached()
                .into_iter()
                .map(|(name, _, _)| name)
                .filter(|name| name.name == args.inferlet)
                .collect();
            match matching.len() {
                0 => bail!(
                    "{} is not downloaded; `pie inferlet list` shows what is",
                    args.inferlet
                ),
                1 => matching.into_iter().next().unwrap(),
                _ => {
                    let versions: Vec<String> =
                        matching.iter().map(|n| n.version.clone()).collect();
                    bail!(
                        "{} has {} versions downloaded ({}); name the one to remove",
                        args.inferlet,
                        versions.len(),
                        versions.join(", ")
                    );
                }
            }
        }
    };
    Ok(if repo.remove(&name)? {
        Answer::did(format!("removed {}@{}", name.name, name.version))
    } else {
        Answer::noop(format!("{}@{} was not downloaded", name.name, name.version))
    })
}

async fn info(args: InfoArgs, global: &bootstrap::GlobalArgs) -> Result<Answer> {
    let (cfg_path, _) = bootstrap::cli_config_path(global);
    let cfg = crate::derive::load_worker_config(&cfg_path)?;

    let program = resolve_inferlet_id(&args.inferlet, &cfg.server.registry).await?;
    let manifest = Manifest::from_url(&cfg.server.registry, &program).await?;

    Ok(Answer::report(InferletInfo {
        name: program.name.clone(),
        version: program.version.clone(),
        description: manifest.package.description.clone(),
        authors: manifest.package.authors.clone(),
        repository: manifest.package.repository.clone(),
        runtime: serde_json::to_value(&manifest.runtime)?,
        dependencies: serde_json::to_value(&manifest.dependencies)?,
        parameters: manifest
            .parameters
            .iter()
            .map(|(name, p)| Parameter {
                name: name.clone(),
                r#type: parameter_type_name(&p.param_type),
                optional: p.optional,
                description: p.description.clone(),
            })
            .collect(),
    }))
}

#[derive(serde::Serialize)]
pub struct InferletInfo {
    name: String,
    version: String,
    description: Option<String>,
    authors: Vec<String>,
    repository: Option<String>,
    runtime: serde_json::Value,
    dependencies: serde_json::Value,
    parameters: Vec<Parameter>,
}

#[derive(serde::Serialize)]
struct Parameter {
    name: String,
    r#type: &'static str,
    optional: bool,
    description: Option<String>,
}

impl ui::Report for InferletInfo {
    fn render(&self, palette: &Palette) {
        println!(
            "{}",
            palette.bold(format!("{}@{}", self.name, self.version))
        );
        if let Some(description) = &self.description {
            println!("{description}");
        }
        if let Some(repository) = &self.repository {
            println!("{}", palette.dim(repository));
        }

        if self.parameters.is_empty() {
            println!("\n{}", palette.dim("(no parameters)"));
            return;
        }

        println!("\n{}", palette.bold("Parameters"));
        let name_width = self
            .parameters
            .iter()
            .map(|p| p.name.chars().count())
            .max()
            .unwrap_or(4)
            .max("name".len());
        let type_width = self
            .parameters
            .iter()
            .map(|p| p.r#type.chars().count())
            .max()
            .unwrap_or(4)
            .max("type".len());

        println!(
            "{}",
            palette.dim(format!(
                "{:<name_width$}  {:<type_width$}  required  description",
                "name", "type"
            ))
        );
        for parameter in &self.parameters {
            let required = format!(
                "{:<8}",
                if parameter.optional {
                    "optional"
                } else {
                    "yes"
                }
            );
            let required = if parameter.optional {
                palette.dim(required).to_string()
            } else {
                palette.green(required).to_string()
            };
            println!(
                "{}  {:<type_width$}  {required}  {}",
                palette.accent(format!("{:<name_width$}", parameter.name)),
                parameter.r#type,
                palette.dim(parameter.description.as_deref().unwrap_or("")),
            );
        }
    }
}

#[derive(Deserialize)]
struct RegistryInferlet {
    versions: Vec<RegistryVersion>,
}

#[derive(Deserialize)]
struct RegistryVersion {
    num: String,
}

pub(crate) async fn resolve_inferlet_id(inferlet: &str, registry_url: &str) -> Result<ProgramName> {
    match inferlet.split_once('@') {
        Some((name, "latest")) => {
            validate_bare_inferlet_name(name)?;
            let version = latest_version(name, registry_url).await?;
            Ok(ProgramName {
                name: name.to_string(),
                version,
            })
        }
        Some(_) => ProgramName::parse(inferlet),
        None => {
            validate_bare_inferlet_name(inferlet)?;
            let version = latest_version(inferlet, registry_url).await?;
            Ok(ProgramName {
                name: inferlet.to_string(),
                version,
            })
        }
    }
}

async fn latest_version(name: &str, registry_url: &str) -> Result<String> {
    let url = format!(
        "{}/api/v1/inferlets/{}",
        registry_url.trim_end_matches('/'),
        name
    );
    let resp = reqwest::get(&url)
        .await
        .with_context(|| format!("resolve latest inferlet version from {url}"))?;
    if !resp.status().is_success() {
        bail!(
            "resolve latest inferlet version: {url} returned {}",
            resp.status()
        );
    }
    let body = resp
        .text()
        .await
        .with_context(|| format!("read latest inferlet metadata from {url}"))?;
    latest_version_from_registry_json(&body)
        .with_context(|| format!("resolve latest version for {name:?}"))
}

fn latest_version_from_registry_json(body: &str) -> Result<String> {
    let info: RegistryInferlet =
        serde_json::from_str(body).context("parse registry inferlet metadata")?;
    info.versions
        .into_iter()
        .find(|v| !v.num.is_empty())
        .map(|v| v.num)
        .ok_or_else(|| anyhow!("registry returned no versions"))
}

fn validate_bare_inferlet_name(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        bail!("inferlet name is empty");
    };
    if !first.is_ascii_alphanumeric() {
        bail!("invalid inferlet name {name:?}: must start with an ASCII letter or digit");
    }
    if chars.any(|c| !(c.is_ascii_alphanumeric() || c == '-' || c == '_')) {
        bail!("invalid inferlet name {name:?}: use only ASCII letters, digits, '-' and '_'");
    }
    Ok(())
}

fn parameter_type_name(param_type: &runtime::inferlet::program::ParameterType) -> &'static str {
    match param_type {
        runtime::inferlet::program::ParameterType::String => "string",
        runtime::inferlet::program::ParameterType::Int => "int",
        runtime::inferlet::program::ParameterType::Float => "float",
        runtime::inferlet::program::ParameterType::Bool => "bool",
    }
}
