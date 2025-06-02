## Plan for Updating Model Installing Logic

**Objective:** Replace the current Python-based model installation script with `huggingface-cli` to streamline the process and leverage its features.

**Current Implementation (Assumptions based on typical Python script usage):**

*   A Python script (`model_installer.py` or similar, likely located in `backend/backend-management-rs/src/` or a shared scripts directory) is responsible for downloading models from Hugging Face Hub.
*   This script might use libraries like `requests` or `huggingface_hub` Python library.
*   The Rust code in `backend/backend-management-rs/src/model_installer.rs` likely calls this Python script using `std::process::Command`.

**Proposed Change:**

Modify `backend/backend-management-rs/src/model_installer.rs` to directly invoke `huggingface-cli download` command.

**Detailed Steps:**

1.  **Verify `huggingface-cli` Installation and Availability:**
    *   Determine if `huggingface-cli` is a prerequisite for the system or if it needs to be installed/packaged with the application.
    *   If it needs to be installed, decide on the installation strategy (e.g., bundled, user-installed, or installed on-demand). For now, we'll assume it's available in the system's PATH.

2.  **Identify Current Python Script Logic:**
    *   Locate the exact Python script being used for model downloading.
    *   Analyze the script to understand:
        *   How it takes model repository IDs and revisions/versions as input.
        *   The target download directory structure.
        *   Any specific file types it prioritizes or excludes (e.g., `.safetensors`, `.bin`, `tokenizer.model`, `config.json`).
        *   How it handles authentication (if any, e.g., via `HUGGING_FACE_HUB_TOKEN`).
        *   Error handling and reporting mechanisms.
        *   Any caching mechanisms.

3.  **Map Python Script Functionality to `huggingface-cli download`:**
    *   **Model ID and Revision:** `huggingface-cli download <repo_id> --revision <commit_hash_or_branch_or_tag>`
    *   **Specific Files:** `huggingface-cli download <repo_id> <filename_1> <filename_2> ...`
    *   **Target Directory:** `huggingface-cli download ... --local-dir <path_to_dir>`
    *   **File Types (Include/Exclude):** `huggingface-cli download ... --include <glob_pattern> --exclude <glob_pattern>` (Need to verify if this level of granularity matches the Python script. If not, multiple `download` commands for specific files might be needed, or download all and then filter/delete).
    *   **Authentication:** `huggingface-cli` automatically uses the `HUGGING_FACE_HUB_TOKEN` environment variable if set. This should align with the existing Python script's behavior if it also uses this standard.
    *   **Caching:** `huggingface-cli` has its own caching mechanism (defaults to `~/.cache/huggingface/hub`). We need to ensure this doesn't conflict with existing caching or decide if the CLI's cache is sufficient. The `--cache-dir` option can be used to specify a custom cache location.
    *   **Symlinks for Local Dir:** The `--local-dir-use-symlinks` option (default `auto`) can be relevant. If the Python script was copying files, using symlinks might be a change in behavior. For model installations, direct files are usually preferred over symlinks to avoid issues if the cache is cleared or moved. We should probably use `--local-dir-use-symlinks False` or ensure the default behavior copies files.

4.  **Update Rust Code in `model_installer.rs`:**
    *   **Remove Python Script Invocation:** Delete the code that calls the Python script (e.g., `Command::new("python").arg("path/to/script.py")...`).
    *   **Construct `huggingface-cli` Command:**
        *   Use `std::process::Command` to build the `huggingface-cli download` command.
        *   Dynamically add arguments based on the model to be downloaded (repo ID, revision, specific files, local directory).
        *   Example (not necessarily complete, adjust based on actual requirements):
            ```rust
            use std::process::Command;
            use std::path::Path;

            fn download_model_with_hf_cli(
                repo_id: &str,
                revision: Option<&str>,
                files_to_download: Option<Vec<&str>>, // Or handle include/exclude patterns
                local_dir: &Path,
                cache_dir: Option<&Path>,
            ) -> Result<(), String> {
                let mut cmd = Command::new("huggingface-cli");
                cmd.arg("download");
                cmd.arg(repo_id);

                if let Some(rev) = revision {
                    cmd.arg("--revision").arg(rev);
                }

                // Handle specific files or include/exclude patterns
                if let Some(files) = files_to_download {
                    for file in files {
                        cmd.arg(file); // Add each specific file to download
                    }
                } else {
                    // Or, if downloading all and relying on patterns:
                    // cmd.arg("--include").arg("*.safetensors");
                    // cmd.arg("--include").arg("tokenizer.model");
                    // cmd.arg("--include").arg("config.json");
                    // cmd.arg("--exclude").arg("*.gguf"); // Example exclude
                }

                cmd.arg("--local-dir").arg(local_dir);
                cmd.arg("--local-dir-use-symlinks").arg("False"); // Prefer copying files

                if let Some(cache) = cache_dir {
                    cmd.arg("--cache-dir").arg(cache);
                }

                // Potentially set HUGGING_FACE_HUB_TOKEN if needed explicitly,
                // though huggingface-cli should pick it up from the environment.
                // cmd.env("HUGGING_FACE_HUB_TOKEN", "your_token_if_needed_explicitly");

                let output = cmd.output().map_err(|e| format!("Failed to execute huggingface-cli: {}", e))?;

                if output.status.success() {
                    // Potentially log stdout/stderr for debugging
                    // println!("hf-cli stdout: {}", String::from_utf8_lossy(&output.stdout));
                    // eprintln!("hf-cli stderr: {}", String::from_utf8_lossy(&output.stderr));
                    Ok(())
                } else {
                    let err_msg = format!(
                        "huggingface-cli failed with status: {}\nStdout: {}\nStderr: {}",
                        output.status,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                    Err(err_msg)
                }
            }
            ```
    *   **Error Handling:**
        *   Check the exit status of `huggingface-cli`.
        *   Parse `stdout` and `stderr` for more detailed error messages if needed. `huggingface-cli` usually provides good error messages.
    *   **Configuration:**
        *   Update any configuration files (e.g., `config.json` in `backend-management-rs`) if they previously pointed to the Python script or had Python-specific settings.
        *   Consider if new configuration options are needed for `huggingface-cli` (e.g., custom cache directory, include/exclude patterns if they are to be configurable).

5.  **Testing:**
    *   **Unit Tests:** If possible, mock the `huggingface-cli` execution or test the command construction logic.
    *   **Integration Tests:**
        *   Test with actual model downloads (small models first).
        *   Verify that models are downloaded to the correct `local_dir`.
        *   Check that the correct files are present.
        *   Test with and without a revision specified.
        *   Test with specific filenames if that feature is used.
        *   Test authentication (e.g., by trying to download a gated model if the system supports it, ensuring `HUGGING_FACE_HUB_TOKEN` is picked up).
        *   Test error conditions (e.g., invalid model ID, network issues if mockable, insufficient disk space).
    *   **Existing Test Suite:** Update any existing tests in `backend/backend-management-rs/tests/` that relied on the old Python script behavior.

6.  **Documentation:**
    *   Update `README.md` files or any other developer documentation that mentions the model installation process.
    *   Specifically, document the new dependency on `huggingface-cli` and how to install/configure it if necessary.
    *   Mention the use of `HUGGING_FACE_HUB_TOKEN` for private/gated models.

7.  **Cleanup:**
    *   Once the new implementation is verified and stable, remove the old Python model installation script and any related helper files or configurations.

**Potential Challenges & Considerations:**

*   **`huggingface-cli` Output Parsing:** While `huggingface-cli` is generally good, if specific progress information or detailed status beyond success/failure is needed, parsing its `stdout` might be required, which can be brittle. The current Python script might have offered more structured output or progress.
*   **Granular File Selection:** If the Python script had very complex logic for selecting which files to download from a repository (beyond simple glob patterns), replicating this with `huggingface-cli download` might require multiple calls or downloading more than needed and then deleting unwanted files. The `huggingface-cli download` command can take multiple filenames as arguments to download only those specific files.
*   **Error Reporting Consistency:** Ensure that error messages propagated back to the user or logged by the Rust service are as informative as, or better than, the previous Python script's errors.
*   **Dependency Management:** How `huggingface-cli` itself is installed and kept up-to-date becomes a new consideration.
*   **Backward Compatibility (if applicable):** If the system needs to support environments where `huggingface-cli` cannot be installed, this change might be problematic. However, the request implies a direct replacement.

**File to be primarily modified:**

*   `/home/sslee/Workspace/symphony/backend/backend-management-rs/src/model_installer.rs`

**Supporting files to check/update:**

*   Configuration files (e.g., `/home/sslee/Workspace/symphony/backend/backend-management-rs/config.json`)
*   Test files (e.g., within `/home/sslee/Workspace/symphony/backend/backend-management-rs/tests/`)
*   Documentation (e.g., `README.md` in relevant directories)
*   Potentially CI/CD scripts if they were involved in setting up the Python environment for the old script.
