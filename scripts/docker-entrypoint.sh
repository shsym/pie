#!/bin/bash
# Docker entrypoint script for PIE
# Handles authentication setup before starting the server

set -e

# Function to set up or update authentication
# Removes existing key if present, then adds the new one (idempotent)
setup_auth() {
    local username="$1"
    local pubkey="$2"
    local keyname="${3:-docker-key}"

    echo "🔐 Setting up authentication for user: $username" >&2

    # Remove existing key if it exists (makes this idempotent)
    pie auth remove "$username" "$keyname" 2>/dev/null || true

    # Add the new key
    if echo "$pubkey" | pie auth add "$username" "$keyname"; then
        echo "✅ Authentication configured successfully" >&2
    else
        echo "❌ Failed to configure authentication" >&2
        exit 1
    fi
}

# Only set up authentication if credentials are explicitly provided
# If not provided, will use cached auth from mounted volume (if exists)
if [ -n "$PIE_AUTH_USER" ] && [ -n "$PIE_AUTH_KEY" ]; then
    setup_auth "$PIE_AUTH_USER" "$PIE_AUTH_KEY" "${PIE_AUTH_KEY_NAME:-docker-key}"
elif [ -n "$PIE_AUTH_USER" ] && [ -n "$PIE_AUTH_KEY_FILE" ] && [ -f "$PIE_AUTH_KEY_FILE" ]; then
    PIE_AUTH_KEY=$(cat "$PIE_AUTH_KEY_FILE")
    setup_auth "$PIE_AUTH_USER" "$PIE_AUTH_KEY" "${PIE_AUTH_KEY_NAME:-docker-key}"
fi

# Execute the main command (pie serve or whatever was passed)
exec "$@"
