"""
Key-Value Store functions for persistent storage.
Mirrors the Rust KVS functions from inferlet/src/lib.rs
"""

from wit_world.imports import kvs as _kvs


def store_get(key: str) -> str | None:
    """
    Retrieves a value from the persistent store for a given key.

    Args:
        key: The key to look up

    Returns:
        The value if found, or None if the key does not exist
    """
    return _kvs.store_get(key)


def store_set(key: str, value: str) -> None:
    """
    Sets a value in the persistent store for a given key.
    This will create a new entry or overwrite an existing one.

    Args:
        key: The key to set
        value: The value to store
    """
    _kvs.store_set(key, value)


def store_delete(key: str) -> None:
    """
    Deletes a key-value pair from the store.
    If the key does not exist, this function does nothing.

    Args:
        key: The key to delete
    """
    _kvs.store_delete(key)


def store_exists(key: str) -> bool:
    """
    Checks if a key exists in the store.

    Args:
        key: The key to check

    Returns:
        True if the key exists, False otherwise
    """
    return _kvs.store_exists(key)


def store_list_keys() -> list[str]:
    """
    Returns a list of all keys currently in the store.

    Returns:
        List of all keys
    """
    return list(_kvs.store_list_keys())
