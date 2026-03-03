from typing import Any, Iterable


def add_state_dict_prefix_aliases(
    state_dict: dict[str, Any],
    *,
    aliases: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Add alias keys for compatible loading across compiled/eager modules.

    Existing keys are kept unchanged; aliases are inserted only when absent.
    """
    remapped_state_dict: dict[str, Any] = dict(state_dict)
    key: str
    value: Any
    for key, value in state_dict.items():
        src_prefix: str
        dst_prefix: str
        for src_prefix, dst_prefix in aliases:
            if not key.startswith(src_prefix):
                continue
            alias_key: str = dst_prefix + key[len(src_prefix) :]
            remapped_state_dict.setdefault(alias_key, value)
    return remapped_state_dict
