BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def encode(n: int, p: int, pad: int | None = None) -> str:
    """
    Convert a non-negative integer to a base-`p` string using the BASE62 alphabet.

    Args:
        n (int): Non-negative integer to encode.
        p (int): Base for encoding (2..62). Uses digits 0-9, A-Z, a-z.
        pad (int | None): Optional left-padding length using the '0' character
            of BASE62 (i.e., '0'). If provided, the result is left-padded to
            exactly this length.

    Returns:
        str: The base-`p` representation of `n`.

    Raises:
        ValueError: If `n` is negative.

    Examples:
        >>> encode(0, p=2)
        '0'
        >>> encode(42, p=62, pad=2)
        '0g'
        >>> encode(123, p=10)
        '123'
        >>> encode(5, p=16, pad=2)
        '05'
    """
    if n < 0:
        raise ValueError("Cannot encode negative numbers.")
    if not (2 <= p <= len(BASE62)):
        raise ValueError(f"Base p must be in [2, {len(BASE62)}], got {p}.")
    res = ""
    while True:
        n, r = divmod(n, p)
        res = BASE62[r] + res
        if n == 0:
            break
    if pad is None:
        return res
    return res.rjust(pad, BASE62[0])


def decode(s: str, p: int) -> int:
    """
    Inverse of encode: convert a base-`p` string using BASE62 alphabet back to int.

    Args:
        s (str): Encoded string (may contain leading '0's from padding).
        p (int): Base used for encoding/decoding (2..62).

    Returns:
        int: Decoded non-negative integer value.

    Raises:
        ValueError: If base is out of range, string is empty, or contains invalid digits for the base.

    Examples:
        >>> encode(42, p=62, pad=2)
        '0g'
        >>> decode('0g', p=62)
        42
        >>> encode(123, p=10)
        '123'
        >>> decode('123', p=10)
        123
    """
    if not isinstance(s, str):
        raise ValueError("Input must be a string.")
    if not s:
        raise ValueError("Cannot decode empty string.")
    if not (2 <= p <= len(BASE62)):
        raise ValueError(f"Base p must be in [2, {len(BASE62)}], got {p}.")

    val = 0
    for ch in s:
        d = BASE62.find(ch)
        if d == -1 or d >= p:
            raise ValueError(f"Invalid digit '{ch}' for base {p}.")
        val = val * p + d
    return val
