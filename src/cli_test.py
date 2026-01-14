from src.cli import format_bytes, DEFAULT_HOST


# ##################################################################
# test format bytes
# verifies byte formatting for various sizes
def test_format_bytes_bytes() -> None:
    result = format_bytes(500)
    assert result == "500.0 B"


# ##################################################################
# test format bytes kilobytes
# verifies kilobyte formatting
def test_format_bytes_kilobytes() -> None:
    result = format_bytes(2048)
    assert result == "2.0 KB"


# ##################################################################
# test format bytes megabytes
# verifies megabyte formatting
def test_format_bytes_megabytes() -> None:
    result = format_bytes(1048576)
    assert result == "1.0 MB"


# ##################################################################
# test format bytes gigabytes
# verifies gigabyte formatting
def test_format_bytes_gigabytes() -> None:
    result = format_bytes(1073741824)
    assert result == "1.0 GB"


# ##################################################################
# test default host
# verifies default host constant is correct
def test_default_host() -> None:
    assert DEFAULT_HOST == "http://localhost:11435"


# ##################################################################
# test format bytes zero
# verifies zero byte formatting
def test_format_bytes_zero() -> None:
    result = format_bytes(0)
    assert result == "0.0 B"
