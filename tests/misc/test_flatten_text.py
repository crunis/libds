from libds.misc import flatten_text

def test_flatten_text():
    assert flatten_text(" ÁéÍóÚ ") == "aeiou"