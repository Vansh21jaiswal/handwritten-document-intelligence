import pytest
from app.extraction.structured import StructuredExtractor

def test_extract_amounts():
    extractor = StructuredExtractor()
    text = "Total amount is $45.00 and shipping is $5."
    res = extractor.extract(text)
    
    assert "amounts" in res
    assert "$45.00" in res["amounts"]
    assert "$5" in res["amounts"]

def test_extract_dates():
    extractor = StructuredExtractor()
    text = "Invoice generated on 12/09/2026 and due 10-Oct-2026."
    res = extractor.extract(text)
    
    assert "dates" in res
    assert "12/09/2026" in res["dates"]
    assert "10-Oct-2026" in res["dates"]

def test_extract_key_values():
    extractor = StructuredExtractor()
    text = "Name: Rahul Sharma\nAddress: 123 Tech Lane"
    res = extractor.extract(text)
    
    assert "key_values" in res
    assert res["key_values"]["Name"] == "Rahul Sharma"
    assert res["key_values"]["Address"] == "123 Tech Lane"

def test_empty_extraction():
    extractor = StructuredExtractor()
    text = "This is just a regular sentence with no specific structured data."
    res = extractor.extract(text)
    
    # Should be empty dict since empty lists are filtered out
    assert len(res) == 0
