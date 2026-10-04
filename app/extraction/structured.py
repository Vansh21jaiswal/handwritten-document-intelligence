import re
from typing import Dict, Any

class StructuredExtractor:
    """
    Extracts structured data (Names, Dates, Amounts, Phones, Emails) from raw OCR text.
    Uses regex patterns common in financial documents, notes, and forms.
    """

    def __init__(self):
        self.patterns = {
            "date": r"\b(?:\d{1,2}[-/th|st|nd|rd\s]*[a-zA-Z]{3,9}[-/,\s]*\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
            "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "amount": r"(?:[$£€₹]\s?\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*[$£€₹])",
            # Basic key-value matches (e.g. "Name: John Doe", "Total Amount: $500")
            "key_value": r"(?i)\b(name|total|amount|date|address|phone|email)\b\s*[:\-]\s*(.+)$"
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Parses the raw text and returns a dictionary of extracted entities.
        """
        extracted = {
            "dates": [],
            "emails": [],
            "phones": [],
            "amounts": [],
            "key_values": {}
        }

        # 1. Standard pattern matching
        extracted["dates"] = list(set(re.findall(self.patterns["date"], text)))
        extracted["emails"] = list(set(re.findall(self.patterns["email"], text)))
        extracted["phones"] = list(set(re.findall(self.patterns["phone"], text)))
        extracted["amounts"] = list(set(re.findall(self.patterns["amount"], text)))

        # 2. Key-Value pairs line by line
        for line in text.split("\n"):
            match = re.search(self.patterns["key_value"], line.strip())
            if match:
                key = match.group(1).title()
                value = match.group(2).strip()
                # Store it, handling duplicates by making a list if necessary
                if key in extracted["key_values"]:
                    if isinstance(extracted["key_values"][key], list):
                        extracted["key_values"][key].append(value)
                    else:
                        extracted["key_values"][key] = [extracted["key_values"][key], value]
                else:
                    extracted["key_values"][key] = value

        # Clean up empty lists to keep the JSON tidy
        return {k: v for k, v in extracted.items() if v}

    def format_as_markdown(self, extraction_dict: Dict[str, Any]) -> str:
        """Helper to format the extracted JSON into a clean markdown string."""
        if not extraction_dict:
            return "No structured data detected."
            
        md = ""
        if "key_values" in extraction_dict:
            md += "### Key Fields\n"
            for k, v in extraction_dict["key_values"].items():
                val_str = ", ".join(v) if isinstance(v, list) else v
                md += f"- **{k}**: {val_str}\n"
                
        for entity_type in ["dates", "amounts", "emails", "phones"]:
            if entity_type in extraction_dict:
                md += f"### {entity_type.title()}\n"
                for item in extraction_dict[entity_type]:
                    md += f"- {item}\n"
                    
        return md
