import re
from typing import Dict, Any

class StructuredExtractor:
    """
    Extracts structured data (Names, Dates, Amounts, Phones, Emails) from raw OCR text.
    Also provides automatic document type classification.
    """

    def __init__(self):
        self.patterns = {
            "date": r"\b(?:\d{1,2}[-/th|st|nd|rd\s]*[a-zA-Z]{3,9}[-/,\s]*\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
            "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "amount": r"(?:[$£€₹]\s?\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*[$£€₹])",
            "key_value": r"(?i)\b(name|total|amount|date|address|phone|email|id|account)\b\s*[:\-]\s*(.+)$"
        }

    def detect_document_type(self, text: str) -> str:
        """Classify the document based on content heuristics."""
        code_markers = ["def ", "class ", "import ", "#include", "public static", "var ", "const ", "let ", "=>", "print(", "console.log"]
        form_markers = ["name:", "date:", "signature", "address:", "dob", "id number"]
        
        lower_text = text.lower()
        if any(marker in lower_text for marker in code_markers):
            return "Source Code"
        elif any(marker in lower_text for marker in form_markers):
            return "ID / Form"
        else:
            return "General Text"

    def extract(self, text: str) -> Dict[str, Any]:
        """Parses the raw text and returns a dictionary of extracted entities."""
        extracted = {
            "dates": [],
            "emails": [],
            "phones": [],
            "amounts": [],
            "key_values": {}
        }

        extracted["dates"] = list(set(re.findall(self.patterns["date"], text)))
        extracted["emails"] = list(set(re.findall(self.patterns["email"], text)))
        extracted["phones"] = list(set(re.findall(self.patterns["phone"], text)))
        extracted["amounts"] = list(set(re.findall(self.patterns["amount"], text)))

        for line in text.split("\n"):
            match = re.search(self.patterns["key_value"], line.strip())
            if match:
                key = match.group(1).title()
                value = match.group(2).strip()
                if key in extracted["key_values"]:
                    if isinstance(extracted["key_values"][key], list):
                        if value not in extracted["key_values"][key]:
                            extracted["key_values"][key].append(value)
                    else:
                        if value != extracted["key_values"][key]:
                            extracted["key_values"][key] = [extracted["key_values"][key], value]
                else:
                    extracted["key_values"][key] = value

        return {k: v for k, v in extracted.items() if v}
