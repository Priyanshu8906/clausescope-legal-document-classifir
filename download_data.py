"""Download the public LEDGAR Parquet splits from the Hugging Face dataset repo."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/ledgar"
LABEL_NAMES = [
    "Adjustments", "Agreements", "Amendments", "Anti-Corruption Laws", "Applicable Laws", "Approvals",
    "Arbitration", "Assignments", "Assigns", "Authority", "Authorizations", "Base Salary", "Benefits",
    "Binding Effects", "Books", "Brokers", "Capitalization", "Change In Control", "Closings",
    "Compliance With Laws", "Confidentiality", "Consent To Jurisdiction", "Consents", "Construction",
    "Cooperation", "Costs", "Counterparts", "Death", "Defined Terms", "Definitions", "Disability",
    "Disclosures", "Duties", "Effective Dates", "Effectiveness", "Employment", "Enforceability",
    "Enforcements", "Entire Agreements", "Erisa", "Existence", "Expenses", "Fees", "Financial Statements",
    "Forfeitures", "Further Assurances", "General", "Governing Laws", "Headings", "Indemnifications",
    "Indemnity", "Insurances", "Integration", "Intellectual Property", "Interests", "Interpretations",
    "Jurisdictions", "Liens", "Litigations", "Miscellaneous", "Modifications", "No Conflicts", "No Defaults",
    "No Waivers", "Non-Disparagement", "Notices", "Organizations", "Participations", "Payments", "Positions",
    "Powers", "Publicity", "Qualifications", "Records", "Releases", "Remedies", "Representations", "Sales",
    "Sanctions", "Severability", "Solvency", "Specific Performance", "Submission To Jurisdiction", "Subsidiaries",
    "Successors", "Survival", "Tax Withholdings", "Taxes", "Terminations", "Terms", "Titles",
    "Transactions With Affiliates", "Use Of Proceeds", "Vacations", "Venues", "Vesting", "Waiver Of Jury Trials",
    "Waivers", "Warranties", "Withholdings",
]


def main() -> None:
    output = Path(__file__).parent / "data" / "raw" / "ledgar"
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        filename = f"{split}-00000-of-00001.parquet"
        path = output / f"{split}.parquet"
        if not path.exists():
            print(f"Downloading {split} split…", flush=True)
            urlretrieve(f"{BASE}/{filename}", path)
        print(f"{split}: {path.stat().st_size / 1_000_000:.1f} MB → {path}", flush=True)
    (output / "labels.json").write_text(json.dumps({"dataset": "coastalcph/lex_glue / ledgar", "label_names": LABEL_NAMES}, indent=2), encoding="utf-8")
    print("Saved label vocabulary.")


if __name__ == "__main__":
    main()
