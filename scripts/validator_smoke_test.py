from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validators import validate_answer


def main() -> None:
    context = (
        "PMID: 41166960\n"
        "Title: Comparative Efficacy and Safety of Direct Oral Anticoagulants Versus Warfarin\n"
        "Abstract: In patients with atrial fibrillation and cancer, DOACs were associated with lower "
        "ischemic stroke/systemic embolism and major hemorrhage compared with warfarin.\n\n---\n\n"
        "PMID: 41178907\n"
        "Title: Anticoagulation therapy and dementia in atrial fibrillation\n"
        "Abstract: DOACs were associated with lower intracerebral bleeding and lower mortality."
    )
    answer = (
        "DOACs are generally associated with lower stroke and major bleeding risk versus warfarin "
        "in atrial fibrillation populations [41166960]."
    )

    result = validate_answer(
        user_query="DOACs vs warfarin for AF stroke prevention",
        answer=answer,
        context=context,
        source_pmids=["41166960", "41178907"],
        model_name="MoritzLaurer/DeBERTa-v3-base-mnli",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
