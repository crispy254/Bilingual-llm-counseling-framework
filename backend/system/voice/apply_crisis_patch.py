"""
apply_crisis_patch.py - run ONCE from the backend folder:  python apply_crisis_patch.py
Makes the fixed crisis response follow the language the student wrote in
(Korean -> Korean, English -> English, mixed -> both). Backups: *.bak
"""
import shutil, sys

def patch(path, edits):
    src = open(path, encoding="utf-8").read()
    if all(new in src for _, new in edits):
        print(f"{path}: already patched"); return
    shutil.copy(path, path + ".bak")
    for old, new in edits:
        if src.count(old) != 1:
            sys.exit(f"STOP: expected exactly one occurrence in {path} of:\n  {old}\nFound {src.count(old)}. Nothing changed in this file.")
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print(f"{path}: patched (backup {path}.bak)")

patch("prompt_builder.py", [(
    '    return CRISIS_RESPONSE_KO if language == "ko" else CRISIS_RESPONSE_EN',
    '    if language == "both":\n        return CRISIS_RESPONSE_KO + "\\n\\n" + CRISIS_RESPONSE_EN\n'
    '    return CRISIS_RESPONSE_KO if language == "ko" else CRISIS_RESPONSE_EN')])

patch("chatbot_engine.py", [
    ("import strategy_selector\n", "import strategy_selector\nimport crisis_detector\n"),
    ('"response": prompt_builder.get_crisis_response(language),',
     '"response": prompt_builder.get_crisis_response(\n                crisis_detector.crisis_response_language(user_message, language)),')])
