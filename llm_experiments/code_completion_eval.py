import ast
import subprocess
import tempfile
import os

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def lint_with_flake8(code: str, verbose: bool = False) -> float:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()

        try:
            result = subprocess.run(
                ["flake8", "--format=%(row)d:%(col)d:%(code)s:%(text)s", f.name],
                capture_output=True,
                text=True,
            )
            errors = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(":", 3)
                    if len(parts) >= 4:
                        errors.append({"code": parts[2], "message": parts[3]})
        finally:
            os.unlink(f.name)

    ans = 0
    if verbose:
        print(f"Linter errors for {code}: {errors}")

    for errors in errors:
        if errors["code"].startswith("F"):
            return -0.5

        if errors["code"] != "E501" and errors["code"].startswith("E"):
            ans -= 0.2

        if errors["code"].startswith("W"):
            ans -= 0.1

    return max(ans, -0.5)


def eval_result(input_text: str, original_completion: str, generated_completion: str):

    generated_code = input_text + generated_completion
    try:
        ast.parse(generated_code)
    except SyntaxError:
        ast_parses = False
    else:
        ast_parses = True

    linter_score = lint_with_flake8(generated_code)

    chencherry = SmoothingFunction()
    bleu_score = sentence_bleu(
        [original_completion.split()],
        generated_completion.split(),
        smoothing_function=chencherry.method1,
    )

    return (int(ast_parses), linter_score, bleu_score)


def eval_results(split_corpus: dict[str, str], results: dict[str, str]):
    for k, v in results.items():
        parser_score, linter_score, bleu_score = eval_result(k, split_corpus[k], v)
        print(f"Score for {k} // {v}: ")
        print(f"(original completion: {split_corpus[k]})")
        print(
            f"-->  parser: {parser_score}, linter: {linter_score}, bleu: {bleu_score}"
        )
