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
        except Exception as e:
            print(f"Error running flake8: {e}")
            errors = []
        finally:
            os.unlink(f.name)

    ans = 0
    if verbose:
        print(f"Linter errors for {code}: {errors}")

    for error in errors:
        if error["code"].startswith("F"):
            return -0.5

        if error["code"] != "E501" and error["code"].startswith("E"):
            ans -= 0.2

        if error["code"].startswith("W"):
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

    smoothing_fn = SmoothingFunction()
    bleu_score = sentence_bleu(
        [original_completion.split()],
        generated_completion.split(),
        smoothing_function=smoothing_fn.method1,
    )

    return (int(ast_parses), linter_score, bleu_score)


def eval_results(split_corpus: dict[str, str], results: dict[str, str]):
    """Evaluate completion results and return structured scores.

    Returns:
        List of dicts with keys: input, output, original_completion,
        parser_score, linter_score, bleu_score
    """
    evaluation_results = []
    for k, v in results.items():
        parser_score, linter_score, bleu_score = eval_result(k, split_corpus[k], v)
        evaluation_results.append({
            "input": k,
            "output": v,
            "original_completion": split_corpus[k],
            "parser_score": parser_score,
            "linter_score": linter_score,
            "bleu_score": bleu_score,
        })
    return evaluation_results
