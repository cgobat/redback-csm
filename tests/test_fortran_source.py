from pathlib import Path

FORTRAN_DIR = Path(__file__).resolve().parents[1] / "fortran"


def test_fortran_free_form_lines_fit_standard_limit():
    # gfortran truncates free-form lines at 132 characters by default, which is
    # a hard error under -Werror=line-truncation (GitHub issue #3).
    too_long = [
        f"{path.name}:{lineno}"
        for path in sorted(FORTRAN_DIR.glob("*.f90"))
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if len(line) > 132
    ]
    assert not too_long, f"Fortran lines longer than 132 characters: {too_long}"
