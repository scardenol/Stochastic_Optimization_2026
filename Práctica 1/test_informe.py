"""Comprobaciones estructurales del informe en LaTeX.

No sustituye a una compilación real, pero detecta los errores más comunes:
entornos sin cerrar, llaves desbalanceadas, figuras referenciadas que no existen,
etiquetas duplicadas y referencias rotas.
"""

import re
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).parent
TEX = RAIZ / "informe" / "informe.tex"


def sin_comentarios(texto):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in texto.splitlines())


def test_entornos(t):
    apertura = re.findall(r"\\begin\{([^}]+)\}", t)
    cierre = re.findall(r"\\end\{([^}]+)\}", t)
    ca, cc = Counter(apertura), Counter(cierre)
    desbalance = {k: (ca[k], cc[k]) for k in set(ca) | set(cc) if ca[k] != cc[k]}
    assert not desbalance, f"Entornos desbalanceados: {desbalance}"
    print(f"OK: {len(apertura)} entornos balanceados")


def test_llaves(t):
    n = 0
    for i, ch in enumerate(t):
        if i and t[i - 1] == "\\":
            continue
        if ch == "{":
            n += 1
        elif ch == "}":
            n -= 1
        assert n >= 0, f"Llave de cierre sin apertura cerca del caracter {i}"
    assert n == 0, f"Quedan {n} llaves sin cerrar"
    print("OK: llaves balanceadas")


def test_matematicas(t):
    dobles = len(re.findall(r"(?<!\\)\$\$", t))
    assert dobles % 2 == 0, f"Hay {dobles} delimitadores $$ (debe ser par)"
    sencillos = len(re.findall(r"(?<!\\)(?<!\$)\$(?!\$)", t))
    assert sencillos % 2 == 0, f"Hay {sencillos} delimitadores $ (debe ser par)"
    print(f"OK: matemáticas balanceadas ({sencillos} delimitadores $)")


def test_figuras(t):
    refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)
    assert refs, "El informe no referencia ninguna figura"
    faltantes = [r for r in refs if not (TEX.parent / r).exists()]
    print(f"OK: {len(refs)} figuras referenciadas")
    if faltantes:
        print("  Aún no generadas (se crean al ejecutar el notebook):")
        for f in faltantes:
            print(f"    - {f}")
    return faltantes


def test_etiquetas(t):
    labels = re.findall(r"\\label\{([^}]+)\}", t)
    dup = [k for k, v in Counter(labels).items() if v > 1]
    assert not dup, f"Etiquetas duplicadas: {dup}"

    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", t))
    citas = set(re.findall(r"\\cite\{([^}]+)\}", t))
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", t))

    rotas = refs - set(labels)
    assert not rotas, f"Referencias a etiquetas inexistentes: {rotas}"
    citas_rotas = citas - bib
    assert not citas_rotas, f"Citas sin entrada bibliográfica: {citas_rotas}"
    print(f"OK: {len(labels)} etiquetas, {len(refs)} referencias, {len(citas)} citas")


def test_pendientes(t):
    pend = re.findall(r"\\pendiente\{", t)
    print(f"\nQuedan {len(pend)} marcadores \\pendiente por completar.")
    return len(pend)


def main():
    crudo = TEX.read_text(encoding="utf-8")
    t = sin_comentarios(crudo)

    test_entornos(t)
    test_llaves(t)
    test_matematicas(t)
    test_etiquetas(t)
    faltantes = test_figuras(t)
    n = test_pendientes(t)

    print(f"\nLíneas: {len(crudo.splitlines())}  |  "
          f"figuras faltantes: {len(faltantes)}  |  pendientes: {n}")
    print("Estructura del informe validada.")


if __name__ == "__main__":
    main()
