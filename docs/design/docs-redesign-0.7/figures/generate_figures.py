#!/usr/bin/env python3
"""Generate the prototype figures of the 0.7 documentation redesign proposal.

Standard library only, in the style of docs/generate_learn_figures.py, which a
later implementation PR would extend instead of keeping this file. Without
arguments it rewrites the SVG files next to it; --check only compares them with
the generated output. The closed forms below are written again in Julia in
check_figures.jl, which checks them against ProximalBase; that check does not read
the coordinates drawn here.
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

OUTPUT = Path(__file__).resolve().parent
INK = "#192438"
QUIET = "#586577"
AXIS = "#8492a6"
FAINT = "#d5dbe5"
BLUE = "#1e55b3"
ORANGE = "#aa4b20"
FILL = "#e3ebf8"


# ---------------------------------------------------------------- primitives

def figure(title, description, width=480, height=260):
    image = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": f"0 0 {width} {height}",
        "role": "img", "aria-labelledby": "title description",
    })
    ET.SubElement(image, "title", id="title").text = title
    ET.SubElement(image, "desc", id="description").text = description
    ET.SubElement(image, "rect", width=str(width), height=str(height), rx="12", fill="#f7f9fc")
    plot = ET.SubElement(image, "g", {
        "font-family": "system-ui, sans-serif", "font-size": "14", "fill": INK,
        "stroke-linecap": "round", "stroke-linejoin": "round",
    })
    label(plot, 20, 28, title, size="15")
    return image, plot


def label(plot, x, y, text, color=INK, anchor="start", size=None, style=None):
    attributes = {"x": f"{x:.1f}", "y": f"{y:.1f}", "fill": color, "text-anchor": anchor}
    if size:
        attributes["font-size"] = size
    if style:
        attributes["font-style"] = style
    ET.SubElement(plot, "text", attributes).text = text


def path(plot, points, color=AXIS, width=1.5, dash=None, fill="none"):
    attributes = {
        "points": " ".join(f"{x:.2f},{y:.2f}" for x, y in points),
        "fill": fill, "stroke": color, "stroke-width": str(width),
    }
    if dash:
        attributes["stroke-dasharray"] = dash
    ET.SubElement(plot, "polygon" if fill != "none" else "polyline", attributes)


def point(plot, x, y, color, hollow=False):
    if hollow:
        ET.SubElement(plot, "circle", cx=f"{x:.2f}", cy=f"{y:.2f}", r="4.5",
                      fill="#f7f9fc", stroke=color, **{"stroke-width": "2"})
    else:
        ET.SubElement(plot, "circle", cx=f"{x:.2f}", cy=f"{y:.2f}", r="4.5", fill=color)


def arrow(plot, start, end, color=QUIET, width=1.5):
    path(plot, [start, end], color, width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    head = [(end[0] - 8 * ux + 4 * uy, end[1] - 8 * uy - 4 * ux), end,
            (end[0] - 8 * ux - 4 * uy, end[1] - 8 * uy + 4 * ux)]
    path(plot, head, color, width)


def rect(plot, x, y, w, h, fill="#ffffff", stroke=AXIS, width=1.2):
    ET.SubElement(plot, "rect", {
        "x": f"{x:.1f}", "y": f"{y:.1f}", "width": f"{w:.1f}", "height": f"{h:.1f}",
        "fill": fill, "stroke": stroke, "stroke-width": str(width), "rx": "3",
    })


def axes(plot, frame, xlim, ylim, xticks, yticks, xname, yname):
    """Draw axes through zero inside frame=(left, top, right, bottom); return the map."""
    left, top, right, bottom = frame
    sx = lambda t: left + (t - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)
    sy = lambda z: bottom - (z - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)
    path(plot, [(left, sy(0)), (right, sy(0))])
    path(plot, [(sx(0), top), (sx(0), bottom)])
    for t in xticks:
        path(plot, [(sx(t), sy(0) - 3), (sx(t), sy(0) + 3)])
        label(plot, sx(t), sy(0) + 17, f"{t:g}", QUIET, "middle", "12")
    for z in yticks:
        path(plot, [(sx(0) - 3, sy(z)), (sx(0) + 3, sy(z))])
        label(plot, sx(0) - 7, sy(z) + 4, f"{z:g}", QUIET, "end", "12")
    label(plot, right + 6, sy(0) + 4, xname, QUIET, style="italic")
    label(plot, sx(0) + 6, top + 2, yname, QUIET, style="italic")
    return sx, sy


def grid(lo, hi, n=400):
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


# ------------------------------------------------- closed forms (see check_figures.jl)

LAMBDA, A_MCP, A_SCAD, GAMMA = 1.0, 3.0, 3.7, 1.0


def l1(t):
    return LAMBDA * abs(t)


def mcp(t):
    m = abs(t)
    return LAMBDA * m - m * m / (2 * A_MCP) if m <= A_MCP * LAMBDA else A_MCP * LAMBDA ** 2 / 2


def scad(t):
    m = abs(t)
    if m <= LAMBDA:
        return LAMBDA * m
    if m <= A_SCAD * LAMBDA:
        return LAMBDA * m - (m - LAMBDA) ** 2 / (2 * (A_SCAD - 1))
    return (A_SCAD + 1) * LAMBDA ** 2 / 2


def soft(t, tau=GAMMA * LAMBDA):
    return math.copysign(max(abs(t) - tau, 0.0), t)


def hard(t):
    # NormL0(λ) at step γ keeps v when v² > 2γλ.
    return t if t * t > 2 * GAMMA * LAMBDA else 0.0


def firm(t):
    # MCP(λ, a) at a step γ < a: soft below γλ, rescaled up to aλ, identity beyond.
    m = abs(t)
    if m <= GAMMA * LAMBDA:
        return 0.0
    if m <= A_MCP * LAMBDA:
        return math.copysign((m - GAMMA * LAMBDA) / (1 - GAMMA / A_MCP), t)
    return t


# --------------------------------------------------------------- the figures

def penalties():
    image, plot = figure(
        "Three sparsity penalties (λ = 1)",
        "The l1 penalty grows linearly. SCAD with a = 3.7 equals l1 up to one and MCP with a = 3 "
        "starts with the same slope and bends at once; both level off, MCP at 1.5 beyond 3 and "
        "SCAD at 2.35 beyond 3.7.")
    sx, sy = axes(plot, (40, 48, 440, 228), (-5, 5), (0, 3.4), [-4, -2, 2, 4], [1, 2, 3], "t", "penalty")
    ts = grid(-5, 5)
    path(plot, [(sx(t), sy(min(l1(t), 3.4))) for t in ts if l1(t) <= 3.4], QUIET, 2, "6 5")
    path(plot, [(sx(t), sy(mcp(t))) for t in ts], BLUE, 3)
    path(plot, [(sx(t), sy(scad(t))) for t in ts], ORANGE, 3, "10 4 2 4")
    label(plot, sx(2.75), sy(3.1), "ℓ₁: λ|t|", QUIET, "end")
    label(plot, sx(4.95), sy(1.5) + 18, "MCP(1, 3)", BLUE, "end")
    label(plot, sx(4.95), sy(2.35) - 9, "SCAD(1, 3.7)", ORANGE, "end")
    return image


def thresholding():
    image, plot = figure(
        "Scalar proximal maps at step γ = 1, λ = 1",
        "Soft thresholding from NormL1 moves inputs of magnitude above one toward zero by one. Hard "
        "thresholding from NormL0 keeps inputs of magnitude above the square root of two and sets "
        "the rest to zero. Firm thresholding from MCP with a = 3 is zero up to magnitude one, "
        "steeper than the identity up to three and the identity beyond.")
    sx, sy = axes(plot, (40, 44, 440, 236), (-4, 4), (-4, 4), [-3, -1, 1, 3], [-3, 3], "v", "prox(v)")
    path(plot, [(sx(-4), sy(-4)), (sx(4), sy(4))], FAINT, 1.2)
    ts = grid(-4, 4, 800)
    path(plot, [(sx(t), sy(soft(t))) for t in ts], QUIET, 2.5, "6 5")
    # Draw hard thresholding piecewise so that the jump is not joined by a line.
    s2 = math.sqrt(2 * GAMMA * LAMBDA)
    for piece in ([t for t in ts if t < -s2], [t for t in ts if abs(t) <= s2], [t for t in ts if t > s2]):
        path(plot, [(sx(t), sy(hard(t))) for t in piece], ORANGE, 2.5, "10 4 2 4")
    point(plot, sx(s2), sy(s2), ORANGE, hollow=True)
    point(plot, sx(-s2), sy(-s2), ORANGE, hollow=True)
    path(plot, [(sx(t), sy(firm(t))) for t in ts], BLUE, 3)
    # Legend in the empty upper-left quadrant: line sample plus name.
    for i, (name, color, width, dash) in enumerate([
            ("soft: NormL1(1)", QUIET, 2.5, "6 5"),
            ("hard: NormL0(1)", ORANGE, 2.5, "10 4 2 4"),
            ("firm: MCP(1, 3)", BLUE, 3, None)]):
        y = 66 + 20 * i
        path(plot, [(58, y - 4), (88, y - 4)], color, width, dash)
        label(plot, 96, y, name, INK, size="12")
    return image


def group_shrink():
    image, plot = figure(
        "Group soft thresholding (GroupNormL2, weight 1)",
        "Points inside the disc of radius γλ = 1 map to zero. Points outside move toward the "
        "origin along their own direction, and their length decreases by one.",
        width=540)
    # Equal scales on both axes, so that the disc is a circle: 220 px for 5.2 units across, 192 px down.
    half = 2.6 * 192 / 220
    sx, sy = axes(plot, (80, 44, 300, 236), (-2.6, 2.6), (-half, half), [-2, 2], [-2, 2], "x₁", "x₂")
    scale = sx(1) - sx(0)
    ET.SubElement(plot, "circle", cx=f"{sx(0):.2f}", cy=f"{sy(0):.2f}", r=f"{scale:.2f}",
                  fill=FILL, stroke=BLUE, **{"stroke-width": "1.5", "stroke-dasharray": "5 4"})
    for u in [(2.0, 1.2), (-1.6, 1.8), (-2.2, -0.9), (0.9, -2.0)]:
        norm = math.hypot(*u)
        z = (u[0] * (1 - 1 / norm), u[1] * (1 - 1 / norm))
        arrow(plot, (sx(u[0]), sy(u[1])), (sx(z[0]), sy(z[1])), QUIET, 1.5)
        point(plot, sx(u[0]), sy(u[1]), QUIET, hollow=True)
        point(plot, sx(z[0]), sy(z[1]), BLUE)
    inner = (0.45, 0.5)
    arrow(plot, (sx(inner[0]), sy(inner[1])), (sx(0) + 4, sy(0) - 4), QUIET, 1.5)
    point(plot, sx(inner[0]), sy(inner[1]), QUIET, hollow=True)
    label(plot, 340, 100, "○ input u", QUIET)
    label(plot, 340, 120, "● prox(u), step γ = 1", BLUE, size="13")
    label(plot, 340, 150, "Inside the disc ‖u‖ ≤ γλ:", QUIET, size="12")
    label(plot, 340, 166, "the whole group is set to 0.", QUIET, size="12")
    label(plot, 340, 190, "Outside: ‖u‖ shrinks by γλ,", QUIET, size="12")
    label(plot, 340, 206, "direction unchanged.", QUIET, size="12")
    return image


def projections():
    image, plot = figure(
        "Projections of the same point onto three unit balls",
        "The point (1.5, 0.3) projects to (1, 0) on the l1 ball, where its second coordinate "
        "becomes zero, to (0.98, 0.20) on the l2 ball and to (1, 0.3) on the box. Each "
        "projection is the nearest point of the set.",
        width=560, height=250)
    u = (1.5, 0.3)
    panels = [
        ("IndBallL1(1)", [(1, 0), (0, 1), (-1, 0), (0, -1)], (1.0, 0.0), "(1, 0)"),
        ("IndBallL2(1)", None, (u[0] / math.hypot(*u), u[1] / math.hypot(*u)), "(0.98, 0.20)"),
        ("IndBox(-1, 1)", [(1, 1), (-1, 1), (-1, -1), (1, -1)], (1.0, 0.3), "(1, 0.3)"),
    ]
    for i, (name, polygon, z, text) in enumerate(panels):
        left = 22 + 180 * i
        sx = lambda t, left=left: left + 70 + 42 * t
        sy = lambda s: 150 - 42 * s
        path(plot, [(sx(-1.55), sy(0)), (sx(1.9), sy(0))], FAINT, 1)
        path(plot, [(sx(0), sy(-1.55)), (sx(0), sy(1.55))], FAINT, 1)
        if polygon:
            path(plot, [(sx(a), sy(b)) for a, b in polygon], BLUE, 1.8, fill=FILL)
        else:
            ET.SubElement(plot, "circle", cx=f"{sx(0):.2f}", cy=f"{sy(0):.2f}", r="42",
                          fill=FILL, stroke=BLUE, **{"stroke-width": "1.8"})
        path(plot, [(sx(u[0]), sy(u[1])), (sx(z[0]), sy(z[1]))], QUIET, 1.5, "4 3")
        point(plot, sx(u[0]), sy(u[1]), QUIET, hollow=True)
        point(plot, sx(z[0]), sy(z[1]), BLUE)
        label(plot, left + 70, 226, name, INK, "middle", "13")
        label(plot, left + 70, 243, f"prox = {text}", BLUE, "middle", "12")
    label(plot, 540, 28, "○ u = (1.5, 0.3)", QUIET, "end", "12")
    return image


def iterates():
    image, plot = figure(
        "One vector, three ways to store it",
        "A dense vector stores every entry. A SparseIterate stores a working set of entry "
        "indices with their values, which may include a deliberately active zero. A BlockIterate "
        "divides the entries into ordered groups and works with whole groups.",
        width=560, height=300)
    values = ["0", "2.5", "0", "0", "-1", "0", "0", "0.7"]
    cell, left = 34, 196

    def row(y, title, note):
        label(plot, 20, y + 21, title, INK, size="13")
        label(plot, 20, y + 37, note, QUIET, size="11")

    # Dense vector
    row(52, "Vector", "every entry stored")
    for i, v in enumerate(values):
        rect(plot, left + i * cell, 52, cell, 28, "#ffffff" if v == "0" else FILL)
        label(plot, left + i * cell + cell / 2, 71, v, INK, "middle", "12")
        label(plot, left + i * cell + cell / 2, 94, str(i + 1), QUIET, "middle", "10")
    # SparseIterate: packed working set (order of activation is kept)
    row(116, "SparseIterate", "working set of entries")
    active = [(2, "2.5"), (5, "-1"), (8, "0.7"), (3, "0")]
    label(plot, left - 6, 133, "ids", QUIET, "end", "11")
    label(plot, left - 6, 163, "values", QUIET, "end", "11")
    for j, (k, v) in enumerate(active):
        x = left + j * (cell + 6)
        rect(plot, x, 116, cell, 24, "#ffffff")
        label(plot, x + cell / 2, 133, str(k), INK, "middle", "12")
        rect(plot, x, 146, cell, 24, FILL if v != "0" else "#ffffff", BLUE if v == "0" else AXIS,
             1.8 if v == "0" else 1.2)
        label(plot, x + cell / 2, 163, v, INK, "middle", "12")
    label(plot, left + 4 * (cell + 6) + 4, 163, "← an active zero", BLUE, size="11")
    # BlockIterate: ordered groups
    row(196, "BlockIterate", "ordered groups of entries")
    groups = [(0, 3), (3, 5), (5, 8)]
    for g, (a, b) in enumerate(groups):
        x0 = left + a * cell + g * 8
        for i in range(a, b):
            v = values[i]
            rect(plot, x0 + (i - a) * cell, 200, cell, 28, "#ffffff" if v == "0" else FILL)
            label(plot, x0 + (i - a) * cell + cell / 2, 219, v, INK, "middle", "12")
        path(plot, [(x0 + 1, 236), (x0 + 1, 241), (x0 + (b - a) * cell - 1, 241),
                    (x0 + (b - a) * cell - 1, 236)], BLUE, 1.5)
        label(plot, x0 + (b - a) * cell / 2, 257, f"group {g + 1}", BLUE, "middle", "11")
    label(plot, 20, 286, "SymmetricSparseIterate (not shown) does the same for the lower triangle of a "
          "symmetric matrix.", QUIET, size="11")
    return image


FIGURES = {
    "penalties": penalties,
    "thresholding": thresholding,
    "group-shrink": group_shrink,
    "projections": projections,
    "iterates": iterates,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, draw in FIGURES.items():
        image = draw()
        ET.indent(image)
        svg = ET.tostring(image, encoding="unicode") + "\n"
        destination = OUTPUT / f"{name}.svg"
        if args.check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != svg:
                parser.exit(1, f"Figure differs from generated output: {destination}\n")
        else:
            destination.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
