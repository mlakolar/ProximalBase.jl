#!/usr/bin/env python3
"""Generate the five Learn figures, or check the tracked SVGs with --check."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

OUTPUT = Path(__file__).resolve().parent / "src/assets/learn"
INK = "#192438"
QUIET = "#586577"
AXIS = "#8492a6"
BLUE = "#1e55b3"
ORANGE = "#aa4b20"


def figure(title, description):
    image = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": "0 0 480 240",
        "role": "img", "aria-labelledby": "title description",
    })
    ET.SubElement(image, "title", id="title").text = title
    ET.SubElement(image, "desc", id="description").text = description
    ET.SubElement(image, "rect", width="480", height="240", rx="12", fill="#f7f9fc")
    plot = ET.SubElement(image, "g", {
        "font-family": "system-ui, sans-serif", "font-size": "15", "fill": INK,
        "stroke-linecap": "round", "stroke-linejoin": "round",
    })
    label(plot, 24, 28, title)
    return image, plot


def label(plot, x, y, text, color=INK, anchor="start"):
    ET.SubElement(plot, "text", {
        "x": str(x), "y": str(y), "fill": color, "text-anchor": anchor,
    }).text = text


def path(plot, points, color=AXIS, width=1.5):
    ET.SubElement(plot, "polyline", {
        "points": " ".join(f"{x:.2f},{y:.2f}" for x, y in points),
        "fill": "none", "stroke": color, "stroke-width": str(width),
    })


def point(plot, x, y, color):
    ET.SubElement(plot, "circle", cx=str(x), cy=str(y), r="4.5", fill=color)


def threshold():
    image, plot = figure("Soft threshold at one", "Slope one outside the interval from minus one to one. Zero on that interval.")
    x = lambda t: 240 + 55 * t
    y = lambda z: 132 - 33 * z
    path(plot, [(54, y(0)), (424, y(0))])
    path(plot, [(x(0), 55), (x(0), 207)])
    path(plot, [(x(t), y(z)) for t, z in [(-3, -2), (-1, 0), (1, 0), (3, 2)]], BLUE, 3.5)
    for t in (-3, -1, 1, 3):
        path(plot, [(x(t), 135), (x(t), 140)])
        label(plot, x(t), 160, str(t), QUIET, "middle")
    label(plot, 430, 137, "t")
    label(plot, 250, 61, "z")
    return image


def majorizer():
    image, plot = figure("A smooth loss and its quadratic model", "The model is above the smooth loss and touches it at x equals zero.")
    coords = lambda x, y: (98 + 90 * x, 200 - 47 * y)
    path(plot, [coords(-0.45, 0), coords(3.6, 0)])
    path(plot, [coords(0, 0), coords(0, 3.1)])
    abscissae = [i / 20 for i in range(-7, 69)]
    path(plot, [coords(x, (x - 3)**2 / 4) for x in abscissae], BLUE, 3)
    path(plot, [coords(x, 2.25 - 1.5 * x + x*x / 2) for x in abscissae], ORANGE, 3)
    point(plot, *coords(0, 2.25), INK)
    label(plot, 224, 106, "quadratic model", ORANGE)
    label(plot, 262, 173, "smooth loss", BLUE)
    label(plot, 95, 223, "0", QUIET, "middle")
    label(plot, 429, 205, "z")
    return image


def residual():
    image, plot = figure("Update the residual along one column", "Subtract the column (1, one half) from residual (2, 3) to obtain (1, 2.5).")
    path(plot, [(75, 197), (427, 197)])
    path(plot, [(75, 197), (75, 56)])
    origin, old, updated = (75, 197), (281, 62), (178, 84.5)
    path(plot, [origin, old], BLUE, 3)
    path(plot, [origin, updated], ORANGE, 3)
    path(plot, [old, updated], QUIET, 2)
    path(plot, [(193, 75), updated, (195, 88)], QUIET, 2)
    point(plot, *old, BLUE)
    point(plot, *updated, ORANGE)
    label(plot, 293, 65, "r = (2, 3)", BLUE)
    label(plot, 165, 70, "r′ = (1, 2.5)", ORANGE, "end")
    label(plot, 192, 54, "subtract Δa", QUIET)
    return image


def group_shrink():
    image, plot = figure("Shrink along the same group direction", "Move from (3, 4) to (1.8, 2.4), decreasing the vector length from five to three.")
    origin, old, updated = (128, 203), (227, 71), (187.4, 123.8)
    path(plot, [(origin[0], 56), origin, (376, origin[1])])
    path(plot, [origin, old], BLUE, 3)
    path(plot, [origin, updated], ORANGE, 4)
    point(plot, *updated, ORANGE)
    point(plot, *old, BLUE)
    label(plot, 239, 75, "u = (3, 4)", BLUE)
    label(plot, 200, 131, "z = (1.8, 2.4)", ORANGE)
    label(plot, origin[0] - 10, 224, "0", QUIET)
    return image


def exponent_range():
    image, plot = figure("Intermediate and final exponents", "Exponent 1032 exceeds the Float64 finite exponent range. Exponent 1020 fits.")
    position = lambda exponent: 54 + 23 * (exponent - 1018)
    limit = position(1024)
    ET.SubElement(plot, "rect", x=str(limit), y="65", width=str(428 - limit), height="107", rx="6", fill="#f8e8df")
    path(plot, [(54, 133), (428, 133)])
    path(plot, [(limit, 63), (limit, 174)], ORANGE, 2)
    for exponent, color in ((1020, BLUE), (1032, ORANGE)):
        point(plot, position(exponent), 133, color)
        label(plot, position(exponent), 158, str(exponent), color, "middle")
    label(plot, position(1020), 109, "exact result", BLUE, "middle")
    label(plot, position(1032), 109, "a × b", ORANGE, "middle")
    label(plot, limit, 194, "2¹⁰²⁴ exceeds the finite range", QUIET, "middle")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    figures = {
        "soft-threshold": threshold,
        "proximal-majorizer": majorizer,
        "residual-update": residual,
        "group-shrink": group_shrink,
        "exponent-range": exponent_range,
    }
    for name, draw in figures.items():
        image = draw()
        ET.indent(image)
        svg = ET.tostring(image, encoding="unicode") + "\n"
        destination = OUTPUT / f"{name}.svg"
        if args.check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != svg:
                parser.exit(1, f"Figure differs from generated output: {destination}\n")
        else:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            destination.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
