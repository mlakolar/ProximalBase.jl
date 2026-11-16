# Check that comments in src/ cite no numerical rule labels (issue #936).
#
# Comments in the sources used to name rules of an older, labeled numerical
# contract by short labels such as R1, T3, M1, E4 or D2. The rendered manual
# (docs/src/numerics.md) defines none of them, so a reader of such a comment has
# no page to look the rule up in. A comment states the behavior in words instead
# and, where docs/src/numerics.md states that behavior, refers to its section.
#
# This script scans src/**/*.jl and reports each token of the form
# <letters><digits> for the label families NR, FT, R, T, M, E, O, A and D that
# occurs in a comment. It takes the comments from Julia's own parser
# (Base.JuliaSyntax), so strings, docstrings, character literals, adjoints and
# macro literals are not comments. A file that the parser rejects fails the check;
# a parser warning alone does not.
#
# Known limit: a comment inside $( ... ) of a plain backtick command is not seen,
# because the parser keeps command text as a string that the command macro parses
# later, so such a label can be missed.
#
# CPU names such as "Apple M3" are exempt (see EXEMPT_PREFIXES).
#
# Usage: julia --startup-file=no docs/check_source_labels.jl [--root DIR] [--self-test]
# Exit status is 1 if a label is found, a file does not parse or a self-test
# fails, 2 for an unknown argument, and 0 otherwise.

const JS = Base.JuliaSyntax

# A label: a family prefix followed by ASCII digits, as a whole word.
const LABEL = r"\b(?:NR|FT|[RTMEOAD])[0-9]+\b"

# CPU names are not labels: a token of one of the given families directly
# preceded by the given prefix, as "M3" in "Apple M3". Pairs (prefix, family).
const EXEMPT_PREFIXES = (("Apple ", "M"),)

message(path, line, token) = string(path, ":", line, ": label ", token,
    " in a comment; state the rule in words and cite the numerics.md section instead")

# Whether the green node n has no children. JuliaSyntax of Julia 1.10 has no
# is_leaf; there children(n) returns nothing or an empty tuple for a leaf.
isleaf(n) = (c = JS.children(n); c === nothing || isempty(c))

# Append (byte offset, byte length) of each Comment leaf under n to acc, in
# source order; offset is the byte index in the source where n starts.
function collect_comments!(acc, n, offset)
    if isleaf(n)
        JS.kind(n) == JS.K"Comment" && push!(acc, (offset, JS.span(n)))
    else
        for child in JS.children(n)
            collect_comments!(acc, child, offset)
            offset += JS.span(child)
        end
    end
    return acc
end

# Vector of (line, text) for the comments of the Julia source src: one entry for
# each comment and each source line it covers, with the part of the comment on
# that line. Throws JS.ParseError if src does not parse.
function comment_lines(src::AbstractString)
    src = String(src)
    tree = JS.parseall(JS.GreenNode, src; ignore_errors=false, ignore_warnings=true)
    bytes = codeunits(src)
    out = Tuple{Int,String}[]
    for (a, len) in collect_comments!(Tuple{Int,Int}[], tree, 1)
        line = 1 + count(==(UInt8('\n')), view(bytes, 1:a-1))
        text = String(view(bytes, a:a+len-1))
        for (k, part) in enumerate(split(text, '\n'))
            push!(out, (line + k - 1, String(part)))
        end
    end
    return out
end

function exempt(text, m)
    before = SubString(text, 1, prevind(text, m.offset))
    return any(((p, f),) -> endswith(before, p) && startswith(m.match, f), EXEMPT_PREFIXES)
end

# (hits, error) for the Julia source src: hits lists (line, token) for each label
# in a comment, and error is nothing, or the parser's message when src does not
# parse (hits is then empty).
function scan(src::AbstractString)
    lines = try
        comment_lines(src)
    catch e
        e isa JS.ParseError || rethrow()
        return Tuple{Int,String}[], sprint(showerror, e)
    end
    hits = Tuple{Int,String}[]
    for (line, text) in lines, m in eachmatch(LABEL, text)
        exempt(text, m) || push!(hits, (line, String(m.match)))
    end
    return hits, nothing
end

# Sorted paths of the .jl files under dir, recursively.
function julia_files(dir)
    files = String[]
    isdir(dir) || return files
    for (d, _, names) in walkdir(dir), name in names
        endswith(name, ".jl") && push!(files, joinpath(d, name))
    end
    return sort!(files)
end

relpath_posix(path, root) = join(splitpath(relpath(path, root)), "/")

# Scan root/src/**/*.jl; return (files scanned, label messages, parse error messages).
function check_tree(root)
    files = julia_files(joinpath(root, "src"))
    messages = String[]
    errors = String[]
    for path in files
        rel = relpath_posix(path, root)
        hits, err = scan(read(path, String))
        if err !== nothing
            push!(errors, string(rel, ": Julia's parser rejects this file, so its comments are not checked\n", err))
        end
        for (line, token) in hits
            push!(messages, message(rel, line, token))
        end
    end
    return length(files), messages, errors
end

function run_check(root)
    nfiles, messages, errors = check_tree(root)
    foreach(println, errors)
    foreach(println, messages)
    summary = string("check_source_labels: ", nfiles, " files scanned, ", length(messages), " hits")
    isempty(errors) || (summary *= string(", ", length(errors), " files that do not parse"))
    println(summary)
    return isempty(messages) && isempty(errors) ? 0 : 1
end

# Each case: (name, Julia source, expected [(line, token), ...]).
const SELF_TEST_CASES = Tuple{String,String,Vector{Tuple{Int,String}}}[
    # flagged
    ("label in a line comment", "x = 1\n# R1 here\ny = 2\n", [(2, "R1")]),
    ("trailing comment after code", "x = f(y)  # see T3\n", [(1, "T3")]),
    ("single-line block comment", "x = 1 #= M1 =# + 2\n", [(1, "M1")]),
    ("multi-line block comment, label on an inner line",
     "a = 1\n#= first\nsecond E4\nthird\n=#\nb = 2\n", [(3, "E4")]),
    ("nested block comment",
     "#= outer #= inner D2 =# still outer =#\ncode\n", [(1, "D2")]),
    ("label on the line after a nested block closes",
     "#= a\n#= b =#\nR3 =#\nx = 1\n", [(3, "R3")]),
    ("two labels on one line", "# (T3, R3)\n", [(1, "T3"), (1, "R3")]),
    ("label after a string that contains a #",
     "s = \"a # b\"  # R1 here\n", [(1, "R1")]),
    ("label after a docstring", "\"\"\"\ndoc # R1\n\"\"\"\nf() = 1  # E2\n", [(4, "E2")]),
    ("label after an escaped quote in a string",
     "s = \"a\\\" # R1\"  # NR1\n", [(1, "NR1")]),
    ("label after a character literal of a quote", "c = '\"'  # FT2\n", [(1, "FT2")]),
    ("label after an escaped character literal", "c = '\\''  # O7\n", [(1, "O7")]),
    ("label after an adjoint", "x = A'  # A1\n", [(1, "A1")]),
    ("label after a backtick command that contains a #",
     "run(`echo # not a comment`)  # D1\n", [(1, "D1")]),
    ("label after a string macro with an escaped backslash",
     "r = r\"a\\\\\"  # R2\n", [(1, "R2")]),
    ("label after a string with an interpolated string",
     "s = \"\$(f(\"# x\")) #\"  # T1\n", [(1, "T1")]),
    ("label after an interpolation in a triple-quoted string",
     "doc = \"\"\"\n\$(f(\")\")) # R1\n\"\"\"  # E3\n", [(3, "E3")]),
    ("label after a quote character literal at the start of the text", "'\"' # R1\n", [(1, "R1")]),
    ("label after an adjoint of a name with a prime", "x′'#' R1\n", [(1, "R1")]),
    ("label after an adjoint of a name with a combining mark", "â'#' R1\n", [(1, "R1")]),
    ("label after a command macro literal that looks like an interpolation",
     "c = custom`\$(# R1)` # R2\n", [(1, "R2")]),
    ("label after a multi-line command macro literal",
     "c = custom`\$(# R1\n)` # R2\n", [(2, "R2")]),
    ("labels next to the CPU name exemption",
     "# Apple R1 and Apple D2 and Apple M3\n", [(1, "R1"), (1, "D2")]),
    # not flagged
    ("code after a character literal # at the start of the text", "'#'; E2 = 1\n", []),
    ("variables named like labels", "E1, E2 = f(x)\nR1 = E2 + 1\n", []),
    ("label in a string", "s = \"text # R1\"\n", []),
    ("label inside a string with a hash", "s = \"a # b R1\"\n", []),
    ("label in a docstring with a hash",
     "\"\"\"\nA docstring with R1 and a # sign.\n\"\"\"\nf() = 1\n", []),
    ("code after a character literal #", "c = '#'; E2 = 1\n", []),
    ("adjoint, comment without a label, label as code afterwards",
     "x = A' * B  # plain comment\nx = A'; M1 = 2\n", []),
    ("backtick command with a #", "run(`echo # R1`)\n", []),
    ("triple backtick command with a #", "run(```echo # R1```)\n", []),
    ("CPU name", "# measured on Apple M3 and Apple M2\n", []),
    ("label in a string after an interpolation", "s = \"\$(x) # R1\"\n", []),
    ("label in a string inside an interpolation", "s = \"\$(f(\"# R1\"))\"\n", []),
    ("label inside a raw string macro", "r = raw\"\\\" # R1\"\n", []),
    ("no comment at all", "function f(x)\n  x + 1\nend\n", []),
    ("label-like words that are not labels",
     "# R10x, XR1, Apple, T3rd, ER1 and E1x\n", []),
    # the exemption does not hide a label next to a CPU name
    ("label next to a CPU name", "# Apple M3 and R1\n", [(1, "R1")]),
    ("M3 not after Apple", "# a machine M3\n", [(1, "M3")]),
    ("plain string after an operator interpolates",
     "x = a ⊗ \"\$(1 # R1\n)\"\n", [(1, "R1")]),
    # the documented limit: Julia's command macro parses the interpolation later
    # and would see R1 as a comment, but the parser keeps it as command text
    ("documented limit: label in an interpolation of a plain command is not seen",
     "c = `\$(1 # R1\n)`\n", []),
    # a quote after unary ! or after return
    ("character literal # after unary !, label as code", "!'#'; E2 = 1\n", []),
    ("label after a quote character literal after unary !", "!'\"' # R1\n", [(1, "R1")]),
    ("character literal # after return, label as code", "f() = return'#'; E2 = 1\n", []),
    ("plain string after unary ! interpolates",
     "!\"\$(1 # R1\n)\" # R2\n", [(1, "R1"), (2, "R2")]),
    # a source with a parser warning (here a space in a relative import) is scanned
    ("label after code with a parser warning", "import . .A  # R1\n", [(1, "R1")]),
]

# Names that end in one of these characters: in these cases, a quote after the
# name is an adjoint (x∇'), and a string or command right after it is a macro literal.
for c in ("∇", "∂", "℘", "₊", "₋", "₌", "₍", "₎")
    push!(SELF_TEST_CASES,
        ("label after an adjoint of a name ending in U+$(uppercase(string(UInt32(c[1]), base=16, pad=4)))",
         "x$(c)'#' R1\n", [(1, "R1")]))
end
for c in ("∇", "₍", "₎")
    u = uppercase(string(UInt32(c[1]), base=16, pad=4))
    push!(SELF_TEST_CASES,
        ("string macro prefix ending in U+$u does not interpolate",
         "s = x$(c)\"\$(# R1)\" # R2\n", [(1, "R2")]))
    push!(SELF_TEST_CASES,
        ("command macro prefix ending in U+$u does not interpolate",
         "c = x$(c)`\$(# R1)` # R2\n", [(1, "R2")]))
end
# A quote after these operators opens a character literal.
for c in ("⊗", "√")
    push!(SELF_TEST_CASES,
        ("character literal # after the operator U+$(uppercase(string(UInt32(c[1]), base=16, pad=4))), label as code",
         "x$(c)'#'; E2 = 1\n", []))
end
# Operators that end in a suffix character: in these cases, a quote after them
# opens a character literal, and a string or command after them is a plain literal.
for s in ("₊", "⁻", "₍", "‵")
    u = uppercase(string(UInt32(s[1]), base=16, pad=4))
    op = "⊗" * s
    push!(SELF_TEST_CASES,
        ("character literal # after the operator ⊗ with suffix U+$u, label as code",
         "x$(op)'#'; E2 = 1\n", []))
    push!(SELF_TEST_CASES,
        ("label after a quote character literal after the operator ⊗ with suffix U+$u",
         "x$(op)'\"' # R1\n", [(1, "R1")]))
    push!(SELF_TEST_CASES,
        ("plain string after the operator ⊗ with suffix U+$u interpolates",
         "x$(op)\"\$(1 # R1\n)\" # R2\n", [(1, "R1"), (2, "R2")]))
    # Julia would also see R1 here; see the documented limit above
    push!(SELF_TEST_CASES,
        ("documented limit: plain command after the operator ⊗ with suffix U+$u, R1 not seen",
         "x$(op)`\$(1 # R1\n)` # R2\n", [(2, "R2")]))
end

function self_test()
    passed = 0
    failed = 0
    report(ok, name, detail="") = if ok
        passed += 1
        println("ok   ", name)
    else
        failed += 1
        println("FAIL ", name, detail)
    end
    for (name, src, expected) in SELF_TEST_CASES
        hits, err = scan(src)
        report(err === nothing && hits == expected, name,
               string(": expected ", expected, ", got ", hits,
                      err === nothing ? "" : string(" and a parse error\n", err)))
    end
    # a source that does not parse is reported as an error, with no hits
    hits, err = scan("f(x  # R1\n")
    report(err !== nothing && isempty(hits), "a source that does not parse is an error",
           string(": got hits ", hits, " and error ", repr(err)))
    # the whole tree: message format, relative paths, nested directories, test/ ignored
    mktempdir() do tmp
        mkpath(joinpath(tmp, "src", "functions"))
        mkpath(joinpath(tmp, "test"))
        write(joinpath(tmp, "src", "a.jl"), "x = 1\n# fine\n")
        write(joinpath(tmp, "src", "functions", "b.jl"), "y = 2\n  # R1 here\n")
        write(joinpath(tmp, "test", "c.jl"), "# R1 in a test comment is outside src/\n")
        nfiles, messages, errors = check_tree(tmp)
        want = [message("src/functions/b.jl", 2, "R1")]
        report(nfiles == 2 && messages == want && isempty(errors), "tree with a label (exit 1)",
               string(": expected 2 files and ", want, ", got ", nfiles, ", ", messages, " and ", errors))
        write(joinpath(tmp, "src", "functions", "b.jl"), "y = 2\n  # fine\n")
        nfiles, messages, errors = check_tree(tmp)
        report(nfiles == 2 && isempty(messages) && isempty(errors), "tree without a label (exit 0)",
               string(": got ", nfiles, ", ", messages, " and ", errors))
    end
    println("check_source_labels self-test: ", passed, " passed, ", failed, " failed")
    return failed == 0 ? 0 : 1
end

function main(args)
    root = dirname(@__DIR__)
    selftest = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--self-test"
            selftest = true
        elseif a == "--root" && i < length(args)
            root = args[i += 1]
        elseif startswith(a, "--root=")
            root = a[length("--root=")+1:end]
        else
            println(stderr, "usage: julia docs/check_source_labels.jl [--root DIR] [--self-test]")
            return 2
        end
        i += 1
    end
    return selftest ? self_test() : run_check(root)
end

exit(main(ARGS))
