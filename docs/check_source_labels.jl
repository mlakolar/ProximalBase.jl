# Check that comments in src/ and test/ and @testset names cite no numerical rule
# labels (issues #936 and #945).
#
# Comments in the sources and tests used to name rules of an older, labeled
# numerical contract, and cases of an older test plan, by short labels such as R1,
# T3, M1, E4, D2 or FN-4. The rendered manual (docs/src/numerics.md) defines none
# of them, so a reader of such a comment has no page to look the rule up in. A
# comment states the behavior in words instead and, where docs/src/numerics.md
# states that behavior, refers to its section. A @testset name states what the
# test checks in words.
#
# This script scans src/**/*.jl and test/**/*.jl and reports each token of the
# form <letters><digits> for the label families NR, FT, R, T, M, E, O, A and D, and
# of the form FN-<digits>, that occurs in a comment or in the literal text of a
# @testset name. It takes the comments and names from Julia's own parser
# (Base.JuliaSyntax), so strings, docstrings, character literals, adjoints and
# macro literals are not comments. A file that the parser rejects fails the check;
# a parser warning alone does not.
#
# The name of a @testset (or Test.@testset) is its first argument that is a string
# literal; only the literal text of that string is searched, not an interpolated
# expression, so "$T1 and $T2" is not a hit and "$T scalar steps (D1)" is. Other
# strings are not searched.
#
# Known limits: a comment inside $( ... ) of a plain backtick command is not seen,
# because the parser keeps command text as a string that the command macro parses
# later, so such a label can be missed. A @testset name that is not a string
# literal (a variable, "a" * "b", a string macro, string(...)) is not searched.
#
# CPU names such as "Apple M3" are exempt (see EXEMPT_PREFIXES).
#
# Usage: julia --startup-file=no docs/check_source_labels.jl [--root DIR] [--self-test]
# Exit status is 1 if a label is found, a file does not parse or a self-test
# fails, 2 for an unknown argument, and 0 otherwise.

const JS = Base.JuliaSyntax

# A label: a family prefix followed by ASCII digits, as a whole word; or FN-, the
# prefix of the test plan items of #504, followed by digits (not inside XFN-4).
const LABEL = r"\b(?:(?:NR|FT|[RTMEOAD])[0-9]+|FN-[0-9]+)\b"

# CPU names are not labels: a token of one of the given families directly
# preceded by the given prefix, as "M3" in "Apple M3". Pairs (prefix, family).
const EXEMPT_PREFIXES = (("Apple ", "M"),)

# Where a label was found: a comment, or the literal text of a @testset name.
const ADVICE = (comment = "a comment; state the rule in words and cite the numerics.md section instead",
                name = "a @testset name; state what the test checks in words instead")

message(path, line, token, kind=:comment) =
    string(path, ":", line, ": label ", token, " in ", ADVICE[kind])

# Whether the green node n has no children. JuliaSyntax of Julia 1.10 has no
# is_leaf; there children(n) returns nothing or an empty tuple for a leaf.
isleaf(n) = (c = JS.children(n); c === nothing || isempty(c))

# Whether the green node n is a call of the macro @testset or Test.@testset. The
# macro name is the first child (Julia 1.13) or, after a leaf @, the second child
# (Julia 1.10), so the text of these children is compared, not a node kind.
function is_testset_call(n, bytes, offset)
    JS.kind(n) == JS.K"macrocall" || return false
    head = JS.children(n)
    len = JS.span(head[1]) + (isleaf(head[1]) ? JS.span(head[2]) : 0)
    return occursin(r"^(?:[A-Za-z_]\w*\.)*@testset$", String(bytes[offset:offset+len-1]))
end

# Append (byte offset, byte length, kind) to acc for each Comment leaf under n
# (kind :comment) and for each literal text piece of the name of a @testset call
# under n (kind :name); offset is the byte index in the source where n starts.
function collect_pieces!(acc, n, offset, bytes)
    isleaf(n) && return JS.kind(n) == JS.K"Comment" ? push!(acc, (offset, JS.span(n), :comment)) : acc
    if is_testset_call(n, bytes, offset)
        o = offset
        for child in JS.children(n)
            if JS.kind(child) == JS.K"string"
                p = o
                for piece in JS.children(child)
                    JS.kind(piece) == JS.K"String" && push!(acc, (p, JS.span(piece), :name))
                    p += JS.span(piece)
                end
                break
            end
            o += JS.span(child)
        end
    end
    for child in JS.children(n)
        collect_pieces!(acc, child, offset, bytes)
        offset += JS.span(child)
    end
    return acc
end

# Vector of (line, text, kind) for the comments and @testset names of the Julia
# source src, in source order: one entry for each comment or name piece and each
# source line it covers, with the part on that line. Throws JS.ParseError if src
# does not parse.
function label_text_lines(src::AbstractString)
    src = String(src)
    tree = JS.parseall(JS.GreenNode, src; ignore_errors=false, ignore_warnings=true)
    bytes = codeunits(src)
    out = Tuple{Int,String,Symbol}[]
    for (a, len, kind) in sort!(collect_pieces!(Tuple{Int,Int,Symbol}[], tree, 1, bytes); by=first)
        line = 1 + count(==(UInt8('\n')), view(bytes, 1:a-1))
        text = String(view(bytes, a:a+len-1))
        for (k, part) in enumerate(split(text, '\n'))
            push!(out, (line + k - 1, String(part), kind))
        end
    end
    return out
end

function exempt(text, m)
    before = SubString(text, 1, prevind(text, m.offset))
    return any(((p, f),) -> endswith(before, p) && startswith(m.match, f), EXEMPT_PREFIXES)
end

# (hits, error) for the Julia source src: hits lists (line, token, kind) for each
# label in a comment (kind :comment) or in the text of a @testset name (kind
# :name), in source order, and error is nothing, or the parser's message when src
# does not parse (hits is then empty).
function scan(src::AbstractString)
    lines = try
        label_text_lines(src)
    catch e
        e isa JS.ParseError || rethrow()
        return Tuple{Int,String,Symbol}[], sprint(showerror, e)
    end
    hits = Tuple{Int,String,Symbol}[]
    for (line, text, kind) in lines, m in eachmatch(LABEL, text)
        exempt(text, m) || push!(hits, (line, String(m.match), kind))
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

# Scan root/src/**/*.jl and root/test/**/*.jl; return (files scanned, label
# messages, parse error messages), in the order of the sorted file paths.
function check_tree(root)
    files = sort!(vcat(julia_files(joinpath(root, "src")), julia_files(joinpath(root, "test"))))
    messages = String[]
    errors = String[]
    for path in files
        rel = relpath_posix(path, root)
        hits, err = scan(read(path, String))
        if err !== nothing
            push!(errors, string(rel, ": Julia's parser rejects this file, so its comments are not checked\n", err))
        end
        for (line, token, kind) in hits
            push!(messages, message(rel, line, token, kind))
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

# FN-<digits> labels, in comments of src/ and test/ files.
append!(SELF_TEST_CASES, [
    ("FN-4 in a comment", "x = 1  # see FN-4\n", [(1, "FN-4")]),
    ("FN-9 in a comment next to a family label", "# FN-9, R1\n", [(1, "FN-9"), (1, "R1")]),
    ("FN-<digits> in a block comment on an inner line", "#= a\nFN-12 =#\n", [(2, "FN-12")]),
    ("FN-4 next to a CPU name exemption", "# Apple M3 and FN-4\n", [(1, "FN-4")]),
    ("FN-<digits> inside a word, FN without digits, FN4", "# XFN-4, FN-, FN-4x, FN4 and FN-a\n", []),
    ("FN-4 in a string and as an expression", "s = \"FN-4\"\nx = FN - 4\n", []),
])

# Each case: (name, Julia source, expected [(line, token, kind), ...]) for the
# names of @testset calls; kind is :name or :comment.
const SELF_TEST_NAME_CASES = Tuple{String,String,Vector{Tuple{Int,String,Symbol}}}[
    # flagged
    ("label in a @testset name",
     "@testset \"R1 cancellation behavior\" begin\n @test true\nend\n", [(1, "R1", :name)]),
    ("label in the literal part of an interpolated name",
     "@testset \"Strong zeros (M1) (\$T)\" for T in (Float32, Float64)\n @test true\nend\n",
     [(1, "M1", :name)]),
    ("label after an interpolation in a name",
     "@testset \"\$T scalar steps (D1)\" for T in (Float32, Float64)\nend\n", [(1, "D1", :name)]),
    ("label in a name after an interpolated expression with a string",
     "@testset \"\$(f(\"R1\")) and T3\" begin\nend\n", [(1, "T3", :name)]),
    ("labels in two literal pieces of one name",
     "@testset \"R1 \$x R2\" begin\nend\n", [(1, "R1", :name), (1, "R2", :name)]),
    ("FN-4 in a @testset name",
     "@testset \"finite overflow depends on the order (FN-4, A1)\" begin\nend\n",
     [(1, "FN-4", :name), (1, "A1", :name)]),
    ("label in a nested @testset name, reported at the inner line",
     "@testset \"outer\" begin\n    x = 1\n    @testset \"inner D2\" begin\n        @test true\n    end\nend\n",
     [(3, "D2", :name)]),
    ("labels in an outer name, an inner name and a comment in the body",
     "@testset \"R1\" begin\n    # T3\n    @testset \"M1\" begin\n    end\nend\n",
     [(1, "R1", :name), (2, "T3", :comment), (3, "M1", :name)]),
    ("label in a @testset name inside a for loop and a let block",
     "for T in (Float32,)\n    let x = 1\n        @testset \"E4\" begin\n        end\n    end\nend\n",
     [(3, "E4", :name)]),
    ("label in a Test.@testset name", "Test.@testset \"R1 here\" begin\nend\n", [(1, "R1", :name)]),
    ("label in a @testset name with an option before the name",
     "@testset verbose = true \"D3 first\" begin\nend\n", [(1, "D3", :name)]),
    ("label in a name after an option, in parentheses",
     "@testset(verbose = true, \"R1 x\", begin\nend)\n", [(1, "R1", :name)]),
    ("label in a triple-quoted name, reported at its line",
     "@testset \"\"\"a\nR1 b\"\"\" begin\nend\n", [(2, "R1", :name)]),
    ("label in the name of a @testset in a function body",
     "function f()\n    @testset \"O7 x\" begin\n    end\nend\n", [(2, "O7", :name)]),
    ("a @testset name and a comment on one line",
     "@testset \"R1\" begin # T3\nend\n", [(1, "R1", :name), (1, "T3", :comment)]),
    # only the first string argument is the name
    ("only the first string literal is the name",
     "@testset \"plain\" \"R1\" begin\nend\n", []),
    # not flagged
    ("descriptive @testset name", "@testset \"scaling by a number follows the broadcast\" begin\nend\n", []),
    ("an interpolated variable named like a label",
     "@testset \"a \$T1 and \$T2\" for T1 in x, T2 in y\nend\n", []),
    ("interpolated expressions named like labels",
     "@testset \"a \$(R1) and \$(T2 + 1)\" begin\nend\n", []),
    ("a label in an ordinary string and in a test",
     "s = \"R1\"\n@test f(\"D2\") == 1\n@test x == \"R1\"\n", []),
    ("a label in a string inside a @testset body",
     "@testset \"descriptive\" begin\n    s = \"R1\"\n    @test f(\"D2\") == 1\nend\n", []),
    ("a variable named like a label inside a @testset body",
     "@testset \"descriptive\" begin\n    R1 = 1\n    D2, T3 = f(R1)\n    @test R1 == D2\nend\n", []),
    ("a label in the string of another macro",
     "@test_throws ErrorException(\"R1\") f()\n@info \"D2\"\n@testset_x \"R1\" begin\nend\n", []),
    ("a label in a string of the options of a @testset",
     "@testset \"descriptive\" label = \"R1\" begin\nend\n", []),
    ("a name that is not a string literal is not searched",
     "name = \"R1\"\n@testset name begin\nend\n@testset \"R1\" * \"x\" begin\nend\n" *
     "@testset string(\"R1\") begin\nend\n@testset raw\"R1\" begin\nend\n", []),
    ("FN-4 inside a word of a name", "@testset \"XFN-4 and FN4\" begin\nend\n", []),
    ("a CPU name in a @testset name", "@testset \"on Apple M3\" begin\nend\n", []),
    ("a CPU name next to a label in a @testset name",
     "@testset \"on Apple M3 and R1\" begin\nend\n", [(1, "R1", :name)]),
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
        want = [(line, token, :comment) for (line, token) in expected]
        report(err === nothing && hits == want, name,
               string(": expected ", want, ", got ", hits,
                      err === nothing ? "" : string(" and a parse error\n", err)))
    end
    for (name, src, expected) in SELF_TEST_NAME_CASES
        hits, err = scan(src)
        report(err === nothing && hits == expected, name,
               string(": expected ", expected, ", got ", hits,
                      err === nothing ? "" : string(" and a parse error\n", err)))
    end
    # a source that does not parse is reported as an error, with no hits
    hits, err = scan("f(x  # R1\n")
    report(err !== nothing && isempty(hits), "a source that does not parse is an error",
           string(": got hits ", hits, " and error ", repr(err)))
    # the whole tree: message format, relative paths, nested directories in src/ and
    # test/, comments and @testset names, files sorted by path
    mktempdir() do tmp
        mkpath(joinpath(tmp, "src", "functions"))
        mkpath(joinpath(tmp, "test", "sub"))
        write(joinpath(tmp, "src", "a.jl"), "x = 1\n# fine\n")
        write(joinpath(tmp, "src", "functions", "b.jl"), "y = 2\n  # R1 here\n")
        write(joinpath(tmp, "test", "c.jl"), "# D2 in a test comment\nz = 3\n")
        write(joinpath(tmp, "test", "sub", "y.jl"),
              "@testset \"outer\" begin\n    @testset \"inner R1 (\$T)\" for T in (Int,)\n    end\nend\n")
        nfiles, messages, errors = check_tree(tmp)
        want = [message("src/functions/b.jl", 2, "R1"),
                message("test/c.jl", 1, "D2"),
                message("test/sub/y.jl", 2, "R1", :name)]
        report(nfiles == 4 && messages == want && isempty(errors), "tree with labels (exit 1)",
               string(": expected 4 files and ", want, ", got ", nfiles, ", ", messages, " and ", errors))
        report(message("test/sub/y.jl", 2, "R1", :name) ==
               "test/sub/y.jl:2: label R1 in a @testset name; state what the test checks in words instead" &&
               message("src/a.jl", 7, "T3") ==
               "src/a.jl:7: label T3 in a comment; state the rule in words and cite the numerics.md section instead",
               "message format for a comment and for a @testset name")
        write(joinpath(tmp, "src", "functions", "b.jl"), "y = 2\n  # fine\n")
        write(joinpath(tmp, "test", "c.jl"), "# fine\nz = 3\n")
        nfiles, messages, errors = check_tree(tmp)
        want = [message("test/sub/y.jl", 2, "R1", :name)]
        report(nfiles == 4 && messages == want && isempty(errors), "labels only in a nested test directory (exit 1)",
               string(": expected 4 files and ", want, ", got ", nfiles, ", ", messages, " and ", errors))
        write(joinpath(tmp, "test", "sub", "y.jl"),
              "@testset \"outer\" begin\n    @testset \"inner (\$T)\" for T in (Int,)\n        R1 = 1\n    end\nend\n")
        nfiles, messages, errors = check_tree(tmp)
        report(nfiles == 4 && isempty(messages) && isempty(errors), "tree without a label (exit 0)",
               string(": got ", nfiles, ", ", messages, " and ", errors))
        # a test file that does not parse is an error, not a silent pass
        write(joinpath(tmp, "test", "sub", "z.jl"), "@testset \"R1\" begin\n")
        nfiles, messages, errors = check_tree(tmp)
        report(nfiles == 5 && isempty(messages) && length(errors) == 1 &&
               startswith(errors[1], "test/sub/z.jl: Julia's parser rejects this file"),
               "a test file that does not parse is an error",
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
