#!/usr/bin/env python3
"""Self-test of the comparison rules of summarize.py: python3 benchmark/groupscad/test_summarize.py"""
import array
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402

CASE = "GroupSCAD/Float64/m4096/range/below/soft/value"      # one value per input


def make(root, name, first, all_inputs=None, declared=None, columns=True):
    """A result directory with the files that agree_all reads."""
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "outputs"))
    file = CASE.replace("/", "_") + ".bin"
    with open(os.path.join(d, "outputs", file), "wb") as f:
        array.array("d", first).tofile(f)
    if all_inputs is not None:
        os.makedirs(os.path.join(d, "outputs_all"))
        with open(os.path.join(d, "outputs_all", file), "wb") as f:
            array.array("d", all_inputs).tofile(f)
    with open(os.path.join(d, "verification.csv"), "w") as f:
        if columns:
            f.write("case,order,aa,x_vs_bigfloat,inputs,inputs_matching\n%s,X/R/XC,pass,pass,%s,%s\n" % (CASE, declared, declared))
        else:
            f.write("case,order,aa,x_vs_bigfloat\n%s,X/R/XC,pass,pass\n" % CASE)
    return d


class AgreeAll(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def result(self, a, b):
        return summarize.agree_all(CASE, a, b)

    def test_every_input_agrees(self):
        a = make(self.root, "a", [1.0], [1.0, 2.0], 2)
        b = make(self.root, "b", [1.0], [1.0, 2.0], 2)
        self.assertEqual(self.result(a, b), ("yes", "all (2 inputs)", ""))

    def test_a_later_input_differs(self):
        a = make(self.root, "a", [1.0], [1.0, 2.0], 2)
        b = make(self.root, "b", [1.0], [1.0, 3.0], 2)
        self.assertEqual(self.result(a, b)[0], "no")

    def test_different_record_counts_are_incompatible(self):
        # first outputs [1.0] on both sides; the all-input files have 2 and 3 values for 2 declared inputs
        a = make(self.root, "a", [1.0], [1.0, 2.0], 2)
        b = make(self.root, "b", [1.0], [1.0, 3.0, 4.0], 2)
        self.assertEqual(self.result(a, b)[0], "incompatible")
        self.assertEqual(self.result(b, a)[0], "incompatible")

    def test_empty_all_input_files_are_incompatible(self):
        a = make(self.root, "a", [1.0], [], 2)
        b = make(self.root, "b", [1.0], [], 2)
        self.assertEqual(self.result(a, b)[0], "incompatible")

    def test_declared_counts_must_agree(self):
        a = make(self.root, "a", [1.0], [1.0, 2.0], 2)
        b = make(self.root, "b", [1.0], [1.0, 2.0, 3.0], 3)
        self.assertEqual(self.result(a, b)[0], "incompatible")

    def test_the_first_record_must_match_the_first_input_file(self):
        a = make(self.root, "a", [1.0], [1.5, 2.0], 2)
        b = make(self.root, "b", [1.0], [1.0, 2.0], 2)
        self.assertEqual(self.result(a, b)[0], "incompatible")

    def test_wrong_width_of_the_first_input_file(self):
        a = make(self.root, "a", [1.0, 2.0], [1.0, 2.0], 2)
        b = make(self.root, "b", [1.0, 2.0], [1.0, 2.0], 2)
        self.assertEqual(self.result(a, b)[0], "incompatible")

    def test_an_older_directory_is_compared_on_the_first_input_and_says_so(self):
        a = make(self.root, "a", [1.0], None, None, columns=False)
        b = make(self.root, "b", [1.0], [1.0, 2.0], 2)
        status, compared, _ = self.result(a, b)
        self.assertEqual((status, compared), ("yes", "first (an older directory)"))
        c = make(self.root, "c", [1.0], None, None, columns=False)
        d = make(self.root, "d", [5.0], None, None, columns=False)
        self.assertEqual(self.result(c, d)[:2], ("no", "first (an older directory)"))


if __name__ == "__main__":
    unittest.main()
