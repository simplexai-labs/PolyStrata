"""0.2.17 GH-LIST #6255 / #6256 (iMilnb): every HIP build the Terminal starts names the chosen ROCm's own include folder before the system's
(`-isystem <root>/include` first in CMAKE_HIP_FLAGS), so a distro's older HIP headers in /usr/include never pair with the chosen ROCm's compiler.
#7138: every path in those flags is quoted, so an install folder named with a space reaches clang++ as one path; the Windows build .bat
keeps the quotes (setup.bat_arg).  The flag strings are read from the source (no ROCm, no compiler here)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import setup

TERMINAL = Path(__file__).resolve().parents[2]


class HipFlagsNameTheRootFirst(unittest.TestCase):
    def flags(self, relative: str) -> list[str]:
        text = (TERMINAL / relative).read_text(encoding="utf-8")
        return re.findall(r"CMAKE_HIP_FLAGS=([^'\n]*)'", text)

    def test_setup_py_has_the_engine_and_the_picture_reader_builds(self):
        # (PolyStrata) and strata-poly's Linux build; the Windows builds ({rocm}: TheRock's wheels) have no system HIP headers to avoid
        found = [f for f in self.flags("setup.py") if "{root}" in f]
        self.assertEqual(len(found), 3, found)
        for flags in found:
            self.assertTrue(flags.startswith('-isystem "{root}/include" --rocm-path="{root}"'), flags)

    def test_the_image_tool_builds_the_same_way(self):
        found = self.flags("tools/image/qimage.py")
        self.assertEqual(len(found), 1, found)
        self.assertTrue(found[0].startswith('-isystem "{rocm}/include" --rocm-path="{rocm}"'), found[0])

    def test_every_path_in_the_flags_is_quoted(self):
        # #7138: an unquoted {root}, {rocm} or {bitcode} splits at a space in the install folder
        found = self.flags("setup.py") + self.flags("tools/image/qimage.py")
        self.assertEqual(len(found), 6, found)
        for flags in found:
            self.assertNotRegex(flags, r'(?<!")\{(root|rocm|bitcode)\}', flags)


class BatArgKeepsAQuotedPathWhole(unittest.TestCase):
    def test_plain_arguments_stay_as_they_were(self):
        self.assertEqual(setup.bat_arg("-DSTRATA_ENABLE_HIP=ON"), "-DSTRATA_ENABLE_HIP=ON")
        self.assertEqual(setup.bat_arg(Path("C:/a b/cmake.exe")), '"' + str(Path("C:/a b/cmake.exe")) + '"')

    def test_quotes_inside_an_argument_are_escaped(self):
        arg = '-DCMAKE_HIP_FLAGS=--rocm-path="D:/Poly Strata/rocm" --rocm-device-lib-path="D:/Poly Strata/bitcode"'
        self.assertEqual(setup.bat_arg(arg),
                         '"-DCMAKE_HIP_FLAGS=--rocm-path=\\"D:/Poly Strata/rocm\\" --rocm-device-lib-path=\\"D:/Poly Strata/bitcode\\""')


if __name__ == "__main__":
    unittest.main()
