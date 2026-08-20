# Copyright 2026 Edgar Chávez and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build config for the graft._core extension (mirrors src/Makefile logic:
-O3 native arch; OpenMP on Linux/gcc; libomp via Homebrew on macOS clang,
serial fallback otherwise)."""

import platform
import subprocess
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra_compile = ["-O3", "-funroll-loops"]
extra_link = []

if platform.system() == "Darwin":
    # NEON is baseline on arm64; -mcpu=native probed by the compiler itself.
    try:
        prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], text=True,
            stderr=subprocess.DEVNULL).strip()
        extra_compile += ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"]
        extra_link += [f"-L{prefix}/lib", "-lomp"]
    except (OSError, subprocess.CalledProcessError):
        print("graft-ann: libomp not found ('brew install libomp' enables the "
              "parallel build); compiling SERIAL", file=sys.stderr)
else:
    extra_compile += ["-march=native", "-fopenmp"]
    extra_link += ["-fopenmp"]

setup(
    ext_modules=[
        Pybind11Extension(
            "graft._core",
            ["python/graft_core.cpp", "src/build.cpp", "src/search.cpp"],
            include_dirs=["src"],
            cxx_std=17,
            extra_compile_args=extra_compile,
            extra_link_args=extra_link,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
