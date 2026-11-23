from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize('bindir,libdir', [('bin', 'lib'), ('bin', 'lib64'), ('tools/bin', 'lib64')])
def test_modelhost_rpath_matches_install_directories(tmp_path, bindir, libdir):
    project = tmp_path / 'CMakeLists.txt'
    project.write_text(f'''cmake_minimum_required(VERSION 3.20)
project(RuntimePathCheck LANGUAGES CXX)
set(CMAKE_INSTALL_BINDIR "{bindir}" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_LIBDIR "{libdir}" CACHE PATH "" FORCE)
include(GNUInstallDirs)
add_library(inference INTERFACE)
add_library(FoamNordic::inference ALIAS inference)
set(FOAMNORDIC_ONNXRUNTIME_LIBRARY "${{CMAKE_CURRENT_BINARY_DIR}}/libonnxruntime.so")
add_subdirectory("{ROOT / 'tools/resident'}" resident)
file(GENERATE OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/runtime-rpath.txt"
    CONTENT "$<TARGET_PROPERTY:foamnordic_model_worker,INSTALL_RPATH>")
''')
    build = tmp_path / 'build'
    subprocess.run(['cmake', '-S', str(tmp_path), '-B', str(build)], check=True, capture_output=True, text=True)

    origin = '@loader_path' if sys.platform == 'darwin' else '$ORIGIN'
    assert (build / 'runtime-rpath.txt').read_text() == f'{origin}/{os.path.relpath(libdir, bindir)}'
