# -*- mode: python ; coding: utf-8 -*-
#
# Build:  pyinstaller MattePro.spec        (from this directory, in the venv)
#
import os
from PyInstaller.utils.hooks import copy_metadata

# NiceGUI ships templates, JS and CSS beside its Python, and reads them from
# disk at runtime, so the whole package is copied in as data rather than being
# left to the module graph. Located via import instead of a hardcoded venv path
# so the spec survives a Python upgrade or a different machine.
import nicegui
_nicegui_dir = os.path.dirname(nicegui.__file__)

datas = [(_nicegui_dir, 'nicegui')]

# MattePro reads shinestacker's version through importlib.metadata to warn when
# it is too old to carry alpha (see _shinestacker_status). Dist metadata is not
# bundled by default, and without it that check cannot report a version.
datas += copy_metadata('shinestacker')

a = Analysis(
    ['timelapse_matte_ng.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    # shinestacker is now an ordinary installed dependency
    # (pip install "shinestacker>=1.17.0"), so PyInstaller traces
    # `from shinestacker import PyramidAutoStack` itself and pulls in the bulk
    # of the tree — including matplotlib, which shinestacker imports
    # unconditionally at module scope.
    #
    # Only genuinely dynamic imports need declaring here: these are reached
    # through lazy or plugin-style loading that static analysis cannot see.
    hiddenimports=[
        'psdtags',        # shinestacker/algorithms/multilayer.py, loaded on demand
        'imagecodecs',    # tifffile compression backend, resolved at runtime
        'scipy.ndimage',
        'scipy.signal',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MattePro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['MattePro.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MattePro',
)
app = BUNDLE(
    coll,
    name='MattePro.app',
    icon='MattePro.icns',
    bundle_identifier='com.pislider.mattepro',
)
