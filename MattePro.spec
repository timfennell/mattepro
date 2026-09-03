# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['timelapse_matte_ng.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/timothyfennell/Documents/paintbot/venv/lib/python3.14/site-packages/nicegui', 'nicegui')],
    # shinestacker is loaded at RUNTIME from ~/Documents/shinestacker-alpha/src
    # (see SHINESTACKER_SRC), so PyInstaller cannot statically trace its imports
    # and will not bundle its dependencies. They must be declared explicitly or
    # stacking fails in the frozen app with "No module named ...". Determined by
    # tracing sys.modules across `from shinestacker import PyramidAutoStack`.
    hiddenimports=[
        'jsonpickle',    # shinestacker/config/settings.py
        'psdtags',       # shinestacker/algorithms/multilayer.py
        'scipy',         # shinestacker algorithms
        'scipy.ndimage',
        'scipy.signal',
        'tqdm',          # shinestacker/core/{logging,core_utils}.py
        'imagecodecs',   # shinestacker/algorithms/multilayer.py
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
    icon=['/Users/timothyfennell/Documents/tilt correction/MattePro.icns'],
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
    icon='/Users/timothyfennell/Documents/tilt correction/MattePro.icns',
    bundle_identifier='com.pislider.mattepro',
)
