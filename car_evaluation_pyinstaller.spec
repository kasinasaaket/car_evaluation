# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['car_evaluation_pyinstaller.py'],
    pathex=[],
    binaries=[],
    datas=[('car_evaluation.csv', '.'), ('logreg_model.pkl', '.'), ('decision_tree_model.pkl', '.'), ('mlp_model.pkl', '.'), ('gs_logreg_model.pkl', '.'), ('gs_tree_model.pkl', '.'), ('gs_mlp_model.pkl', '.')],
    hiddenimports=[],
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
    a.binaries,
    a.datas,
    [],
    name='car_evaluation_pyinstaller',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
