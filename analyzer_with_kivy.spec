# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['analyzer_with_kivy.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\\\Python311\\\\Lib\\\\site-packages\\\\kivy_deps\\\\glew', 'glew'), ('C:\\\\Python311\\\\Lib\\\\site-packages\\\\kivy_deps\\\\sdl2', 'sdl2')],
    hiddenimports=['kivy', 'kivy_deps.sdl2', 'kivy_deps.glew'],
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
    name='analyzer_with_kivy',
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
    icon=['favicon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=False,
    upx_exclude=[],
    name='analyzer_with_kivy',
)
