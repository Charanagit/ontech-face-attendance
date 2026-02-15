# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['recognize_webcam.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\USER\\Desktop\\FR5\\venv_311\\Lib\\site-packages\\mediapipe', 'mediapipe'), ('C:\\Users\\USER\\.insightface\\models\\buffalo_s', 'insightface\\models\\buffalo_s'), ('OnTech.png', '.')],
    hiddenimports=['insightface', 'onnxruntime', 'onnxruntime.providers', 'mediapipe', 'cv2'],
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
    name='recognize_webcam',
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
    icon=['C:\\Users\\USER\\Desktop\\FR5\\OnTech.ico'],
)
