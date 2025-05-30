block_cipher = None

from PyInstaller.utils.hooks import collect_submodules  # noqa: E402
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT  # noqa: E402

# Include all hidden imports that PySide6 may pull in dynamically
hidden_imports = collect_submodules("PySide6")

# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

a = Analysis(
    ["massive_injector.py"],  # main script
    pathex=["."],  # search path for local imports
    binaries=[],
    datas=[
        ("labels.json", "."),
        ("dtico3.ico", "."),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

# ----------------------------------------------------------------------------
# Python modules archive (PYZ) and executable (EXE)
# ----------------------------------------------------------------------------

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="massive_injector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI-only; switch to True to keep a console
    icon="dtico3.ico",  # Add an .ico path here if desired
    uac_admin=True,  # set to True if admin privileges are needed
)

# ----------------------------------------------------------------------------
# Bundle everything into the dist/injector/ folder (onedir build)
# ----------------------------------------------------------------------------

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="massive_injector",  # output folder name (dist/massive_injector
)
