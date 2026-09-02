# PyInstaller entrypoint for the Windows AstroBatch V2 desktop build.
a = Analysis(['astrobatch/__main__.py'], pathex=['.'], hiddenimports=['PySide6'], datas=[])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, name='AstroBatchV2', console=False)
