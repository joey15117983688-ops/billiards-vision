"""Create a local macOS app bundle that launches this checkout's virtual environment."""
from pathlib import Path
import os
import plistlib
import shlex
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
APP=Path.home()/'Desktop'/'台球视觉识别.app'
IDENTIFIER='local.joey.billiards-vision'


def main():
    existing=APP/'Contents'/'Info.plist'
    if APP.exists() and (not existing.exists() or plistlib.loads(existing.read_bytes()).get('CFBundleIdentifier')!=IDENTIFIER):
        raise RuntimeError(f'目标已存在且不是本项目启动器：{APP}')
    source=ROOT/'assets'/'macos'/'logo.png'
    with tempfile.TemporaryDirectory() as temporary:
        icons=Path(temporary)/'Billiards.iconset';icons.mkdir()
        for size in (16,32,128,256,512):
            for scale in (1,2):
                target=icons/f'icon_{size}x{size}{"@2x" if scale==2 else ""}.png'
                subprocess.run(['/usr/bin/sips','-z',str(size*scale),str(size*scale),str(source),'--out',str(target)],check=True,stdout=subprocess.DEVNULL)
        icon=ROOT/'assets'/'macos'/'Billiards.icns'
        subprocess.run(['/usr/bin/iconutil','-c','icns',str(icons),'-o',str(icon)],check=True)
    contents=APP/'Contents';resources=contents/'Resources';binary=contents/'MacOS'
    resources.mkdir(parents=True,exist_ok=True);binary.mkdir(exist_ok=True)
    shutil.copy2(icon,resources/'Billiards.icns')
    info={'CFBundleName':'台球视觉识别','CFBundleDisplayName':'台球视觉识别',
          'CFBundleIdentifier':IDENTIFIER,'CFBundleVersion':'1','CFBundleShortVersionString':'1.0',
          'CFBundlePackageType':'APPL','CFBundleExecutable':'BilliardsLauncher',
          'CFBundleIconFile':'Billiards.icns','NSHighResolutionCapable':True,
          'LSMinimumSystemVersion':'12.0'}
    (contents/'Info.plist').write_bytes(plistlib.dumps(info))
    launcher=binary/'BilliardsLauncher'
    launcher.write_text('#!/bin/bash\nset -euo pipefail\nPROJECT_DIR='+shlex.quote(str(ROOT))+'''\ncd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"
exec >>"$PROJECT_DIR/logs/desktop_launcher.log" 2>&1
export QT_MAC_WANTS_LAYER=1
exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/main.py" --source video
''')
    launcher.chmod(0o755)
    (contents/'PkgInfo').write_bytes(b'APPL????')
    os.utime(APP,None)
    print(APP)


if __name__=='__main__':main()
