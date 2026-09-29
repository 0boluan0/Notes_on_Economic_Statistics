#!/usr/bin/env python3
"""Deploy the reviewed bridge and local user-entered credentials over SSH stdin."""
import base64
import argparse
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import base64, json, os, pathlib, plistlib, subprocess, sys
os.umask(0o077)
bundle=json.load(sys.stdin)
state=pathlib.Path.home()/'Library/Application Support/StudentOSTelegram'
state.mkdir(parents=True, exist_ok=True, mode=0o700)
os.chmod(state,0o700)
config=state/'config.json'
if config.exists():
    previous=json.loads(config.read_text())
    if previous.get('token') != bundle['config']['token']:
        raise SystemExit('Existing remote bot differs; no configuration changed')
else:
    with config.open('x') as f:
        os.chmod(config,0o600)
        json.dump(bundle['config'],f)
app=state/'app'
app.mkdir(exist_ok=True,mode=0o700)
for name, encoded in bundle['files'].items():
    if name not in ('bridge.py','telegram_api.py','reconciliation.py','voice.py'):
        raise SystemExit('Unexpected bundle filename')
    p=app/name
    p.write_bytes(base64.b64decode(encoded))
    os.chmod(p,0o600)
vault=pathlib.Path.home()/'Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic'
if not (vault/'AGENTS.md').is_file():
    raise SystemExit('Expected Academic vault is unavailable')
label='local.student-os.telegram'
codex=pathlib.Path('/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex')
if not codex.is_file():
    raise SystemExit('The verified app-bundled Codex executable is unavailable')
plist=pathlib.Path.home()/'Library/LaunchAgents'/ (label+'.plist')
plist.parent.mkdir(parents=True,exist_ok=True)
settings={'Label':label,'ProgramArguments':['/opt/homebrew/bin/python3','-B',str(app/'bridge.py'),'--vault',str(vault),'--codex',str(codex),'serve'],
          'WorkingDirectory':str(vault),'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':30,
          'EnvironmentVariables':{'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONUNBUFFERED':'1'},
          'StandardOutPath':str(state/'service.log'),'StandardErrorPath':str(state/'service-error.log')}
unchanged = plist.exists() and plistlib.loads(plist.read_bytes()) == settings
plist.write_bytes(plistlib.dumps(settings))
os.chmod(plist,0o600)
domain='gui/'+str(os.getuid())
target=domain+'/'+label
registered = subprocess.run(['launchctl','print',target],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
if registered and unchanged:
    subprocess.run(['launchctl','kickstart','-k',target],check=True,stdout=subprocess.DEVNULL)
else:
    if registered:
        subprocess.run(['launchctl','bootout',target],check=True,stdout=subprocess.DEVNULL)
    # A just-unloaded launchd label can take a moment to become bootstrappable.
    import time
    for attempt in range(3):
        outcome=subprocess.run(['launchctl','bootstrap',domain,str(plist)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if outcome.returncode == 0:
            break
        time.sleep(1)
    else:
        raise SystemExit('LaunchAgent bootstrap failed; configuration preserved')
print(json.dumps({'installed':True,'service':label,'bot_username':bundle['config'].get('bot_username')}))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-username',required=True)
    parser.add_argument('--host', choices=('macmini','macmini-zt'), default='macmini')
    args=parser.parse_args()
    state=Path.home()/'Library/Application Support/StudentOSTelegram'
    config=json.loads((state/'config.json').read_text())
    if config.get('bot_username') != args.expected_username.lstrip('@'):
        raise SystemExit('Bot username does not match the requested bot')
    root=Path(__file__).resolve().parent
    files={name:base64.b64encode((root/name).read_bytes()).decode() for name in ('bridge.py','telegram_api.py','reconciliation.py','voice.py')}
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',args.host,
                           '/opt/homebrew/bin/python3 -c '+shlex.quote(REMOTE)],
                          input=json.dumps({'config':config,'files':files}),text=True,
                          capture_output=True,timeout=60)
    if result.returncode:
        print('Deployment did not complete; remote state requires inspection (no credentials printed).')
        raise SystemExit(1)
    print(result.stdout.strip())


if __name__=='__main__':
    main()
