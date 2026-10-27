"""Exercise the actual Qt window outside Git, without starting any servers."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('PyQt5'), 'requires simulator extra')
class ControlGuiInstallationTests(unittest.TestCase):
    def test_window_and_log_paths_outside_git(self):
        from baldrapp.paranal_simulator import paths
        package_root = paths.CONFIG_DIR.parents[2]
        code = '''
from pathlib import Path
from unittest.mock import patch
import sys
from PyQt5 import QtWidgets
import subprocess
original_check_output = subprocess.check_output
def checked_output(args, *positional, **keywords):
    assert Path(args[0]).name != "git", "GUI must not invoke Git"
    return original_check_output(args, *positional, **keywords)
with patch("subprocess.check_output", side_effect=checked_output):
    from baldrapp.paranal_simulator.guis import baldr_sim_control_gui as gui
    assert gui.DEFAULT_LOG_PATH == Path(sys.argv[1])
    app = QtWidgets.QApplication([])
    with patch.object(gui.zmq.Context, "instance", side_effect=AssertionError("Unexpected network access")):
        window = gui.BaldrSimControlGui()
        window.log_timer.stop()
        window.status_timer.stop()
        assert window.log_path_edit.text() == sys.argv[1]
        window.show()
        app.processEvents()
        window.close()
'''
        with tempfile.TemporaryDirectory(prefix='baldr gui outside git ') as directory:
            root = Path(directory)
            for mode in ('default', 'xdg', 'override'):
                env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONPATH=str(package_root),
                           HOME=str(root), MPLCONFIGDIR=str(root / 'mpl'))
                env.pop('BALDR_STATE_DIR', None)
                env.pop('XDG_STATE_HOME', None)
                state = root / '.local/state/baldrapp'
                if mode in ('xdg', 'override'):
                    env['XDG_STATE_HOME'] = str(root / 'xdg')
                    state = root / 'xdg/baldrapp'
                if mode == 'override':
                    env['BALDR_STATE_DIR'] = str(root / 'custom state')
                    state = root / 'custom state'
                result = subprocess.run([sys.executable, '-c', code, str(state / 'logs/baldr_sim.log')],
                                        cwd=root, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, mode + '\n' + result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
