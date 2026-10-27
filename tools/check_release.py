"""Validate the paired release archives without starting simulator servers."""
from email.parser import BytesParser
from hashlib import sha256
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "release-artifacts"
VERSION = "0.1.9"
CORE = DIST / f"baldrapp-{VERSION}-py3-none-any.whl"
NATIVE = DIST / f"baldrapp_simulator_native-{VERSION}-py3-none-manylinux_2_39_x86_64.whl"
SDIST = DIST / f"baldrapp-{VERSION}.tar.gz"
with zipfile.ZipFile(CORE) as wheel:
    names = wheel.namelist()
    metadata = BytesParser().parsebytes(wheel.read(f"baldrapp-{VERSION}.dist-info/METADATA"))
    requirements = metadata.get_all("Requires-Dist")
    assert metadata['Version'] == VERSION
    assert 'simulator' in metadata.get_all('Provides-Extra')
    for dependency in ('baldrapp-simulator-native', 'xaosim', 'shmview', 'asgard_lab_DM_tools', 'pyqt5', 'pyqtgraph', 'pyzmq'):
        found = [r for r in requirements if r.lower().replace('_','-').startswith(dependency.lower().replace('_','-'))]
        assert found and all('extra == "simulator"' in r for r in found), (dependency, found)
    for required in ('paranal_simulator/paths.py', 'paranal_simulator/scripts/heimbal_simulation_servers.sh',
                     'paranal_simulator/fake_configs/baldr_config.json', 'paranal_simulator/fake_configs/cred1_split.json',
                     'data/Exposed_Ma-N_1405_optical_constants.txt'):
        assert 'baldrapp/' + required in names, required
    for name in names:
        assert '/apps/' not in name and '/things_purged/' not in name, name
        assert not name.endswith(('.tar.gz', '.zip', '.pyc')), name
        assert not wheel.read(name).startswith(b'\x7fELF'), name
with zipfile.ZipFile(NATIVE) as wheel:
    tag = wheel.read(f'baldrapp_simulator_native-{VERSION}.dist-info/WHEEL').decode()
    assert 'Root-Is-Purelib: false' in tag
    assert 'Tag: py3-none-manylinux_2_39_x86_64' in tag
    for name in ('shm_creator_sim', 'sim_mdm_server'):
        entry = wheel.getinfo(f'baldrapp_simulator_native/bin/{name}')
        assert (entry.external_attr >> 16) & 0o111, name
        original = ROOT / 'src/baldrapp/paranal_simulator/scripts' / name
        assert sha256(wheel.read(entry)).digest() == sha256(original.read_bytes()).digest(), name
with tarfile.open(SDIST) as archive:
    names = [name.partition('/')[2] for name in archive.getnames()]
    assert 'tests/test_installed_runtime.py' in names
    for name in names:
        assert not name.startswith(('baldrapp/', 'native/', '.release-venv/', '.smoke-venv/')), name
        assert not name.endswith(('/shm_creator_sim', '/sim_mdm_server', '.tar.gz', '.whl', '.pyc')), name
DIST.joinpath('SHA256SUMS').write_text(''.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in (CORE, NATIVE, SDIST)))
print('Both wheels and core sdist passed content, dependency, permissions, and binary-identity checks.')
