import subprocess
import sys
from pathlib import Path

def run_heimbal():
    """Wrapper to run the heimbal_simulation_servers.sh bash script."""
    # Resolve the path to the bash script relative to this file
    cli_dir = Path(__file__).resolve().parent
    script_path = cli_dir / "scripts" / "heimbal_simulation_servers.sh"
    
    if not script_path.exists():
        print(f"Error: Could not find script at {script_path}")
        sys.exit(1)
        
    # Run the bash script and pass along any command line arguments (like 'start' or 'stop')
    # sys.argv[1:] captures anything typed after the command
    result = subprocess.run([str(script_path)] + sys.argv[1:])
    
    # Exit with the same code the bash script returned
    sys.exit(result.returncode)