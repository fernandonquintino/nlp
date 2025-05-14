import subprocess
import sys
from pathlib import Path
from src.ensure_resources import ensure_resources_files

# --- Project root based on where main.py is ---
project_root = Path(__file__).parent.resolve()
venv_path = project_root / "venv"
python_in_venv = venv_path / "Scripts" / "python.exe" if sys.platform == "win32" else venv_path / "bin" / "python"
requirements_file = project_root / "requirements.txt"

# --- Pipeline steps ---
scripts = [
    '1_label_target.py',
    '2_fine_tuning_label.py',
    '3_preprocessing.py',
    '4_lemmatization.py',
    '5_train_models.py',
    '6_predictions.py'
]

def run_command(command, cwd=project_root):
    print(f"\n Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode != 0:
        print(f" Command failed: {' '.join(command)}")
        sys.exit(1)

def ensure_venv():
    if not python_in_venv.exists():
        print(" Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    else:
        print(" Virtual environment already exists.")

def install_requirements():
    print(" Installing requirements...")
    run_command([str(python_in_venv), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(python_in_venv), "-m", "pip", "install", "-r", str(requirements_file)])

def run_pipeline_scripts():
    for script in scripts:
        run_command([str(python_in_venv), script])

if __name__ == "__main__":
    ensure_resources_files()
    ensure_venv()
    install_requirements()
    run_pipeline_scripts()
    print("\n All steps completed successfully using venv:", python_in_venv)
