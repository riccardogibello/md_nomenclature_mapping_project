import subprocess
import sys
import os


# Check if GPU is available
def is_gpu_available():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_and_activate_venv(
        venv_name='.venv'
) -> None:
    # Create virtual environment
    subprocess.run([sys.executable, '-m', 'venv', venv_name])

    # Activate virtual environment
    if os.name == 'nt':  # Windows
        python_executable = os.path.join(venv_name, 'Scripts', 'python.exe')
    else:  # Unix or MacOS
        python_executable = os.path.join(venv_name, 'bin', 'python')

    # Install setuptools and wheel in the virtual environment
    subprocess.run(
        [
            python_executable, '-m', 'pip', 'install',
            'setuptools',
            'wheel'
        ],
        check=True
    )

    # Install the requirements from requirements.txt
    subprocess.run(
        [
            python_executable, '-m', 'pip', 'install',
            '-r', 'requirements.txt'
        ],
        check=True
    )

    if is_gpu_available():
        print('GPU is available. Installing PyTorch with CUDA support.')
        subprocess.run(
            [python_executable,
             '-m', 'pip', 'install',
             'torch',
             'torchvision',
             'torchaudio',
             '--index-url', 'https://download.pytorch.org/whl/cu118'
             ],
            check=True
        )
    else:
        print('GPU is not available. Installing PyTorch without CUDA support.')
        subprocess.run(
            [python_executable,
             '-m', 'pip', 'install',
             'torch==2.4.1',
             'torchvision==0.19.1',
             'torchaudio==2.4.1'
             ],
            check=True
        )


if __name__ == '__main__':
    create_and_activate_venv()
