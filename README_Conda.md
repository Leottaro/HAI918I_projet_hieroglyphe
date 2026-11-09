# Installer Glyphnet avec Conda sous Linux et macOS

Choisissez le fichier qui correspond à votre système. Les deux environnements utilisent Python 3.11 et les mêmes versions de base des bibliothèques du projet.

| Système | Fichier | Périphérique utilisé par Glyphnet |
| --- | --- | --- |
| Mac Apple Silicon sous macOS 14 ou ultérieur | [`environment-macos.yml`](environment-macos.yml) | MPS si disponible, sinon CPU |
| Linux x86_64 ou aarch64, y compris Asahi Linux | [`environment-linux.yml`](environment-linux.yml) | CUDA avec une carte NVIDIA et un pilote compatibles, sinon CPU |

## 1. Installer Conda si nécessaire

Dans un terminal, vérifiez si Conda est déjà disponible sur votre machine.

```bash
conda --version
```

Si la commande fonctionne, passez à l’étape suivante. Sinon, vérifiez l’architecture avec `uname -m`, puis téléchargez l’installateur correspondant à votre système sur la [page de Miniforge](https://conda-forge.org/download/).

Depuis le dossier du fichier téléchargé, lancez la commande correspondant à votre système. Acceptez l’initialisation du shell à la fin de l’installation, puis ouvrez un nouveau terminal.

```bash
# Linux
bash "Miniforge3-Linux-$(uname -m).sh"

# macOS sur Apple Silicon
bash Miniforge3-MacOSX-arm64.sh
```

## 2. Créer l’environnement du projet

Depuis la racine du dépôt local, choisissez un des deux environnements.

### macOS

```bash
conda env create -f environment-macos.yml
conda activate glyphnet-macos
```

### Linux

```bash
conda env create -f environment-linux.yml
conda activate glyphnet-linux
```

L’environnement Linux installe la version CUDA 13.0 de PyTorch et ses bibliothèques d’exécution, même sur les machines où l’entraînement se fait sur CPU. Pour utiliser l’accélération NVIDIA, vérifiez avec `nvidia-smi` que la carte est détectée et que le pilote est au moins en version 580.65.06. [Compatibilité CUDA et pilotes NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

Conda installe Python et pip, puis pip installe les six bibliothèques utilisées par le projet et leurs dépendances indirectes.

Dans un nouveau terminal, réactivez l’environnement choisi. Dans VS Code, sélectionnez-le via **Python: Select Interpreter**.

## 3. Vérifier l’installation

```bash
python --version
python -m pip check
python -c "import torch, torchvision, torchinfo, PIL, sklearn, matplotlib; print(torch.__version__, torchvision.__version__)"
PYTHONPATH=src python -c "from config import device; print(device)"
```

Les versions attendues sont `2.14.1` et `0.29.1` sous macOS, et `2.14.1+cu130` et `0.29.1+cu130` sous Linux. Depuis la racine du dépôt, la dernière commande affiche le périphérique que Glyphnet utilisera. Le résultat peut être `mps` ou `cpu` sous macOS, et `cuda` ou `cpu` sous Linux.

Sur Mac, `True` à la commande suivante indique que Glyphnet peut utiliser MPS avec l’environnement actif.

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Sous Linux, si le périphérique affiché est `cuda`, vous pouvez vérifier que PyTorch effectue un calcul sur la carte NVIDIA.

```bash
python -c "import torch; print(torch.ones(1, device='cuda') + 1)"
```

Le résultat attendu est `tensor([2.], device='cuda:0')`.