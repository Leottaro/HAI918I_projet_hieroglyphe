'''a test file to see if all works'''

from collections import Counter
from pathlib import Path

from config import *
from dataset import *

from glyphnet import *


def main() -> None:
    torch.manual_seed(CONFIG.seed)
    
    df: pd.DataFrame = pd.read_csv(DATA_DIR / "Tableau_imagette_MB.csv")
    df = df[~df["Deno"].str.match(R"^.* \([0-9]+\)$")] # filter to keep single sign images
        
    models: list[str] = df.columns[1:].array
        
    for model_name in models:       
        df2 = df[df[model_name].notna()]
    
        image_paths: list[Path] = [DATA_DIR / (deno.strip() +".png") for deno in df2["Deno"].array]
        
        labels: list[str] = df2[model_name].array
        num_labels: int = len(set(labels))
        
        try:
            class_to_idx, train_loader, train_eval_loader, val_loader, test_loader = read_dataset(image_paths, labels)
        except:
            continue

        print(f"\n\n\n\nModel: {model_name}")
        print(f"Images: {len(image_paths)}")
        print(f"Labels: {num_labels}")
        print(f"class -> index: {class_to_idx}")
        print(f"\nTrain set numbers: {len(train_loader.dataset.samples)}")
        print(f"Val set numbers: {len(val_loader.dataset.samples)}")
        print(f"Test set numbers: {len(test_loader.dataset.samples)}")

        for images, labels in train_loader:
            print(f"\nImages shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Labels dtype: {labels.dtype}")
            break

        model = Glyphnet(num_labels).to(device)
        
        model.eval()
            
        x: Tensor = torch.randn(1, 1, 75, 50, device=device,)
        
        summary(
            model,
            input_data=x,
            col_names=("input_size", "output_size", "num_params"),
            depth=3,
            mode="eval",
        )


if __name__ == "__main__":
    main()