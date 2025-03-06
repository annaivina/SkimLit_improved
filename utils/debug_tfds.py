import tensorflow_datasets as tfds
import os
from skimlit_dataset import skimlit_dataset

data_dir = "/Users/annushka/Desktop/SkimLit_improved/data/data_tfds_200k/"

print("Checking if dataset exists at:", data_dir)
if os.path.exists(data_dir):
    print("✅ Directory exists. Contents:")
    print(os.listdir(data_dir))
else:
    print("❌ ERROR: Dataset directory does NOT exist!")

# Try manually calling download_and_prepare()
print("\n🔍 Running SkimlitDataset manually...")
try:
    builder = skimlit_dataset.SkimlitDataset(data_dir=data_dir)
    builder.download_and_prepare()
    print("✅ Dataset should now be created!")
except Exception as e:
    print(f"❌ ERROR: {e}")