import pandas as pd
from sklearn.model_selection import train_test_split

# import kaggle

# # Make sure to replace 'mssmartypants/rice-typeclassification' with the correct dataset path
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files('mssmartypants/rice-typeclassification', path='D:/code-test/1/', unzip=True)

# Local path to the downloaded file
file_path = "D:/code-test/1/riceClassification.csv"  # Adjust path as necessary
data = pd.read_csv(file_path)
