import os
import medmnist
import torch
# Import modular code
from Code.Model_A.svm_classifier import run_svm
from Code.Model_B.cnn_classifier import run_cnn

def main():
    # ==========================================
    # 1. Path Configuration
    # ==========================================
    # Get the directory where the current script is located.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the dataset path (leave it blank when submitting the assignment)
    data_path = os.path.join(base_dir, 'Datasets')
    
    # Define the result save path (where the pictures are saved)
    results_dir = os.path.join(base_dir, 'Results')
    
    # Automatically create non-existent folders
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print(f"Project initialized.")
    print(f"Dataset Path: {data_path}")
    print(f"Results Path: {results_dir}")
    
    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Information on using the medmnist library
    info = medmnist.INFO['breastmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    try:
        print("Loading/Downloading data...")
        # download=True: If there are no files in the folder, download them; if there are files, read them directly.
        train_data = DataClass(split='train', download=True, root=data_path)
        val_data = DataClass(split='val', download=True, root=data_path)
        test_data = DataClass(split='test', download=True, root=data_path)
        
        # Extract data for model use.
        x_train, y_train = train_data.imgs, train_data.labels.ravel()
        x_val, y_val = val_data.imgs, val_data.labels.ravel()
        x_test, y_test = test_data.imgs, test_data.labels.ravel()
        
        print(f"Data Loaded Successfully: Train {x_train.shape}, Test {x_test.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your internet connection or the 'Datasets' folder.")
        return

    # ==========================================
    # 3. Run Models
    # ==========================================
    
    # ---  Model A: SVM ---
   
    try:
        run_svm(x_train, y_train, x_test, y_test, results_dir)
    except Exception as e:
        print(f"Error running SVM: {e}")
    
    # ---  Model B: CNN ---
    # Detect whether there is a GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CNN on device: {device}")
    
    try:
        # Note: Here, results_dir is also passed in.
        run_cnn(x_train, y_train, x_val, y_val, x_test, y_test, device, results_dir)
    except Exception as e:
        print(f"Error running CNN: {e}")

# ==========================================
# 4. Entry Point
# ==========================================
if __name__ == "__main__":
    main()