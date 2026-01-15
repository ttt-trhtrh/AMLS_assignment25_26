import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV  # New: Grid Search
from skimage.feature import hog

def run_svm(x_train, y_train, x_test, y_test, save_dir):
    print("\n" + "="*50)
    print("Model A: SVM Comparison (Raw vs HOG with GridSearch)")
    print("="*50)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ==========================================
    # Phase 1: Baseline (Raw Pixels)
    # ==========================================
    print("\n[Phase 1] Training Baseline: Raw Pixels...")
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    scaler_raw = StandardScaler()
    x_train_raw = scaler_raw.fit_transform(x_train_flat)
    x_test_raw = scaler_raw.transform(x_test_flat)
    
    # Raw pixels usually do not require complex tuning; keeping defaults 
    # or performing a simple search is sufficient.
    svm_raw = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_raw.fit(x_train_raw, y_train)
    
    y_pred_raw = svm_raw.predict(x_test_raw)
    acc_raw = accuracy_score(y_test, y_pred_raw)
    print(f"Baseline (Raw) Accuracy: {acc_raw:.4f}")

    # ==========================================
    # Phase 2: Proposed (HOG Features) - Optimized Version
    # ==========================================
    print("\n[Phase 2] Training Proposed: HOG Features...")

    # 1. Attempt finer HOG parameters
    # pixels_per_cell=(4, 4) is relatively balanced. If results are poor, 
    # keep parameters fixed and tune SVM instead.
    def extract_hog(images):
        feats = []
        for img in images:
            fd = hog(img, orientations=9, pixels_per_cell=(4, 4),
                     cells_per_block=(2, 2), visualize=False)
            feats.append(fd)
        return np.array(feats)

    print("Extracting HOG features...")
    x_train_hog = extract_hog(x_train)
    x_test_hog = extract_hog(x_test)
    
    scaler_hog = StandardScaler()
    x_train_hog_scaled = scaler_hog.fit_transform(x_train_hog)
    x_test_hog_scaled = scaler_hog.transform(x_test_hog)

    # 2. Key Step: GridSearchCV (Automatically find the best SVM parameters)
    print("Running Grid Search to optimize SVM for HOG...")
    
    # Define the parameter grid to try
    param_grid = {
        'C': [0.1, 1, 10, 100],           # Penalty parameter
        'gamma': ['scale', 0.1, 0.01, 0.001], # Kernel coefficient
        'kernel': ['rbf', 'poly']         # Try different kernels
    }
    
    # Start search (n_jobs=-1 means using all CPU cores for acceleration)
    grid = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=1, cv=3, n_jobs=-1)
    grid.fit(x_train_hog_scaled, y_train)
    
    print(f"Best Parameters found: {grid.best_params_}")
    
    # Predict using the best model found
    best_svm_hog = grid.best_estimator_
    y_pred_hog = best_svm_hog.predict(x_test_hog_scaled)
    acc_hog = accuracy_score(y_test, y_pred_hog)
    
    print(f"Proposed (HOG Optimized) Accuracy: {acc_hog:.4f}")
    
    # ==========================================
    # Phase 3: Results Display
    # ==========================================
    diff = acc_hog - acc_raw
    print("\n" + "-"*30)
    print(f"Improvement: {diff*100:.2f}%")
    print("-" * 30)

    # Generate table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    rec_raw = recall_score(y_test, y_pred_raw)
    rec_hog = recall_score(y_test, y_pred_hog)
    sign = "+" if diff >= 0 else ""
    
    table_data = [
        ['Raw Pixels', f"{acc_raw:.4f}", f"{rec_raw:.4f}", "-"],
        [f'HOG (Optimized)', f"{acc_hog:.4f}", f"{rec_hog:.4f}", f"{sign}{diff:.4f}"]
    ]
    
    table = ax.table(cellText=table_data, colLabels=['Model', 'Accuracy', 'Recall', 'Improvement'], 
                     loc='center', cellLoc='center')
    table.scale(1.2, 1.8)
    
    plt.title(f"Comparison: Raw vs HOG (Grid Search Optimized)")
    save_path = os.path.join(save_dir, 'Model_A_Comparison_Optimized.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[SUCCESS] Comparison Table saved to: {save_path}")
    plt.close()

    # Regenerate Success Cases (based on the optimized model)
    indices = [i for i in range(len(y_test)) if y_pred_raw[i] != y_test[i] and y_pred_hog[i] == y_test[i]]
    if len(indices) > 0:
        idx = indices[0]
        img = x_test[idx]
        _, hog_vis = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.imshow(img, cmap='gray'); ax1.set_title('Original')
        ax2.imshow(img, cmap='gray'); ax2.set_title('Raw (Wrong)', color='red')
        ax3.imshow(hog_vis, cmap='gray'); ax3.set_title('HOG (Correct)', color='blue')
        plt.savefig(os.path.join(save_dir, 'Model_A_Success_Cases_Optimized.png'), bbox_inches='tight', dpi=300)
        plt.close()

    return acc_hog