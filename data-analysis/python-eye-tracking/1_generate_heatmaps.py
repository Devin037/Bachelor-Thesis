import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- USER CONFIGURATION ---
FULL_DATA_FILE = "newest_combined_eyetracking_data.csv"
BACKGROUND_IMAGE_FILE = "carl.png"

def generate_heatmap(data_filepath, image_filepath, target_robot, target_difficulty):
    """
    Generates a high-visibility heatmap with a shorter, closer colorbar legend.
    """
    print(f"Generating heatmap for: {target_robot} / {target_difficulty}...")
    
    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv(data_filepath, decimal=',')
        bg_img = mpimg.imread(image_filepath)
        img_height, img_width, _ = bg_img.shape
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find a required file. {e}")
        return

    coord_cols = ['Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)']
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    condition_df = df[
        (df['Robot'] == target_robot) & 
        (df['difficulty'] == target_difficulty) &
        (df['Eye movement type'] == 'Fixation')
    ].dropna(subset=coord_cols).copy()
    
    if condition_df.empty:
        print("No fixation data found for the selected condition.")
        return
        
    print(f"Found {len(condition_df)} fixations for this condition.")

    condition_df['x_pixel'] = condition_df['Fixation point X (MCSnorm)'] * img_width
    condition_df['y_pixel'] = condition_df['Fixation point Y (MCSnorm)'] * img_height

    # --- 2. Create the Heatmap Plot ---
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Display the background image
    ax.imshow(bg_img)
    
    # --- UPDATED: Final adjustments for legend size and position ---
    sns.kdeplot(
        x=condition_df['x_pixel'],
        y=condition_df['y_pixel'],
        ax=ax,
        fill=True,
        cmap="rocket_r",
        alpha=0.75,
        thresh=0.05,
        bw_adjust=0.8,
        cbar=True,
        cbar_kws={
            'label': 'Fixation Density',
            'shrink': 0.4,    # --- REDUCED: Makes the colorbar even shorter (40% of plot height) ---
            'pad': 0.02       # --- ADDED: Moves the colorbar closer to the plot ---
        }
    )
    
    ax.set_title(f"Fixation Heatmap for: {target_robot} / {target_difficulty}", fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# --- Main Execution Block (Set to compare Carl: Easy vs. Hard) ---
if __name__ == "__main__":
    print("--- Generating heatmap for carl / easy ---")
    generate_heatmap(data_filepath=FULL_DATA_FILE,
                     image_filepath=BACKGROUND_IMAGE_FILE,
                     target_robot='Carl condition',
                     target_difficulty='easy')

    print("\n--- Generating a second heatmap for comparison ---")
    generate_heatmap(data_filepath=FULL_DATA_FILE,
                     image_filepath=BACKGROUND_IMAGE_FILE,
                     target_robot='Carl condition',
                     target_difficulty='hard')