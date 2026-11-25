import pandas as pd
from src.orchestrator import PipelineOrchestrator
from src.config import COUNTRY_PROFILES

def generate_dummy_target(index):
    """Generates dummy target returns for demonstration since we don't have real price files yet."""
    import numpy as np
    return pd.Series(np.random.randn(len(index)), index=index)

def main():
    countries = list(COUNTRY_PROFILES.keys())
    all_results = []

    for country in countries:
        print(f"\nProcessing {country}...")
        orch = PipelineOrchestrator(country_code=country)

        # Load macro data first to get the index for the dummy target
        try:
            macro_data = orch.loader.load_data()
        except Exception as e:
            print(f"Skipping {country}: Failed to load initial data. Error: {e}")
            continue

        target_df = pd.DataFrame({"returns": generate_dummy_target(macro_data.index)})

        # Run pipeline with PCA
        print(f"Running PCA-based pipeline for {country}...")
        res_pca = orch.run_pipeline(target_df, use_autoencoder=False)
        if res_pca is None: continue

        # Run pipeline with Autoencoder
        print(f"Running AE-based pipeline for {country}...")
        res_ae = orch.run_pipeline(target_df, use_autoencoder=True)
        if res_ae is None: continue

        # Aggregate metrics
        for method in ["RandomForest", "LSTM"]:
            all_results.append({
                "Country": country,
                "FeatureExtraction": "PCA",
                "Model": method,
                "MAE": res_pca["metrics"][method]["MAE"],
                "R2": res_pca["metrics"][method]["R2"]
            })
            all_results.append({
                "Country": country,
                "FeatureExtraction": "Autoencoder",
                "Model": method,
                "MAE": res_ae["metrics"][method]["MAE"],
                "R2": res_ae["metrics"][method]["R2"]
            })

    # Create comparison table
    comparison_df = pd.DataFrame(all_results)
    print("\n=== Final Comparative Study Results ===")
    print(comparison_df.to_string())

    # Save results
    comparison_df.to_csv("finalproject/results/comparative_study.csv", index=False)
    print("\nResults saved to finalproject/results/comparative_study.csv")

if __name__ == "__main__":
    main()
