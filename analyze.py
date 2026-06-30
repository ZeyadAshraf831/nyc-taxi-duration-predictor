from taxi_predictor.analysis import run_analysis

def main():
    print("=" * 60)
    print("Residual Analysis + Feature Importance")
    print("=" * 60)

    results = run_analysis(sample_size=20000, save_selected=True)

    print(f"\nValidation sample R2:   {results['r2']:.3f}")
    print(f"Validation sample RMSE: {results['rmse']:.2f} seconds")
    print(f"\nReports saved to: {results['reports_dir']}")
    print(f"Importance table: {results['importance_path']}")

    print("\nResidual summary by distance bucket:")
    for bucket, stats in results["residual_summary"].items():
        print(
            f"  {bucket}: count={stats['count']}, "
            f"mean_error={stats['mean_error']:.1f}s, mae={stats['mae']:.1f}s"
        )

    print(f"\nSelected features ({len(results['selected_features'])}):")
    print(", ".join(results["selected_features"]))

    dropped = results["dropped_features"]
    if dropped:
        print(f"\nDropped weak features ({len(dropped)}):")
        print(", ".join(dropped))
    else:
        print("\nNo features dropped.")

    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()
