import os

def audit_dl_evaluations():
    print("--- Evaluation Engineer (Deep Learning Audit) ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reports_dir = os.path.join(base_dir, "reports")
    
    expected_reports = [
        "forecasting_report.md",
        "vision_evaluation_report.md"
    ]
    
    print("\nAuditing Deep Learning Evaluation Reports:\n")
    for report in expected_reports:
        report_path = os.path.join(reports_dir, report)
        if os.path.exists(report_path):
            print(f"[PASSED] Found {report}.")
            # Print a snippet of the metrics
            with open(report_path, "r") as f:
                content = f.read()
                print("   Snippet:")
                for line in content.split("\n"):
                    if "Accuracy" in line or "F1 Score" in line or "MAE" in line or "Precision" in line or "Recall" in line:
                        print(f"      {line.strip()}")
        else:
            print(f"[FAILED] Missing {report}. Please run the respective dl_engineer pipeline.")
            
if __name__ == "__main__":
    audit_dl_evaluations()
