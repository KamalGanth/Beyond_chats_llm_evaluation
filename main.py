import os
from preprocess import run_preprocess
from evaluate import run_evaluation
from report import run_report

DATA_DIR = "data"

def list_json_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

def choose_file(prompt):
    files = list_json_files()
    print("\n" + prompt)
    for i, f in enumerate(files, start=1):
        print(f"{i}. {f}")
    choice = int(input("Enter choice number: ").strip())
    return os.path.join(DATA_DIR, files[choice - 1])

def main():
    print("\n=== LLM Evaluation Pipeline (Auto File Selector) ===")

    # STEP 1 — select conversation file
    conv_path = choose_file("Select a CONVERSATION file:")
    print("Selected conversation file:", conv_path)

    # STEP 2 — select context file
    ctx_path = choose_file("Select a CONTEXT-VECTORS file:")
    print("Selected context file:", ctx_path)

    # STEP 3 — Run preprocessing
    print("\nStep 1: Preprocessing JSON files...")
    processed_json = run_preprocess(conv_path, ctx_path)

    # STEP 4 — Ask evaluation method
    print("\nChoose Evaluation Mode:")
    print("1 → TF-IDF")
    print("2 → Embedding")
    print("3 → LLM Judge (Gemini)")
    mode_sel = input("Enter choice (1/2/3): ").strip()

    if mode_sel == "1":
        method = "tfidf"
    elif mode_sel == "2":
        method = "embedding"
    elif mode_sel == "3":
        method = "judge"
    else:
        raise ValueError("Invalid choice")

    # STEP 5 — Evaluation
    print(f"\nStep 2: Running evaluation in '{method.upper()}' mode...")
    eval_path = run_evaluation(method, input_path=processed_json)

    # STEP 6 — Generate final report
    print("\nStep 3: Generating final report...")
    report_path = run_report(input_path=eval_path)

    print("\n=== PIPELINE COMPLETE ===")
    print("Processed Input File: ", processed_json)
    print("Evaluation Result File:", eval_path)
    print("Final Report File    :", report_path)


if __name__ == "__main__":
    main()
