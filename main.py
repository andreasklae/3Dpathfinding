# main.py
import sys
import run_and_export

def build_and_visualize(length: int):
    _, html_path = run_and_export.run_and_export(
        length=length, density=0.25, seed=41, auto_open_html=True
    )
    return html_path

if __name__ == "__main__":
    # default length
    length = 100

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            val = int(arg)
            if 10 <= val <= 1000:
                length = val
            else:
                print(f"Invalid length '{arg}'. Must be an integer between 10 and 1000. Using default {length}.")
        except ValueError:
            print(f"Invalid argument '{arg}'. Must be an integer between 10 and 1000. Using default {length}.")

    print("Creating", "50x50x" + length, "maze...")
    html = build_and_visualize(length)
    print(f"Done. HTML written to: {html}")