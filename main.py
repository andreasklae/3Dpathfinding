# main.py
import run_and_export

def build_and_visualize(length: int):
    _, html_path = run_and_export.run_and_export(
        length=length, density=0.25, seed=41, auto_open_html=True
    )
    return html_path

if __name__ == "__main__":
    LENGTH = 100
    html = build_and_visualize(LENGTH)
    print(f"Done. HTML written to: {html}")