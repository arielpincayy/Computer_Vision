import os
from urllib.parse import quote

GITHUB_USER = "eugeniomorocho"
REPO_NAME = "Computer_Vision"
BRANCH = "main"

BASE_GITHUB_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}/blob/{BRANCH}"

START = "<!-- START NOTEBOOK LIST -->"
END = "<!-- END NOTEBOOK LIST -->"


def github_badge(url):
    safe_url = quote(url, safe="")
    return f"[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-181717?logo=github)]({url})"


def clean_name(filename):
    return os.path.splitext(filename)[0].replace("_", " ")


def build_section():
    sections = {}

    for root, dirs, files in os.walk("."):
        if root.startswith("./.") or ".github" in root:
            continue

        notebooks = sorted(f for f in files if f.endswith(".ipynb"))
        if notebooks:
            section = root.replace("./", "")
            sections[section] = notebooks

    content = []

    for section in sorted(sections.keys()):
        content.append(f"\n### 📁 {section}\n")

        for nb in sections[section]:
            name = clean_name(nb)
            path = os.path.join(section, nb).replace("\\", "/")
            url = f"{BASE_GITHUB_URL}/{path}"
            badge = github_badge(url)

            content.append(f"- **{name}**  \n  {badge}")

    return "\n".join(content)


with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

before = readme.split(START)[0]
after = readme.split(END)[1]

new_section = build_section()

new_readme = before + START + "\n" + new_section + "\n" + END + after

with open("README.md", "w", encoding="utf-8") as f:
    f.write(new_readme)

print("README actualizado con lista de notebooks.")
