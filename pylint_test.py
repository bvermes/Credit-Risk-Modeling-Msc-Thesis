import os
import subprocess


def list_python_files(directory):
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def generate_pylint_reports():
    directory_to_search = os.path.dirname(__file__)
    output_directory = os.path.join(directory_to_search, "testing", "pylint")
    os.makedirs(output_directory, exist_ok=True)

    python_files = list_python_files(directory_to_search)

    for file in python_files:
        output_file = os.path.join(
            output_directory, os.path.basename(file) + ".pylint.txt"
        )
        subprocess.run(
            [
                "pylint",
                file,
                "--output-format=text",
                "--reports=n",
                "--rcfile=/dev/null",
                "--disable=all",
                "--enable=unused-import,unused-variable,unused-argument,invalid-name",
                "--exit-zero",
            ],
            stdout=open(output_file, "w"),
        )

    print("Pylint analysis completed. Results saved in:", output_directory)


def generate_class_diagram():
    project_directory = "credit-risk-models"  # Name of your project directory
    parent_directory = os.path.abspath(
        os.path.join(project_directory, os.pardir)
    )  # Get the absolute path of the parent directory
    subprocess.run(
        ["pyreverse", "-o", "png", "-p", "TMP", project_directory],
        cwd=parent_directory,
        check=True,
    )


def main():

    print("Console Menu:")
    print("1. Generate pylint reports")
    print("2. Generate class diagram")

    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        generate_pylint_reports()
    elif choice == "2":
        generate_class_diagram()
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
