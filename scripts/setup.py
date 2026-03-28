#!/usr/bin/env python3
"""
Setup script for fin-agent
"""
import subprocess
import sys


def run_command(cmd, description):
    """Run command and print output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Success: {description}")


def main():
    """Run setup steps"""
    print("🚀 Setting up fin-agent...")

    # Create virtual environment if not exists
    import os
    if not os.path.exists("venv"):
        run_command(
            "python3 -m venv venv",
            "Creating virtual environment"
        )

    # Activate and install
    venv_bin = "venv/bin" if os.name != "nt" else "venv/Scripts"

    # Install Python dependencies
    run_command(
        f"{venv_bin}/pip install -r requirements.txt",
        "Installing Python dependencies"
    )

    # Download NLTK data
    run_command(
        f"{venv_bin}/python3 -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"",
        "Downloading NLTK data"
    )

    # Create necessary directories
    run_command(
        "mkdir -p data/processed/chunks indexes embeddings logs",
        "Creating directories"
    )

    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. source venv/bin/activate  # Activate virtual environment")
    print("2. cp .env.example .env && vim .env  # Configure environment")
    print("3. python3 scripts/build_index.py  # Build search indexes")
    print("4. streamlit run ui/streamlit_app.py  # Start the UI")


if __name__ == "__main__":
    main()
