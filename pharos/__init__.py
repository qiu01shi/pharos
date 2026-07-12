"""pharos: typed dataflow runtime for LLM workflows."""

from pharos.env import load_dotenv

# Load ~/.pharos/.env so providers (imported later) can read keys.
# Done at package import time; shell env still wins.
load_dotenv()

__version__ = "0.3.0"
