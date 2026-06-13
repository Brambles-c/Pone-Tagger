from pathlib import Path
import os, dotenv

dotenv.load_dotenv()
images_path = Path(os.getenv('IMAGES_PATH'))
