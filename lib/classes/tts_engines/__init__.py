import sys

from .xtts import XTTS
from .bark import Bark
from .vits import Vits
from .fairseq import Fairseq
from .tortoise import Tortoise
from .glowtts import GlowTTS
from .tacotron import Tacotron2
from .piper import Piper
from .yourtts import YourTTS

if sys.version_info >= (3, 12):
    from .omnivoice import OmniVoiceEngine
