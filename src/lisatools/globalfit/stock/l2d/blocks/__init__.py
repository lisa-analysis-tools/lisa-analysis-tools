from __future__ import annotations

from typing import Annotated

from pydantic import Field

from . import emri, mbh, psd
from .base import BlockBuild, BuildContext, SourceBlock

BlockSpec = Annotated[
    psd.PSDBlock | mbh.MBHBlock | emri.EMRIBlock, Field(discriminator="type")
]
# block type -> builder(block, ctx); the source blocks share SourceBlock.build_source
BUILDERS: dict[str, callable] = {
    "psd": psd.build,
    "mbh": SourceBlock.build_source,
    "emri": SourceBlock.build_source,
}
