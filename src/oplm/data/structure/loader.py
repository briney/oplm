"""Structure loading.

Parses PDB/mmCIF files into :class:`StructureData` (sequence plus backbone
N/CA/C coordinates) for the structure eval modality. Biopython is lazy-imported
inside the parsing functions and is only required under the ``train`` extra.
"""

from __future__ import annotations
