"""Twin that corrects the residual of a frozen LR + MF base (design: "Twin sửa phần dư của nền LR + MF").

Modules T0-T9 of the design, section 4.2. Only T3 and T5 call an LLM; T2 uses a text embedder offline.
"""

__version__ = "0.2.0"
