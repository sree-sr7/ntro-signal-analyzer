# Diag_16x16_S1 project definition

`Diag_16x16_S1` is defined for this project as a 16×16 matrix diagonal/helical
interleaver with array step size `S=1`, based on the documented MathWorks
Matrix Helical Scan Interleaver construction. A repository search before this
implementation found no historical S1 permutation, traversal order, or index
mapping. This is therefore a project-scoped adoption from external research,
not a recovered repository definition.

This definition does not claim that the exact name or permutation comes from
an IEEE standard, nor that it is the only possible diagonal-interleaver
construction.

## Exact forward permutation

The mode operates independently on each complete 256-bit block. For block
number `b`, input bits fill a matrix row-major:

```text
X[r, c] = bits[b * 256 + 16 * r + c]
```

where `0 <= r,c < 16`. The diagonals are emitted with starting rows
`d = 0, 1, ..., 15`. Each diagonal starts at `(d, 0)`, one row below the
previous diagonal's start. Within a diagonal, `c` increases from 0 to 15,
while the row increases by the step `S=1` and wraps modulo 16:

```text
output[b * 256 + 16 * d + c] = X[(d + c) mod 16, c]
```

Thus, row-major input index `i = 16*r + c` maps to row-major output index

```text
permutation[i] = 16 * ((r - c) mod 16) + c
```

The output block order is the input block order. There is no padding,
truncation, or partial-block behavior: input length must be a positive exact
multiple of 256 bits. The inverse uses the exact inverse permutation.

## Research basis and scope

MathWorks documents row-wise matrix filling, diagonal output, both row and
column indices increasing along a diagonal, each next diagonal beginning one
row below the previous start, and array step size as the row-index increase
per column. Its 6×4 step-1 example is consistent with the traversal convention
adopted here. The project fixes that documented construction at 16×16 and
step 1.

Li et al. identify TBI as a diagonal interleaver in ATSC 3.0 and give a
general block-interleaver mapping formulation using a reading interval and a
cyclic-shift vector. That paper is recorded as context for diagonal/block
permutation notation; it is not the source of the exact 16×16 helical mapping
above.

References: [MathWorks Matrix Helical Scan Interleaver](https://www.mathworks.com/help/comm/ref/matrixhelicalscaninterleaver.html);
Li et al., [CRCSI: A Generic Block Interleaver for the Next Generation
Terrestrial Broadcast Systems](https://doi.org/10.3390/app12042025), *Applied
Sciences* 12(4), article 2025 (2022).
