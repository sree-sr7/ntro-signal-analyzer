# Step 82 — IEEE 802.11n QC-LDPC Research Note

## Why IEEE 802.11n QC-LDPC material was selected

The initial repository and environment audit found no verified IEEE 802.11n
LDPC matrices. The environment pins generic `ldpc==2.4.1` and `komm==0.36.0`,
but their installed contents did not provide the nine project-specific
prototypes. A prior Step 82 review stopped at that point rather than creating
unverified matrices. External research was then used to locate the protocol
matrix definitions.

IEEE P802.11n/D11.0 Annex R provides the exact prototype family needed for the
project's three block lengths. Its compact QC representation uses 24-column
prototypes lifted by Z=27, 54, or 81, and the draft defines the selected rates
and information lengths. This gives a traceable source for the nine requested
configurations and allows H to be generated deterministically. The choice is a
scope and provenance decision; no comparative performance or superiority claim
is made.

The IEEE draft is the primary protocol and matrix-definition source available
for this implementation. It is explicitly an unapproved draft, subject to
change, and the draft itself says it must not be used for conformance/compliance
purposes. This project therefore does not claim compliance with the final IEEE
802.11n standard or implement the complete 802.11n PPDU coding process.

## Matrix source and cross-check

The explicit prototypes in `core/fec/ldpc_decoder.py` map to:

| Codeword N | Rate | Information K | Checks M=N-K | Z | Primary matrix table |
| ---: | ---: | ---: | ---: | ---: | --- |
| 648 | 1/2 | 324 | 324 | 27 | Annex R, Table R.1(a) |
| 648 | 2/3 | 432 | 216 | 27 | Annex R, Table R.1(b) |
| 648 | 3/4 | 486 | 162 | 27 | Annex R, Table R.1(c) |
| 1296 | 1/2 | 648 | 648 | 54 | Annex R, Table R.2(a) |
| 1296 | 2/3 | 864 | 432 | 54 | Annex R, Table R.2(b) |
| 1296 | 3/4 | 972 | 324 | 54 | Annex R, Table R.2(c) |
| 1944 | 1/2 | 972 | 972 | 81 | Annex R, Table R.3(a) |
| 1944 | 2/3 | 1296 | 648 | 81 | Annex R, Table R.3(b) |
| 1944 | 3/4 | 1458 | 486 | 81 | Annex R, Table R.3(c) |

The 78 selected prototype rows were parsed from the IEEE draft and compared
entry-by-entry with the corresponding tables in the implementation: all rows
had 24 entries and there were zero differences. The nine prototypes were also
compared exactly with the `H_648_*`, `H_1296_*`, or `H_1944_*` arrays in
`tavildar/LDPC/LdpcM/LDPCCode.m`; there were zero differences. The right-shift
convention matched as well. In the draft, a vacant table entry is a
zero submatrix and integer i denotes P_i, an identity matrix whose columns are
cyclically shifted right by i. tavildar's lifting code uses the corresponding
row-to-column map `(row + i) mod Z`. The repository is used only as an
independent implementation cross-check. Its README expressly warns that its
implementation is not necessarily specification compliant and omits protocol
behaviors such as puncturing, padding, and stream parsing.

The project's tests independently validate all nine prototype dimensions,
lifted H dimensions, binary entries, systematic encoder outputs, and zero
syndromes. These checks validate the local implementation against its stored
tables; they do not turn the unapproved draft into final-standard conformance
evidence.

## Implementation and framing

- `LDPCConfig` records the configuration name, N, K, rate, Z, prototype, H
  dimensions, IEEE draft reference ID, independent cross-check reference ID,
  and matrix identifier.
- H is generated from its prototype as an `(N-K) x N` binary matrix. For each
  nonnegative shift i, its Z by Z block has ones at
  `H[r*Z+t, c*Z+(t+i) mod Z]`; -1 generates an all-zero block.
- The encoder places K information bits in the first K codeword positions and
  solves `H[:, K:] p = H[:, :K] u` over GF(2). The rightmost parity submatrix
  is inverted over GF(2) per configuration and cached; each output is checked
  against H before it is returned. No arbitrary generator matrix is used.
- The decoder uses flooding normalized min-sum in the LLR domain. Positive
  LLRs favor bit 0, negative LLRs favor bit 1, and magnitudes affect
  variable-to-check and check-to-variable messages. The default normalization
  is 0.8 and the default iteration limit is 50.
- A decode succeeds only after the full generated matrix produces a zero
  syndrome. The result includes syndrome status/weight, iterations, N, K,
  rate, algorithm, decoded information bits, and failure diagnostics.
- A codeword API accepts exactly N LLRs; a frame API accepts a positive exact
  multiple of N and processes each codeword independently. The encoder accepts
  exactly K payload bits per word. No padding, shortening, puncturing,
  truncation, or partial-word decoding is done.
- The existing trial engine accepts hard bits, so its LDPC candidates map 0 to
  +8.0 and 1 to -8.0 before min-sum decoding. Callers with channel reliability
  data should call `decode_ldpc` or `decode_ldpc_frame` with the measured LLRs.
  The trial engine evaluates one no-interleaver candidate per LDPC configuration
  and requires an expected payload length to confirm exact structure.
- IEEE 802.11n rate 5/6 is not included. It appears in the draft but is outside
  this project's nine-configuration scope. PPDU stream parsing, shortening,
  puncturing, and interleaving/stream processing are also outside this slice.

## Source roles and limitations

The IEEE draft is the primary evidence for the matrix tables and lifting
definition. tavildar/LDPC is an independently authored software reference used
for cross-checking the nine table values and implementation conventions; it is
not evidence of full IEEE compliance. Peyic et al. is secondary technical
literature used to contextualize the code family and LLR/min-sum method, not to
source exact matrix entries. Richardson and Urbanke's *Modern Coding Theory*
was not found in existing project research files and was not consulted for
this implementation. The simgunz repository was linked from tavildar's README
but was not directly consulted.

## Sources used in Step 82

1. IEEE 802.11 Working Group, *IEEE P802.11n/D11.0*, June 2009, Section
   20.3.11.6.4 and Annex R Tables R.1–R.3. [Draft PDF](https://people.iith.ac.in/tbr/teaching/docs/802.11n-DraftStd_June2009.pdf).
   Primary draft source; explicitly not the final standard.
2. tavildar/LDPC, `LdpcM/LDPCCode.m` and README. [GitHub repository](https://github.com/tavildar/LDPC).
   Independent table/lifting cross-check only; README disclaims full
   specification compliance.
3. M. Peyic, H. Baba, E. Guleyuboglu, I. Hamzaoglu, and M. Keskinöz, “A low
   power multi-rate decoder hardware for IEEE 802.11n LDPC codes,”
   *Microprocessors and Microsystems*, vol. 36, no. 3, pp. 159–166, May 2012,
   DOI 10.1016/j.micpro.2011.12.006. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0141933111001281).
   Secondary technical literature.

Full metadata and reusable BibTeX are in [`research_references.md`](research_references.md)
and [`references.bib`](references.bib).
