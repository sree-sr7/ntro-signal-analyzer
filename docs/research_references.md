# Research and Reference Ledger

This ledger preserves the sources and scope decisions used in the project. It
was started for Step 82 on 2026-09-30. Before this step, the repository had no
`docs` directory, bibliography, or LDPC matrix research note. The README
contained only a general FEC roadmap entry; the Step 78 audit files record
repository state, not research sources. No pre-existing reference history was
found to consolidate or replace.

## Step 82: IEEE 802.11n QC-LDPC

### R1 — IEEE P802.11n/D11.0 draft

- **Title:** *IEEE P802.11n/D11.0, Draft Standard for Information Technology—Telecommunications and Information Exchange Between Systems—Local and Metropolitan Area Networks—Specific Requirements—Part 11: Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications, Amendment 5: Enhancements for Higher Throughput*.
- **Author(s):** IEEE 802.11 Working Group of the 802 Committee; no individual technical authors are identified on the cover.
- **Date:** June 2009.
- **Source type:** IEEE standards draft.
- **Venue:** IEEE P802.11 Working Group, D11.0 draft.
- **Stable source location:** [Draft PDF](https://people.iith.ac.in/tbr/teaching/docs/802.11n-DraftStd_June2009.pdf).
- **Exact technical claims used:** Annex R Tables R.1, R.2, and R.3 define the QC parity-check prototypes for N=648/Z=27, N=1296/Z=54, and N=1944/Z=81 respectively. Section 20.3.11.6.4 defines the lifted blocks as cyclically right-shifted identity matrices P_i or null matrices, with a table integer i selecting P_i. Table 20-14 gives the information lengths for rates 1/2, 2/3, 3/4, and 5/6 at those N values.
- **Evidence role:** Primary source for the chosen code family, information/codeword lengths, matrix prototypes, and lifting convention.
- **Limitation/disclaimer:** The cover says this is an unapproved draft subject to change and must not be used for conformance/compliance purposes. It is not treated as the final paid IEEE standard, and the project makes no full-standard compliance claim.
- **Implementation mapping:** `core/fec/ldpc_decoder.py` records `IEEE80211N-D11.0-2009` on every configuration. Its prototype entries map -1 to a zero block and each nonnegative i to `H[r*Z+t, c*Z+(t+i) mod Z] = 1`. Table R.1(a-c), R.2(a-c), and R.3(a-c) map to the nine supported configuration IDs.

### R2 — tavildar/LDPC implementation

- **Title:** *LDPC codes* (C and MATLAB implementations).
- **Author(s):** Not verified. `tavildar` is the repository owner name, not asserted here as an individual author.
- **Date:** Not verified. Repository content was reviewed on 2026-09-30; no immutable release or commit identifier was verified.
- **Source type:** Software repository and source code.
- **Venue:** GitHub, `tavildar/LDPC`.
- **Stable source locations:** [Repository README](https://github.com/tavildar/LDPC) and [`LdpcM/LDPCCode.m`](https://github.com/tavildar/LDPC/blob/master/LdpcM/LDPCCode.m). The `master` links are mutable; the revision reviewed is recorded by date above.
- **Exact technical claims used:** `LDPCCode.m` contains `H_648_*`, `H_1296_*`, and `H_1944_*` prototype tables, including the nine project configurations. Its `lifted_ldpc` method places a cyclic permutation at each nonnegative prototype entry, and the README describes Wi-Fi LDPC construction, back-substitution encoding, and iterative BP/min-sum decoding with an LLR-vector interface.
- **Evidence role:** Primary evidence for what this repository contains; independent secondary cross-check for the IEEE draft matrix values and lifting convention.
- **Limitation/disclaimer:** The README expressly says the code is not necessarily specification compliant and identifies omissions including puncturing, padding, and stream parsing. It is not used as proof of IEEE compliance, and its encoder/decoder implementation was not copied into this project.
- **Implementation mapping:** All nine selected prototype tables and the right-shift lifting convention were compared against R1. No table-value or lifting-convention discrepancy was found. The project derives its own encoder from H over GF(2) and implements its own normalized min-sum decoder.

### R3 — Peyic et al. research paper

- **Title:** “A low power multi-rate decoder hardware for IEEE 802.11n LDPC codes.”
- **Author(s):** Merve Peyic, Hakan Baba, Erdem Guleyuboglu, Ilker Hamzaoglu, and Mehmet Keskinöz.
- **Date:** May 2012; volume 36, issue 3, pages 159–166.
- **Source type:** Peer-reviewed research paper.
- **Venue:** *Microprocessors and Microsystems*.
- **Stable source locations:** [Publisher record and abstract](https://www.sciencedirect.com/science/article/pii/S0141933111001281), DOI [10.1016/j.micpro.2011.12.006](https://doi.org/10.1016/j.micpro.2011.12.006); author and venue metadata cross-checked against [DBLP](https://dblp.org/rec/journals/mam/PeyicBGHK12).
- **Exact technical claims used:** The publisher abstract and introduction describe the 12-code family over block lengths 648, 1296, and 1944 and rates 1/2, 2/3, 3/4, and 5/6, and describe layered min-sum decoding in the LLR domain.
- **Evidence role:** Secondary technical literature supporting the family-level and decoding-method context; not the authority for exact prototype values.
- **Limitation/disclaimer:** The paper is about a hardware implementation and does not establish project compliance or replace the draft tables.
- **Implementation mapping:** Used to contextualize the QC-LDPC family and LLR/min-sum choice. The project implements flooding normalized min-sum, so the paper's layered schedule is not claimed as the schedule used here.

## Initial repository and package audit

- The initial repository search found no IEEE 802.11n prototypes, matrix files, reference bibliography, or `docs` directory. The earlier Step 82 attempt stopped at this point instead of inventing a matrix.
- The pinned environment lists `ldpc==2.4.1` and `komm==0.36.0`. Inspection of the installed package contents found generic matrix/code utilities but no verified IEEE 802.11n prototype set for these nine configurations. That made the installed packages insufficient as the source of standard matrices.
- The new implementation does not depend on either package for matrix data or decoding. It stores the sourced prototypes explicitly, generates H from them, derives parity bits from the corresponding parity submatrix, and performs normalized min-sum updates directly from supplied LLRs.

## Research trail status

- **Richardson & Urbanke, *Modern Coding Theory*:** No copy or prior citation was found in repository research notes, and the book was not consulted for this design. It is not used as implementation evidence in Step 82.
- **simgunz/802.11n-ldpc:** The tavildar README links to this repository, but it was not opened or used for Step 82 cross-validation. It is not represented as a consulted source.

## Step 83B: Project-scoped diagonal interleaver definition

### Repository search and terminology decision

The repository search performed before implementation found no historical
`Diag_16x16_S1` mapping, diagonal traversal order, indexing convention, or
step rule. Step 83 was stopped at that gate. External research was then used
to resolve the ambiguity; the chosen definition and its exact index formula
are recorded in [`diagonal_interleaver_note.md`](diagonal_interleaver_note.md).

The project defines `Diag_16x16_S1` as a 16×16 matrix diagonal/helical-scan
interleaver with array step size `S=1`. `S1` means that the row index advances
by one for each column advance inside a diagonal. This is a project-scoped
adoption of a documented construction. It is not a historical repository
definition, an assertion that the name or permutation comes from an IEEE
standard, or a claim that it is the only possible diagonal-interleaver rule.

### R4 — MathWorks Matrix Helical Scan Interleaver

- **Title:** *Matrix Helical Scan Interleaver — Permute input symbols by selecting matrix elements along diagonals*.
- **Author(s):** The MathWorks, Inc.
- **Date:** Documentation page; publication date not stated. Reviewed 2026-09-30.
- **Source type:** Official product documentation.
- **Venue:** Simulink Communications Toolbox documentation.
- **Stable source location:** [MathWorks documentation](https://www.mathworks.com/help/comm/ref/matrixhelicalscaninterleaver.html).
- **Exact technical claims used:** The input fills a matrix row by row; output reads elements along diagonals of length equal to the number of columns; row and column indices both increase; each following diagonal begins one row below the prior diagonal's first element; Array Step Size is the row-index increase per column-index increase. The documented 6×4, step-1 example produces `[1, 6, 11, 16, 5, 10, 15, 20, 9, 14, 19, 24, 13, 18, 23, 4, 17, 22, 3, 8, 21, 2, 7, 12]` for input `[1:24]`.
- **Evidence role:** Primary source for the selected helical traversal convention and the meaning of step size 1. The project specializes this construction to 16×16 and writes its row-wrap behavior as an explicit modulo-16 permutation.
- **Limitation/scope:** This source documents a general construction, not the repository-specific mode name or a 16×16 project profile. The project definition is not represented as an IEEE-standard permutation.

### R5 — Li et al., CRCSI

- **Title:** “CRCSI: A Generic Block Interleaver for the Next Generation Terrestrial Broadcast Systems.”
- **Author(s):** Ruijia Li, Jinfeng Tian, Xin Bian, and Mingqi Li.
- **Date:** 15 February 2022.
- **Source type:** Peer-reviewed research article.
- **Venue:** *Applied Sciences*, 12(4), article 2025.
- **Stable source location:** [Publisher article](https://www.mdpi.com/2076-3417/12/4/2025); DOI [10.3390/app12042025](https://doi.org/10.3390/app12042025).
- **Exact technical claims used:** Section 2 identifies TBI as the diagonal interleaver used in ATSC 3.0. Equation (1) gives a general permutation mapping using a reading interval and a periodic cyclic-shift vector; Equation (2) expresses a block mapping through row/column mappings.
- **Evidence role:** Context for diagonal interleaver terminology and mathematical block-permutation notation. It does not define the exact 16×16 helical scan selected for this project.
- **Limitation/scope:** The CRCSI article's proposed construction and equations are not substituted for the MathWorks helical-scan rule, and no ATSC/IEEE compliance claim is made for the project mode.

## Step 84: Project-scoped pseudo-random interleaver

### Repository definition check

The interleaver source, trial engine, tests, and project documentation were
searched for a pseudo-random mode, seed, RNG, or permutation before this
implementation. No exact repository definition was present. The default mode
below is therefore a project-scoped adoption; it is not attributed to an
existing SIH requirement or a communications standard.

### R6 — Durstenfeld random permutation

- **Title:** “Algorithm 235: Random permutation.”
- **Author:** Richard Durstenfeld.
- **Date:** July 1964.
- **Source type:** Published algorithm note.
- **Venue:** *Communications of the ACM*, 7(7), p. 420.
- **Stable source location:** [ACM DOI record](https://doi.org/10.1145/364520.364540).
- **Exact technical claims used:** The descending in-place swap method that selects a bounded position for each remaining element and swaps it with the current position. The project does not use a library shuffle.
- **Implementation mapping:** The project applies the descending Durstenfeld rule to an identity array and uses unbiased rejection sampling to map SplitMix64 words to each bounded choice.
- **Limitation/scope:** This is a permutation-generation algorithm, not an interleaver standard or a prescribed communications profile.

### R7 — SplitMix64

- **Title:** “Fast Splittable Pseudorandom Number Generators.”
- **Authors:** Guy L. Steele Jr., Doug Lea, and Christine H. Flood.
- **Date:** October 2014.
- **Source type:** Peer-reviewed conference paper.
- **Venue:** OOPSLA ’14, pp. 453–472, Portland, Oregon.
- **Stable source locations:** [Author-hosted paper PDF](https://gee.cs.oswego.edu/dl/papers/oopsla14.pdf); DOI [10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195).
- **Exact technical claims used:** The paper describes deterministic seeded PRNG state, a 64-bit SplitMix sequence, and a 64-bit mixing function. The project pins the add/mix constants and unsigned 64-bit wraparound in source so the permutation does not depend on a platform library RNG.
- **Implementation mapping:** core/interleaver/pseudorandom.py uses the SplitMix64 add/mix recurrence, Durstenfeld descending swaps, and unbiased bounded draws. Default configuration is frame length 256 and seed 26147. Each permutation is applied independently to a complete frame; incomplete frames raise an error.
- **Limitation/scope:** This adopted pseudo-random block interleaver is not an IEEE, DVB, or SIH-specified mode. The paper's statistical claims are not treated as evidence about this interleaver's performance or suitability for an external radio protocol.

### Project-scoped profile

PseudoRandom_256_Seed_26147 is the explicit default. The full output-to-input
index array is available through PseudoRandomInterleaver.permutation and
pseudorandom_permutation_for. Tests include a literal 256-index reference
vector. Trial-engine candidates are added only when the received stream has
complete 256-bit frames and the downstream FEC/codeword framing is compatible.

## Bibliography

Machine-readable entries for the sources used in Steps 82, 83B, and 84 are maintained in
[`references.bib`](references.bib).

## Tier-2 independent baseline sources

These entries cover only external technical sources consulted for the Tier-2
generator and baseline methodology. GNU Radio documentation was inspected for
the preferred-tool assessment, but GNU Radio was not installed and no signal
was generated with it.

### T2-R1 — Komm convolutional-code reference

- **Title:** “ConvolutionalCode.”
- **Author/organization:** Komm project documentation.
- **Date:** Not stated on the consulted reference page; accessed 2026-09-30.
- **Source type:** Official software API reference.
- **Stable source location:** [Komm ConvolutionalCode reference](https://komm.dev/ref/ConvolutionalCode/).
- **What was used:** The separate `komm.ConvolutionalCode` encoder API and feedforward-polynomial convention for independent rate-1/2 convolutional encoding. The generator invokes Komm directly and does not import application FEC code.
- **Scope:** This establishes an external encoder implementation, not a claim of GNU Radio use or of conformance to an unspecified radio profile.

### T2-R2 — Reed-Solomon reference parameters

- **Title:** `reedsolo.py` source, Reed-Solomon codec implementation.
- **Author/organization:** `tomerfiliba-org/reedsolomon` project; source file metadata does not state a publication date.
- **Date:** Not stated; consulted against the installed `reedsolo` 1.7.0 behavior on 2026-09-30.
- **Source type:** Upstream software implementation reference.
- **Source location:** [reedsolo.py on the project repository](https://github.com/tomerfiliba-org/reedsolomon/blob/master/src/reedsolo/reedsolo.py).
- **What was used:** The RS(255,223) parity length and GF(256) parameter convention relevant to the independent local RS encoder: primitive polynomial `0x11d`, generator element 2, first consecutive root 0. The generator implements those operations locally and imports neither `reedsolo` nor `core.fec`.
- **Scope:** The source was used to identify the decoder's expected convention; generator independence is preserved through a separately written polynomial encoder.

### T2-R3 — CRC-16/CCITT-FALSE catalogue entry

- **Title:** “CRC-16/IBM-3740” (alias CRC-16/CCITT-FALSE).
- **Author/organization:** RevEng CRC Catalogue; individual author/date are not stated on the consulted entry.
- **Date:** Catalogue page reports last update 2024-12-11; accessed 2026-09-30.
- **Source type:** Technical CRC parameter catalogue.
- **Stable source location:** [RevEng CRC-16 catalogue](https://reveng.sourceforge.io/crc-catalogue/16.htm).
- **What was used:** Width 16, polynomial `0x1021`, initial value `0xffff`, no reflection, xorout zero. The Tier-2 generator computes these bits independently.
- **Scope:** The catalog entry supplies the named CRC parameters; it does not define the project's payload protocol.

### T2-R4 — NumPy inverse FFT reference

- **Title:** “numpy.fft.ifft.”
- **Author/organization:** NumPy developers.
- **Date:** The consulted versioned documentation page does not state a publication date; accessed 2026-09-30.
- **Source type:** Official software API reference.
- **Stable source location:** [NumPy 2.2 `ifft` reference](https://numpy.org/doc/2.2/reference/generated/numpy.fft.ifft.html).
- **What was used:** The inverse discrete Fourier transform API for constructing the independent OFDM test waveform.
- **Scope:** OFDM is an out-of-distribution waveform in this test. The reference supports the transform operation, not a claim that the capture represents an external OFDM standard profile.
