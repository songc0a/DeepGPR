unifont-subset.woff
===================

A subset of GNU Unifont 15.1 (https://unifoundry.com/unifont/) used as the
website's pixel typeface. It contains Basic Latin, Latin-1, Greek, general
punctuation, super/subscripts, arrows, mathematical operators, box drawing,
block elements, geometric shapes and a few symbols (about 1,250 glyphs).

Created with fontTools:

  pyftsubset unifont.otf --flavor=woff --output-file=unifont-subset.woff \
    --unicodes="U+0020-007E,U+00A0-00FF,U+0131,U+0152-0153,U+0370-03FF,
    U+2010-2027,U+2030-205F,U+2070-209F,U+20AC,U+2122,U+2190-21FF,
    U+2200-22FF,U+2300-232B,U+2500-259F,U+25A0-25FF,U+2600-266F,U+2713-2717"

License: GNU Unifont is Copyright (C) 1998-2023 Roman Czyborra, Paul Hardy,
Qianqian Fang, Andrew Miller, Johnnie Weaver, David Corbett, Nils Moskopp,
Rebecca Bettencourt, Minseo Lee, Ho-Seok Ee, et al. It is dual-licensed under
the SIL Open Font License 1.1 and the GNU GPL version 2 or later with the GNU
Font Embedding Exception. This file is distributed under the SIL OFL 1.1
(https://openfontlicense.org). It is not covered by the repository's MIT
license.
