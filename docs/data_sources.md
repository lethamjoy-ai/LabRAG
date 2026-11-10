# Public reference collection plan

Start with [Mayo Clinic Laboratories Test Catalog](https://www.mayocliniclabs.com/test-catalog), [Quest Diagnostics Test Directory](https://testdirectory.questdiagnostics.com/), and [MedlinePlus Medical Tests](https://medlineplus.gov/lab-tests/). These are source discovery entry points, not evidence that a particular document has been collected or approved for reuse.

No application scraping or automatic downloading is implemented. Review each site's current terms and document permissions. Manually locate a relevant reference, obtain permission where necessary, and save only material permitted for this educational project. Record the actual document title, laboratory/provider, exact source URL, test name/code, revision date if available, and access date. Do not invent test-specific URLs, revision dates, or clinical requirements.

The 15 requested topics appear in `data/planned_topics.csv` with status `planned` and empty source URLs. After manual collection, upload each document and enter its provenance; only successfully saved documents belong in `data/source_manifest.csv`. Keep laboratory-specific documents separate, retain conflicting statements and original units, and never merge conflicting instructions into an unsupported universal rule.

Before sharing the repository, inspect manifests and generated assets for attribution and accidental private information. Public accessibility alone does not imply permission to redistribute. Synthetic development files are explicitly labeled and do not provide clinical requirements.
