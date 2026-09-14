# Datasets and attribution

Cosmic Detective uses Galaxy Zoo 2 (GZ2) morphology classifications and SDSS
galaxy imagery. Dataset files are downloaded locally and are not included in the
project's source-code license.

## Galaxy Zoo 2

The local preparation scripts expect these source files under
`data/raw/galaxy-zoo-2/`:

- `gz2_hart16.csv` — updated volunteer vote fractions and counts
- `gz2_filename_mapping.csv` — image asset IDs mapped to SDSS object IDs
- `kaggle-images/images_gz2/images/*.jpg` — GZ2 image cutouts

Galaxy Zoo publishes the GZ2 catalog and citation guidance on its
[data-release site](https://data.galaxyzoo.org/). The image release is linked
there through [Zenodo](https://doi.org/10.5281/zenodo.3565489). Work using these
materials should cite Willett et al. (2013) and, for the updated classifications,
Hart et al. (2016).

The source pages request citation but do not state that the project may relicense
the images or tables under MIT. Keep source data out of the repository unless
you have separately confirmed its redistribution terms.

## Generated web catalog

`pipelines/ingest/build_web_catalog.py` joins a small local fixture to vote
counts, computes a compact brightness descriptor, and writes:

- `apps/web/client/public/catalog.json`
- `apps/web/client/public/galaxies/*.jpg`

`pipelines/ingest/build_archive_pool.py` creates
`apps/web/client/public/archive-pool.json.gz` for the optional full-catalog
shuffle. During local development, Vite serves the corresponding JPEGs from the
GZ2 image directory. Set `GZ2_IMAGE_DIR` if those images are stored elsewhere.

These generated outputs are ignored by Git. The 200-object catalog supports the
app's reference gallery; it is not a held-out scientific evaluation set.

## Label limits

The app reduces GZ2's branching vote tree to a two-label demonstration. “Spiral”
requires visible spiral evidence; “elliptical” means smooth-looking for this
task. Artifacts, edge-on systems, mergers, weak images, and ambiguous examples
do not fit cleanly into that binary contract.
