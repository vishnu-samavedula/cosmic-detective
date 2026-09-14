# Cosmic Detective web client

The web client is a Vinext application with a server-side inference proxy. It
compares a base 450M vision model with a morphology-tuned checkpoint, displays
request timing, ranks local reference images, and stores collected cards in the
browser.

Use the repository-level [README](../../../README.md) for setup, data
preparation, security boundaries, and licensing.

## Commands

```sh
npm install
npm run dev
npm run lint
npm run build
```

`public/catalog.json`, `public/archive-pool.json.gz`, and `public/galaxies/` are
generated local data. Build them with the scripts in `pipelines/ingest/`; they
are excluded from version control.

The interface background is
[NASA’s Webb dark-matter map](https://www.flickr.com/photos/nasawebbtelescope/55118095230/),
credited to NASA/STScI/J. DePasquale/A. Pagan and licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is displayed with
responsive cropping and a dark color overlay.
