import { sites } from '@openai/sites-vite-plugin';
import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig, type Plugin } from 'vite';
import { createReadStream } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import hostingConfig from './.openai/hosting.json' with { type: 'json' };

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  '00000000-0000-4000-8000-000000000000';

const { d1, r2 } = hostingConfig;

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === 'seatbelt';
const localGz2Images = process.env.GZ2_IMAGE_DIR
  ? resolve(process.env.GZ2_IMAGE_DIR)
  : fileURLToPath(
      new URL(
        '../../../data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images/',
        import.meta.url,
      ),
    );

const localGz2Plugin: Plugin = {
  name: 'local-gz2-images',
  configureServer(server) {
    server.middlewares.use('/gz2-all', (request, response, next) => {
      const filename = (request.url || '').split('?')[0].replace(/^\/+/, '');
      if (!/^\d+\.jpg$/.test(filename)) return next();
      response.setHeader('Content-Type', 'image/jpeg');
      response.setHeader('Cache-Control', 'public, max-age=86400');
      if (request.method === 'HEAD') return response.end();
      createReadStream(resolve(localGz2Images, filename))
        .on('error', () => {
          response.statusCode = 404;
          response.end();
        })
        .pipe(response);
    });
  },
};

const localBindingConfig = {
  main: 'vinext/server/fetch-handler',
  compatibility_flags: ['nodejs_compat'],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: 'site-creator-d1',
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: 'site-creator-r2',
        },
      ]
    : [],
};

export default defineConfig(async () => {
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= 'false';
  process.env.WRANGLER_LOG_PATH ??= '.wrangler/logs';
  process.env.MINIFLARE_REGISTRY_PATH ??= '.wrangler/registry';

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import('@cloudflare/vite-plugin');

  return {
    css: { postcss: { plugins: [tailwindcss()] } },
    server: isCodexSeatbeltSandbox
      ? { watch: { useFsEvents: false, usePolling: true } }
      : undefined,
    plugins: [
      localGz2Plugin,
      vinext(),
      sites(),
      cloudflare({
        viteEnvironment: { name: 'rsc', childEnvironments: ['ssr'] },
        config: localBindingConfig,
      }),
    ],
  };
});
