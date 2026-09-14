# Security policy

## Reporting a vulnerability

Please use the repository host's private vulnerability-reporting feature. If it
is unavailable, contact the maintainer privately rather than opening a public
issue. Include the affected version, reproduction steps, and likely impact.

## Secrets and deployment

- Keep inference credentials in server-side environment variables.
- Never commit `.env` files, access tokens, model credentials, or cloud job
  state.
- Add authentication and rate limiting before exposing a deployment that uses a
  billable inference endpoint to untrusted users.
- Treat uploaded images as untrusted input and retain the existing type, size,
  and pixel-count checks when changing the upload flow.

The maintainers will acknowledge a complete report as soon as practical and
coordinate disclosure after a fix is available.
