// Runs apify/page_function.js outside Apify with a minimal cheerio-scraper-like context.
// Usage: node harness.mjs <cases.json>  -> prints [{enqueued:[...], result:{...}}] as JSON.
// Each case: {userData, url, contentType: "json"|"html", body}
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import * as cheerio from 'cheerio';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '..', '..', 'apify', 'page_function.js'), 'utf8');
const pageFunction = new Function(`${src}; return pageFunction;`)();

const cases = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const c of cases) {
    const enqueued = [];
    const ctx = {
        request: { url: c.url, userData: c.userData },
        log: { info() {} },
        enqueueRequest: async (r) => { enqueued.push({ url: r.url, userData: r.userData }); },
        body: c.body,
        json: c.contentType === 'json' ? JSON.parse(c.body) : undefined,
        $: c.contentType === 'html' ? cheerio.load(c.body) : undefined,
    };
    const result = await pageFunction(ctx);
    out.push({ enqueued, meta: result.meta, payload: result.payload });
}
process.stdout.write(JSON.stringify(out));
