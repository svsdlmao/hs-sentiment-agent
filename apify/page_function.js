// pageFunction for the official Apify actor `apify/cheerio-scraper`.
//
// It is sent as the actor's `pageFunction` input by hsagent/ingest/apify_runner.py.
// Responsibilities (kept deliberately small):
//   1. return the raw page payload (JSON body, or the Telegram message HTML) with its userData;
//   2. follow pagination until the post timestamps fall behind the look-back cutoff;
//   3. for LIHKG, enqueue the reply page of the most-discussed matching threads.
// Field parsing happens in Python (hsagent/ingest/parsers.py) so it is unit-tested once
// and identical for the Apify path and the local fallback crawler.
// The pagination rules mirror hsagent/ingest/paginate.py (see tests/test_page_function_parity.py).
async function pageFunction(context) {
    const { request, log } = context;
    const ud = request.userData || {};
    const kind = ud.kind;

    // Keep only the fields the parser uses; busy boards return ~750 pages and the raw
    // rich-text JSON would make the dataset ~10x larger. Mirrors paginate.slim_naver().
    function slimNaver(p) {
        const res = (p && p.result) || {};
        const keep = ['id', 'writtenAt', 'title', 'contentSwReplacedButImg', 'recommendCount', 'notRecommendCount',
            'viewCount', 'commentCount', 'replyCount', 'isHolderVerified', 'isCleanbotPassed'];
        const posts = (res.posts || []).map(x => {
            const o = {};
            for (const k of keep) if (x[k] !== undefined) o[k] = x[k];
            o.writer = { nickname: (x.writer || {}).nickname, isHolderVerified: (x.writer || {}).isHolderVerified };
            return o;
        });
        return { result: { lastOffset: res.lastOffset, posts } };
    }

    function nextUrl(cursor) {
        return ud.nextTemplate.replace('{cursor}', encodeURIComponent(String(cursor)));
    }

    function nextRequests(payload, tg) {
        const out = [];
        const page = ud.page || 1;
        const more = page < (ud.maxPages || 1);
        if (kind === 'naver_list') {
            const res = (payload && payload.result) || {};
            const posts = res.posts || [];
            if (posts.length && more && res.lastOffset) {
                const oldest = Math.min(...posts.map(p => Date.parse(p.writtenAt + '+09:00') / 1000));
                if (oldest > ud.cutoffTs) out.push({ url: nextUrl(res.lastOffset), userData: { ...ud, page: page + 1 } });
            }
        } else if (kind === 'lihkg_search') {
            const items = (payload && payload.response && payload.response.items) || [];
            if (items.length >= 30 && more) {
                const oldest = Math.min(...items.map(i => i.create_time));
                if (oldest > ud.cutoffTs) out.push({ url: nextUrl(page + 1), userData: { ...ud, page: page + 1 } });
            }
            const active = items
                .filter(i => (i.last_reply_time || i.create_time) >= ud.cutoffTs)
                .sort((a, b) => (b.no_of_reply || 0) - (a.no_of_reply || 0) || String(a.thread_id).localeCompare(String(b.thread_id)))
                .slice(0, ud.threadsPerPage || 10);
            for (const t of active) {
                out.push({
                    url: `https://lihkg.com/api_v2/thread/${t.thread_id}/page/1?order=reply_time`,
                    userData: {
                        source: 'lihkg', kind: 'lihkg_thread', exchange: ud.exchange, ticker: ud.ticker,
                        keyword: ud.keyword, threadId: String(t.thread_id), cutoffTs: ud.cutoffTs, headers: ud.headers,
                    },
                });
            }
        } else if (kind === 'telegram_search') {
            if (tg.ids.length && more) {
                const oldest = Math.min(...tg.times);
                if (oldest > ud.cutoffTs) out.push({ url: nextUrl(Math.min(...tg.ids)), userData: { ...ud, page: page + 1 } });
            }
        }
        return out;
    }

    let payload; let payloadType; const meta = {};
    let tg = { ids: [], times: [] };
    if (kind === 'telegram_search' || kind === 'telegram_channel') {
        const $ = context.$;
        payloadType = 'html';
        payload = $('.tgme_widget_message_wrap').map((i, el) => $.html(el)).get().join('\n');
        $('.tgme_channel_info_counter').each((i, el) => {
            if (/subscriber/i.test($(el).find('.counter_type').text())) meta.subscribers = $(el).find('.counter_value').text().trim();
        });
        if (!meta.subscribers) {
            const h = $('.tgme_header_counter').first().text().trim();
            if (/subscriber/i.test(h)) meta.subscribers = h.replace(/subscribers?/i, '').trim();
        }
        $('.tgme_widget_message').each((i, el) => {
            const post = $(el).attr('data-post') || '';
            const id = parseInt(post.split('/').pop(), 10);
            const dt = $(el).find('.tgme_widget_message_date time').attr('datetime');
            if (!Number.isNaN(id) && dt) { tg.ids.push(id); tg.times.push(Date.parse(dt) / 1000); }
        });
        if (kind === 'telegram_channel') payload = ''; // only the subscriber count is needed
    } else {
        payloadType = 'json';
        payload = context.json;
        if (!payload) {
            const body = typeof context.body === 'string' ? context.body : context.body.toString('utf8');
            payload = JSON.parse(body);
        }
        if (kind === 'naver_list') payload = slimNaver(payload);
    }

    for (const r of nextRequests(payload, tg)) {
        await context.enqueueRequest({
            url: r.url,
            headers: r.userData.headers || {},
            userData: r.userData,
            uniqueKey: `${r.url}#${r.userData.ticker || ''}`,
        });
    }
    log.info(`${kind} ${ud.ticker || ud.channel || ''} page=${ud.page || 1}`);
    return { url: request.url, userData: ud, payloadType, payload, meta, fetchedAt: new Date().toISOString() };
}
