import { realpathSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

export function attachTokenUsage(span, usage, attributeKey) {
    if (span.parentId || !usage || Object.keys(usage).length === 0) {
        return;
    }
    const end = span.end.bind(span);
    span.end = (options) => {
        span.setAttribute(attributeKey, usage);
        return end(options);
    };
}

async function main() {
    const [exporterPath, rawPayload] = process.argv.slice(2);
    if (!exporterPath || !rawPayload) {
        throw new Error('Expected the mlflow-codex executable path and notify payload');
    }
    const exporter = realpathSync(exporterPath);
    const require = createRequire(exporter);
    const codex = await import(pathToFileURL(resolve(dirname(exporter), '../dist/index.js')));
    const core = require('@mlflow/core');
    const payload = JSON.parse(rawPayload);
    if (payload.type !== 'agent-turn-complete') {
        throw new Error('Expected an agent-turn-complete payload');
    }
    if (!codex.ensureInitialized()) {
        throw new Error('MLflow tracing configuration could not be initialized');
    }
    core.registerOnSpanStartHook((span) => {
        attachTokenUsage(span, payload['token-usage'], core.SpanAttributeKey.TOKEN_USAGE);
    });
    await codex.processNotify(payload);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
    main().catch((error) => {
        console.error('[mlflow]', error);
        process.exitCode = 1;
    });
}
