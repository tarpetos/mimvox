use std::io::Read;
use std::sync::LazyLock;

use axum::http::{HeaderMap, HeaderValue, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};

const TABLE: &str = include_str!("../openapi/methods.json");
const SCALAR_GZ: &[u8] = include_bytes!("../vendor/scalar/standalone.js.gz");
const ICON: &str = include_str!("../assets/docs-icon.svg");

/// Served without the token: they carry no data.
pub const PUBLIC_PATHS: &[&str] = &["/docs", "/docs/scalar.js", "/docs/init.js", "/docs/icon.svg", "/openapi.json"];

pub const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; \
img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; base-uri 'none'; \
form-action 'none'; frame-ancestors 'none'";

const PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mimvox API</title>
<link rel="icon" type="image/svg+xml" href="/docs/icon.svg">
</head>
<body>
<div id="app"></div>
<script src="/docs/scalar.js"></script>
<script src="/docs/init.js"></script>
</body>
</html>
"#;

// Scalar's online extras (fonts, telemetry, AI agent, MCP, its hosted client) are off; the CSP blocks them anyway.
const INIT: &str = r#"Scalar.createApiReference('#app', {
  url: '/openapi.json',
  favicon: '/docs/icon.svg',
  customCss: `
    .scalar-app aside::before, .t-doc__header header::after {
      content: ''; flex: none; background: url(/docs/icon.svg) no-repeat left center / contain;
    }
    .scalar-app aside::before {
      content: 'Mimvox API'; display: flex; align-items: center; height: 40px; margin: 14px 12px 4px;
      padding-left: 50px; background-size: 40px 40px; font-size: 16px; font-weight: 600;
      color: var(--scalar-color-1);
    }
    .t-doc__header header::after { width: 28px; height: 28px; margin-right: 8px; }
  `,
  withDefaultFonts: false,
  telemetry: false,
  agent: { disabled: true },
  mcp: { disabled: true },
  hideClientButton: true,
  showDeveloperTools: 'never',
  persistAuth: false,
  authentication: { preferredSecurityScheme: 'bearer' },
});
"#;

static SPEC_JSON: LazyLock<String> = LazyLock::new(|| spec().to_string());
static SCALAR_JS: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(SCALAR_GZ).read_to_end(&mut out).expect("the bundled Scalar is valid gzip");
    out
});

pub async fn openapi() -> Response {
    ([(header::CONTENT_TYPE, "application/json")], SPEC_JSON.as_str()).into_response()
}

pub async fn page() -> Response {
    let mut res = ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], PAGE).into_response();
    let h = res.headers_mut();
    h.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(CSP));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    h.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    res
}

pub async fn icon() -> Response {
    let mut res = ([(header::CONTENT_TYPE, "image/svg+xml")], ICON).into_response();
    res.headers_mut().insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    res
}

pub async fn init_js() -> Response {
    script(INIT.as_bytes().to_vec(), None)
}

pub async fn scalar_js(headers: HeaderMap) -> Response {
    let gzip = headers.get(header::ACCEPT_ENCODING).and_then(|v| v.to_str().ok()).is_some_and(accepts_gzip);
    let mut res = if gzip { script(SCALAR_GZ.to_vec(), Some("gzip")) } else { script(SCALAR_JS.clone(), None) };
    res.headers_mut().insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    res
}

fn script(body: Vec<u8>, encoding: Option<&'static str>) -> Response {
    let mut res = ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], body).into_response();
    let h = res.headers_mut();
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    if let Some(e) = encoding {
        h.insert(header::CONTENT_ENCODING, HeaderValue::from_static(e));
    }
    res
}

fn accepts_gzip(value: &str) -> bool {
    value.split(',').any(|item| {
        let mut parts = item.split(';').map(str::trim);
        let name = parts.next().unwrap_or("");
        let refused = parts.any(|p| p.strip_prefix("q=").and_then(|q| q.parse::<f32>().ok()) == Some(0.0));
        (name.eq_ignore_ascii_case("gzip") || name == "*") && !refused
    })
}

/// A minimal value the schema accepts: the request examples of the docs page.
fn example(schema: &Value) -> Value {
    for key in ["const", "default"] {
        if let Some(v) = schema.get(key) {
            return v.clone();
        }
    }
    if let Some(v) = schema.get("examples").and_then(|e| e.get(0)).or_else(|| schema.get("enum").and_then(|e| e.get(0)))
    {
        return v.clone();
    }
    let ty = match &schema["type"] {
        Value::Array(types) => types.iter().filter_map(Value::as_str).find(|t| *t != "null").unwrap_or("null"),
        t => t.as_str().unwrap_or("object"),
    };
    match ty {
        "string" => json!(""),
        "integer" | "number" => json!(0),
        "boolean" => json!(false),
        "array" => json!([]),
        "null" => Value::Null,
        _ => {
            let props = &schema["properties"];
            let required = schema["required"].as_array().into_iter().flatten().filter_map(Value::as_str);
            Value::Object(required.map(|k| (k.to_string(), example(&props[k]))).collect())
        }
    }
}

fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

fn titled(mut schema: Value, title: &str) -> Value {
    if let Some(o) = schema.as_object_mut() {
        o.insert("title".into(), json!(title));
    }
    schema
}

fn schema_ref(name: &str) -> Value {
    json!({ "$ref": format!("#/components/schemas/{name}") })
}

fn response_ref(name: &str) -> Value {
    json!({ "$ref": format!("#/components/responses/{name}") })
}

fn error_responses(names: &[(&str, &str)]) -> Map<String, Value> {
    names.iter().map(|(status, name)| (status.to_string(), response_ref(name))).collect()
}

/// The OpenAPI 3.1 description of the API: the engine methods and events come from `openapi/methods.json`.
pub fn spec() -> Value {
    let table: Value = serde_json::from_str(TABLE).expect("openapi/methods.json is valid JSON");
    let mut schemas = Map::new();
    let (mut requests, mut results, mut mapping, mut examples) = (Vec::new(), Vec::new(), Map::new(), Map::new());
    let mut method_rows = String::new();
    for m in table["methods"].as_array().into_iter().flatten() {
        let name = m["name"].as_str().unwrap_or_default();
        let summary = m["summary"].as_str().unwrap_or_default();
        let detail = m["description"].as_str().unwrap_or_default();
        let job = m["job"].as_bool().unwrap_or(false);
        let (req, res) = (format!("rpc.{name}"), format!("result.{name}"));
        schemas.insert(
            req.clone(),
            json!({
                "title": name,
                "description": format!("{summary}. {detail}"),
                "type": "object",
                "required": ["method"],
                "properties": { "method": { "const": name }, "params": m["params"] },
            }),
        );
        schemas.insert(res.clone(), titled(m["result"].clone(), name));
        requests.push(schema_ref(&req));
        results.push(schema_ref(&res));
        mapping.insert(name.into(), json!(format!("#/components/schemas/{req}")));
        examples.insert(
            name.into(),
            json!({ "summary": format!("{name}: {summary}"), "value": { "method": name, "params": example(&m["params"]) } }),
        );
        let events = m["events"].as_array().into_iter().flatten().filter_map(Value::as_str);
        let events = events.map(|e| format!("`{e}`")).collect::<Vec<_>>().join(", ");
        method_rows += &format!("| `{name}` | {} | {} | {events} |\n", cell(summary), if job { "yes" } else { "" });
    }
    let (mut event_refs, mut event_rows) = (Vec::new(), String::new());
    for e in table["events"].as_array().into_iter().flatten() {
        let name = e["name"].as_str().unwrap_or_default();
        let job =
            if e["job"].as_bool().unwrap_or(false) { json!({ "type": "string" }) } else { json!({ "type": "null" }) };
        let key = format!("event.{name}");
        schemas.insert(
            key.clone(),
            json!({
                "title": name,
                "description": e["description"],
                "type": "object",
                "required": ["event", "job", "data"],
                "properties": { "event": { "const": name }, "job": job, "data": e["data"] },
            }),
        );
        event_refs.push(schema_ref(&key));
        event_rows += &format!("| `{name}` | {} |\n", cell(e["description"].as_str().unwrap_or_default()));
    }
    schemas.insert(
        "Error".into(),
        json!({
            "type": "object",
            "required": ["code", "message"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The engine's code, or the server's own (unauthorized, forbidden, too_large, timeout)",
                    "examples": ["bad_request", "not_found", "busy", "not_ready", "paused", "cancelled", "no_memory",
                        "internal", "unauthorized", "forbidden", "too_large", "timeout"],
                },
                "message": { "type": "string" },
            },
        }),
    );
    schemas.insert(
        "OpenAiError".into(),
        json!({
            "type": "object",
            "required": ["error"],
            "properties": { "error": {
                "type": "object",
                "properties": {
                    "message": { "type": "string" },
                    "type": { "type": "string", "enum": ["invalid_request_error", "server_error"] },
                    "code": { "type": "string" },
                },
            } },
        }),
    );

    let mut responses = Map::new();
    for (name, schema, text) in [
        ("BadRequest", "Error", "bad_request: a malformed request or invalid params"),
        ("Unauthorized", "Error", "unauthorized: the bearer token is missing or wrong"),
        ("Forbidden", "Error", "forbidden: without a token only local clients that are not web pages are served"),
        ("NotFound", "Error", "not_found: an unknown method, voice, job or file"),
        ("Busy", "Error", "busy: the engine is already doing that"),
        ("TooLarge", "Error", "too_large: the body is over the limit"),
        ("Internal", "Error", "internal, and every other engine code (paused, cancelled, ...)"),
        ("NotReady", "Error", "not_ready or no_memory: the engine or its models are not ready"),
        ("AiBadRequest", "OpenAiError", "Invalid request (bad input, unknown voice or format)"),
        ("AiServerError", "OpenAiError", "The speech failed"),
        ("AiNotReady", "OpenAiError", "not_ready: the engine, its models or an encoder is missing"),
        ("AiTimeout", "OpenAiError", "timeout: the job ran past 30 minutes"),
    ] {
        responses.insert(
            name.into(),
            json!({ "description": text, "content": { "application/json": { "schema": schema_ref(schema) } } }),
        );
    }
    let auth = [("401", "Unauthorized"), ("403", "Forbidden")];
    let with = |extra: &[(&str, &str)]| {
        let mut r = error_responses(&auth);
        r.extend(error_responses(extra));
        r
    };

    let mut rpc_responses = with(&[
        ("400", "BadRequest"),
        ("404", "NotFound"),
        ("409", "Busy"),
        ("413", "TooLarge"),
        ("500", "Internal"),
        ("503", "NotReady"),
    ]);
    rpc_responses.insert(
        "200".into(),
        json!({
            "description": "The engine's result as is; its shape depends on the method",
            "content": { "application/json": { "schema": { "anyOf": results } } },
        }),
    );
    let mut health_responses = with(&[("503", "NotReady")]);
    health_responses.insert(
        "200".into(),
        json!({
            "description": "The engine answered",
            "content": { "application/json": { "schema": {
                "type": "object",
                "properties": {
                    "status": { "const": "ok" },
                    "api_version": { "type": "string" },
                    "engine": schema_ref("result.hello"),
                },
            } } },
        }),
    );
    let mut events_responses = with(&[]);
    events_responses.insert(
        "200".into(),
        json!({
            "description": "Server-Sent Events: each event's `data:` line is one JSON object, the default SSE event type",
            "content": { "text/event-stream": { "schema": { "oneOf": event_refs } } },
        }),
    );
    let mut upload_responses = with(&[("400", "BadRequest"), ("413", "TooLarge"), ("500", "Internal")]);
    upload_responses.insert(
        "200".into(),
        json!({
            "description": "Saved; pass `path` to methods that take files",
            "content": { "application/json": { "schema": {
                "type": "object",
                "properties": { "path": { "type": "string" }, "size_bytes": { "type": "integer" } },
            } } },
        }),
    );
    let mut download_responses = with(&[("400", "BadRequest"), ("404", "NotFound")]);
    download_responses.insert(
        "200".into(),
        json!({ "description": "The file, streamed", "content": { "*/*": { "schema": { "type": "string", "format": "binary" } } } }),
    );
    let ai = [("401", "Unauthorized"), ("403", "Forbidden"), ("500", "AiServerError"), ("503", "AiNotReady")];
    let mut models_responses = error_responses(&auth);
    models_responses.insert(
        "200".into(),
        json!({ "description": "The one model", "content": { "application/json": { "schema": {
            "type": "object",
            "properties": {
                "object": { "const": "list" },
                "data": { "type": "array", "items": { "type": "object", "properties": {
                    "id": { "const": "mimvox" }, "object": { "const": "model" }, "created": { "type": "integer" },
                    "owned_by": { "type": "string" },
                } } },
            },
        } } } }),
    );
    let mut voices_responses = error_responses(&ai);
    voices_responses.insert(
        "200".into(),
        json!({ "description": "The voices.list result", "content": { "application/json": { "schema": schema_ref("result.voices.list") } } }),
    );
    let mut speech_responses = error_responses(&ai);
    speech_responses.extend(error_responses(&[("400", "AiBadRequest"), ("404", "AiBadRequest"), ("504", "AiTimeout")]));
    let audio = ["audio/mpeg", "audio/wav", "audio/flac", "audio/ogg", "audio/aac", "audio/mp4", "audio/pcm"];
    let audio: Map<String, Value> =
        audio.iter().map(|t| (t.to_string(), json!({ "schema": { "type": "string", "format": "binary" } }))).collect();
    speech_responses.insert(
        "200".into(),
        json!({ "description": "The audio, in the asked format (aac: raw ADTS; pcm: 16-bit LE mono 24 kHz)", "content": audio }),
    );
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Mimvox local API",
            "version": env!("CARGO_PKG_VERSION"),
            "license": { "name": "PolyForm-Noncommercial-1.0.0", "identifier": "PolyForm-Noncommercial-1.0.0" },
            "description": INFO,
        },
        "servers": [{ "url": "/", "description": "This server" }],
        "security": [{ "bearer": [] }, {}],
        "tags": [
            { "name": "Engine", "description": format!("Every engine method through `POST /rpc`; jobs report through `GET /events`.\n\n| Method | | Job | Events |\n| --- | --- | --- | --- |\n{method_rows}") },
            { "name": "Events", "description": format!("| Event | When |\n| --- | --- |\n{event_rows}") },
            { "name": "Files" },
            { "name": "OpenAI-compatible" },
            { "name": "Docs" },
        ],
        "paths": {
            "/health": { "get": {
                "tags": ["Engine"], "operationId": "health", "summary": "Check the engine",
                "description": "Asks the engine for hello; 503 while it is down or does not answer within 10 s.",
                "responses": health_responses,
            } },
            "/rpc": { "post": {
                "tags": ["Engine"], "operationId": "rpc", "summary": "Call an engine method",
                "description": "Any engine method, `shutdown` included (the server then starts a fresh engine). Body up to 16 MiB. Methods that answer `{job}` report through /events. Pick a method in the examples.",
                "requestBody": { "required": true, "content": { "application/json": {
                    "schema": { "oneOf": requests, "discriminator": { "propertyName": "method", "mapping": mapping } },
                    "examples": examples,
                } } },
                "responses": rpc_responses,
            } },
            "/events": { "get": {
                "tags": ["Events"], "operationId": "events", "summary": "Stream engine events",
                "description": "Every engine event as one `data:` line, the same JSON as the engine's stdout. The server adds engine.exited and api.lagged.",
                "responses": events_responses,
            } },
            "/files": {
                "post": {
                    "tags": ["Files"], "operationId": "upload", "summary": "Upload a file",
                    "description": "The raw body (not multipart), saved under <data>/api/uploads. Over the upload limit is 413 and nothing is kept; uploads older than 24 h are deleted.",
                    "parameters": [{ "name": "name", "in": "query", "description": "The file's name (its extension matters)", "schema": { "type": "string" } }],
                    "requestBody": { "required": true, "content": { "application/octet-stream": { "schema": { "type": "string", "format": "binary" } } } },
                    "responses": upload_responses,
                },
                "get": {
                    "tags": ["Files"], "operationId": "download", "summary": "Download a result",
                    "description": "Files inside <data>/outputs, <data>/tmp or <data>/api/uploads, and books' segments/<NNN>.json. 403 outside, 400 relative or a folder, 404 missing.",
                    "parameters": [{ "name": "path", "in": "query", "required": true, "description": "Absolute path on the API's machine", "schema": { "type": "string" } }],
                    "responses": download_responses,
                },
            },
            "/v1/models": { "get": {
                "tags": ["OpenAI-compatible"], "operationId": "models", "summary": "List models",
                "responses": models_responses,
            } },
            "/v1/audio/voices": { "get": {
                "tags": ["OpenAI-compatible"], "operationId": "voices", "summary": "List voices",
                "responses": voices_responses,
            } },
            "/v1/audio/speech": { "post": {
                "tags": ["OpenAI-compatible"], "operationId": "speech", "summary": "Speak text",
                "description": "Runs tts.generate and returns the audio. Requests queue one at a time; a client that leaves cancels its job.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["input", "voice"],
                    "properties": {
                        "model": { "type": "string", "description": "Ignored", "examples": ["mimvox"] },
                        "input": { "type": "string", "maxLength": 4096 },
                        "voice": { "type": "string", "description": "A voice id or name (case-insensitive)" },
                        "response_format": { "type": "string", "enum": ["mp3", "wav", "flac", "opus", "aac", "pcm"], "default": "mp3",
                            "description": "opus is Ogg; aac is raw ADTS and needs ffmpeg; pcm is 16-bit LE mono at 24 kHz" },
                        "speed": { "type": "number", "description": "Clamped to 0.8-1.3 by the engine", "default": 1 },
                        "language": { "type": "string", "description": "Not in OpenAI's API: the language of input; the voice's own by default" },
                    },
                } } } },
                "responses": speech_responses,
            } },
            "/openapi.json": { "get": {
                "tags": ["Docs"], "operationId": "openapi", "summary": "This description", "security": [],
                "responses": { "200": { "description": "OpenAPI 3.1", "content": { "application/json": {} } } },
            } },
            "/docs": { "get": {
                "tags": ["Docs"], "operationId": "docs", "summary": "This page", "security": [],
                "responses": { "200": { "description": "The interactive reference", "content": { "text/html": {} } } },
            } },
        },
        "components": {
            "securitySchemes": { "bearer": {
                "type": "http",
                "scheme": "bearer",
                "description": "The API token (Settings, API, or `mimvox-server token show`). Optional on loopback unless the API requires it; always required on the local network.",
            } },
            "schemas": schemas,
            "responses": responses,
        },
    })
}

const INFO: &str = "Mimvox clones a voice and speaks text with it, on this computer. This API is the app's, or \
`mimvox-server`'s: whoever reaches it can do what the app can.\n\n\
**Token.** Send `Authorization: Bearer <token>` (the Authentication box on this page). On 127.0.0.1 it is optional \
unless the API requires it; on the local network it is always required. Without a token only local programs get in, \
never web pages (403).\n\n\
**Errors** are `{\"code\",\"message\"}` (`/v1/*`: OpenAI's `{\"error\":{...}}`): bad_request 400, unauthorized 401, \
forbidden 403, not_found 404, busy 409, too_large 413, not_ready and no_memory 503, timeout 504, any other code 500.\n\n\
**Use.** Speech made with the OmniVoice weights (CC BY-NC 4.0) is for non-commercial use only, and only your own \
voice or one whose owner gave explicit permission may be cloned.";

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    use super::*;

    fn repo() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn python_files(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                python_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "py") {
                out.push(path);
            }
        }
    }

    /// The engine's registry: every `@method("...")`, and `shutdown`, which server.py answers itself.
    fn engine_methods() -> BTreeSet<String> {
        let mut files = Vec::new();
        python_files(&repo().join("engine/src/mimvox_engine"), &mut files);
        let mut names = BTreeSet::new();
        for f in files {
            let text = std::fs::read_to_string(&f).unwrap();
            for rest in text.split("@method(\"").skip(1) {
                names.insert(rest.split('"').next().unwrap().to_string());
            }
        }
        let server = std::fs::read_to_string(repo().join("engine/src/mimvox_engine/server.py")).unwrap();
        assert!(server.contains("name == \"shutdown\""), "server.py no longer answers shutdown itself");
        names.insert("shutdown".into());
        names
    }

    fn table() -> Value {
        serde_json::from_str(TABLE).unwrap()
    }

    fn names(key: &str) -> Vec<String> {
        table()[key].as_array().unwrap().iter().map(|m| m["name"].as_str().unwrap().to_string()).collect()
    }

    #[test]
    fn the_spec_has_every_engine_method_and_no_other() {
        let listed = names("methods");
        let set: BTreeSet<String> = listed.iter().cloned().collect();
        assert_eq!(set.len(), listed.len(), "a method is listed twice in openapi/methods.json");
        let engine = engine_methods();
        assert!(engine.len() > 40, "the engine's methods were not found: {engine:?}");
        let missing: Vec<_> = engine.difference(&set).collect();
        let extra: Vec<_> = set.difference(&engine).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "openapi/methods.json: missing {missing:?}, not in the engine {extra:?}"
        );
    }

    #[test]
    fn methods_and_events_are_the_contracts() {
        let contract = std::fs::read_to_string(repo().join("contract/engine-protocol.md")).unwrap();
        let events = names("events");
        let set: BTreeSet<&String> = events.iter().collect();
        assert_eq!(set.len(), events.len(), "an event is listed twice in openapi/methods.json");
        for name in names("methods").iter().chain(&events) {
            let named = |end: char| contract.contains(&format!("`{name}{end}"));
            assert!(named('`') || named(' '), "{name} is not in contract/engine-protocol.md");
        }
        for m in table()["methods"].as_array().unwrap() {
            for e in m["events"].as_array().into_iter().flatten() {
                assert!(set.contains(&e.as_str().unwrap().to_string()), "{} emits {e}, which has no entry", m["name"]);
            }
        }
    }

    fn refs<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
        match v {
            Value::Object(o) => {
                if let Some(r) = o.get("$ref").and_then(Value::as_str) {
                    out.push(r);
                }
                o.values().for_each(|v| refs(v, out));
            }
            Value::Array(a) => a.iter().for_each(|v| refs(v, out)),
            _ => {}
        }
    }

    #[test]
    fn the_spec_is_openapi_3_1_and_its_refs_resolve() {
        let spec = spec();
        assert_eq!(spec["openapi"], "3.1.0");
        assert_eq!(spec["info"]["version"], env!("CARGO_PKG_VERSION"));
        let mut found = Vec::new();
        refs(&spec, &mut found);
        assert!(found.len() > 100);
        for r in found {
            let target = spec.pointer(r.trim_start_matches('#')).or_else(|| {
                let (base, name) = r.rsplit_once('/')?;
                spec.pointer(base.trim_start_matches('#'))?.get(name)
            });
            assert!(target.is_some_and(|t| !t.is_null()), "{r} does not resolve");
        }
        let examples = spec["paths"]["/rpc"]["post"]["requestBody"]["content"]["application/json"]["examples"]
            .as_object()
            .unwrap();
        assert_eq!(examples.len(), names("methods").len());
        let voices = &examples["dataset.add"]["value"]["params"];
        assert!(voices.get("voice_id").is_some(), "examples carry the required params: {voices}");
    }

    #[test]
    fn the_icon_draws_offline() {
        assert!(ICON.starts_with("<svg") && ICON.contains("Mimvox API"));
        let outside = ICON.replace("http://www.w3.org/2000/svg", "");
        assert!(!outside.contains("http") && !outside.contains("<text") && !outside.contains("font"), "{ICON}");
        assert!(INIT.contains("/docs/icon.svg") && PAGE.contains("/docs/icon.svg"));
    }

    #[test]
    fn gzip_is_sent_to_clients_that_take_it() {
        for v in ["gzip", "gzip, deflate, br", "br;q=1.0, GZIP;q=0.5", "*"] {
            assert!(accepts_gzip(v), "{v}");
        }
        for v in ["", "identity", "br", "gzip;q=0", "gzip; q=0.0"] {
            assert!(!accepts_gzip(v), "{v}");
        }
        assert!(SCALAR_JS.len() > SCALAR_GZ.len());
    }
}
