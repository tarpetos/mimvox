//! Runs mimvox-mcp over stdio and streamable HTTP against the API hosted in-process on app/examples/fake_engine.py.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use mimvox_api::{ApiConfig, Hub};
use mimvox_engine_client::{EngineClient, EngineConfig};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const WAIT: Duration = Duration::from_secs(30);
const TOOLS: [&str; 12] = [
    "engine_call",
    "list_voices",
    "speak",
    "import_text",
    "start_book",
    "book_status",
    "pause_book",
    "resume_book",
    "job_status",
    "wait_job",
    "upload_file",
    "download_file",
];

fn python() -> PathBuf {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .flat_map(|d| ["python3", "python", "python.exe"].map(|n| d.join(n)))
        // The first one that starts, as the tests write it: on Windows PATH can list a Store alias that does not.
        .filter(|p| p.is_file())
        .find(|p| {
            let p = p.display().to_string().replace('\\', "/");
            std::process::Command::new(p).arg("--version").output().is_ok_and(|o| o.status.success())
        })
        .expect("python3 on PATH")
}

/// The HTTP API with the fake engine, as mimvox-server serves it.
struct Api {
    url: String,
    data: PathBuf,
    stop: CancellationToken,
    _engine: Arc<EngineClient>,
    // Last: the engine ends before its folder goes.
    _dir: tempfile::TempDir,
}

impl Drop for Api {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

async fn api(token: Option<&str>) -> Api {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let python = python().display().to_string().replace('\\', "/");
    std::fs::write(dir.path().join("runtime.toml"), format!("[python]\nengine = \"{python}\"\n")).unwrap();
    let cfg = EngineConfig {
        runtime_toml: dir.path().join("runtime.toml"),
        engine_src: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/examples/fake_engine_src"),
        data_dir: data.clone(),
    };
    let (hub, _exits) = Hub::new();
    let mut events = hub.subscribe();
    let engine = Arc::new(EngineClient::spawn(&cfg, hub.sink()).unwrap());
    hub.attach_shared(engine.clone());
    tokio::time::timeout(WAIT, async { while events.recv().await.unwrap().event != "ready" {} })
        .await
        .expect("the fake engine did not get ready");
    let stop = CancellationToken::new();
    let config = ApiConfig { data_dir: data.clone(), token: token.map(str::to_string), max_upload_bytes: 1 << 20 };
    let app = mimvox_api::router(hub, config, stop.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(axum::serve(listener, app).with_graceful_shutdown(stop.clone().cancelled_owned()).into_future());
    Api { url, data, stop, _engine: engine, _dir: dir }
}

/// mimvox-mcp with a clean environment: no token unless given, `data` as its MIMVOX_DATA.
fn command(url: &str, data: &Path, args: &[&str], env: &[(&str, &str)]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mimvox-mcp"));
    cmd.arg("--url")
        .arg(url)
        .args(args)
        .env_remove("MIMVOX_API_TOKEN")
        .env_remove("MIMVOX_URL")
        .env("MIMVOX_DATA", data)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd
}

/// An MCP client over the child's stdio, one JSON-RPC message per line.
struct Mcp {
    _child: Child,
    stdin: ChildStdin,
    rx: mpsc::UnboundedReceiver<Value>,
    next_id: u64,
    notes: Vec<Value>,
}

impl Mcp {
    async fn start(url: &str, data: &Path, args: &[&str], env: &[(&str, &str)]) -> Self {
        let mut child = command(url, data, args, env).spawn().unwrap();
        let stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap()).lines();
        let mut stderr = BufReader::new(child.stderr.take().unwrap()).lines();
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Ok(Some(line)) = stdout.next_line().await {
                let msg: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("not JSON-RPC ({e}): {line}"));
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });
        tokio::spawn(async move {
            while let Ok(Some(line)) = stderr.next_line().await {
                eprintln!("[mcp] {line}");
            }
        });
        let mut mcp = Self { _child: child, stdin, rx, next_id: 1, notes: Vec::new() };
        let init = mcp
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": { "name": "mimvox-test", "version": "0" },
                }),
            )
            .await;
        assert_eq!(init["result"]["serverInfo"]["name"], "mimvox-mcp", "{init}");
        assert!(init["result"]["capabilities"]["tools"].is_object(), "{init}");
        assert!(init["result"]["capabilities"]["resources"].is_object(), "{init}");
        mcp.send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).await;
        mcp
    }

    async fn send(&mut self, msg: Value) {
        let mut line = msg.to_string();
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await.unwrap();
        self.stdin.flush().await.unwrap();
    }

    /// The whole response message; notifications that come first are kept in `notes`.
    async fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })).await;
        tokio::time::timeout(WAIT, async {
            loop {
                let msg = self.rx.recv().await.expect("mimvox-mcp closed its output");
                if msg["id"] == id && msg.get("method").is_none() {
                    return msg;
                }
                if msg.get("id").is_some() && msg.get("method").is_some() {
                    let reply = json!({ "jsonrpc": "2.0", "id": msg["id"], "result": {} });
                    self.send(reply).await;
                } else {
                    self.notes.push(msg);
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("no answer to {method}"))
    }

    async fn call(&mut self, name: &str, args: Value) -> Value {
        let res = self.request("tools/call", json!({ "name": name, "arguments": args })).await;
        res.get("result").cloned().unwrap_or_else(|| panic!("{name}: {res}"))
    }

    async fn ok(&mut self, name: &str, args: Value) -> Value {
        let res = self.call(name, args).await;
        assert_ne!(res["isError"], true, "{name}: {res}");
        res["structuredContent"].clone()
    }

    async fn err(&mut self, name: &str, args: Value) -> Value {
        let res = self.call(name, args).await;
        assert_eq!(res["isError"], true, "{name}: {res}");
        res
    }

    fn progress(&mut self, token: &str) -> Vec<Value> {
        let all = std::mem::take(&mut self.notes);
        let (mine, rest) = all
            .into_iter()
            .partition(|n| n["method"] == "notifications/progress" && n["params"]["progressToken"] == token);
        self.notes = rest;
        mine
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

#[tokio::test]
async fn tools_files_and_resources() {
    let api = api(None).await;
    let home = tempfile::tempdir().unwrap();
    let mut mcp = Mcp::start(&api.url, home.path(), &[], &[]).await;

    let list = mcp.request("tools/list", json!({})).await;
    let tools = list["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    for name in TOOLS {
        assert!(names.contains(&name), "{name} missing from {names:?}");
    }
    assert_eq!(names.len(), TOOLS.len(), "{names:?}");
    for tool in tools {
        assert_eq!(tool["inputSchema"]["type"], "object", "{tool}");
        assert!(tool["description"].as_str().unwrap().len() > 40, "{tool}");
    }
    let schema = |name: &str| tools.iter().find(|t| t["name"] == name).unwrap()["inputSchema"].clone();
    assert_eq!(schema("speak")["required"], json!(["text"]));
    assert!(schema("speak")["$defs"]["AudioFormat"]["enum"].as_array().unwrap().contains(&json!("mp3")));
    assert_eq!(schema("engine_call")["required"], json!(["method"]));
    assert_eq!(schema("wait_job")["required"], json!(["job"]));
    assert_eq!(schema("upload_file")["required"], json!(["local_path"]));

    let hello = mcp.ok("engine_call", json!({ "method": "hello" })).await;
    assert_eq!(hello["version"], "0.1.0-fake");
    let analyzed = mcp.ok("engine_call", json!({ "method": "text.analyze", "params": { "text": "one two" } })).await;
    assert_eq!(analyzed["words"], 2);
    let res = mcp.err("engine_call", json!({ "method": "no.such.method" })).await;
    assert_eq!(res["structuredContent"]["error"]["code"], "not_found");
    assert_eq!(res["structuredContent"]["error"]["http_status"], 404);
    assert!(res["content"][0]["text"].as_str().unwrap().contains("no.such.method"), "{res}");

    let voices = mcp.ok("list_voices", json!({})).await;
    assert_eq!(voices["voices"][0]["id"], "fake-voice");

    let spoken = mcp.call("speak", json!({ "text": "Hello there.", "voice": "FAKE VOICE" })).await;
    assert_ne!(spoken["isError"], true, "{spoken}");
    let summary = &spoken["structuredContent"];
    assert_eq!(summary["state"], "done");
    assert_eq!(summary["voice_id"], "fake-voice");
    let path = summary["path"].as_str().unwrap().to_string();
    assert_eq!(Path::new(&path).parent().unwrap().file_name().unwrap(), "outputs");
    let content = spoken["content"].as_array().unwrap();
    let audio = content.iter().find(|c| c["type"] == "audio").expect("audio content");
    assert_eq!(audio["mimeType"], "audio/wav");
    let wav = BASE64.decode(audio["data"].as_str().unwrap()).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(wav, std::fs::read(&path).unwrap());
    let link = content.iter().find(|c| c["type"] == "resource_link").expect("resource link");
    let uri = link["uri"].as_str().unwrap().to_string();
    assert!(uri.starts_with("mimvox://file/"), "{uri}");
    assert_eq!(summary["uri"], uri.as_str());

    let res = mcp.err("speak", json!({ "text": "Hi.", "voice": "nobody" })).await;
    assert_eq!(res["structuredContent"]["error"]["code"], "not_found");
    assert!(res["content"][0]["text"].as_str().unwrap().contains("Fake voice"), "{res}");

    let resources = mcp.request("resources/list", json!({})).await;
    let listed = resources["result"]["resources"].as_array().unwrap();
    assert!(listed.iter().any(|r| r["uri"] == uri.as_str() && r["mimeType"] == "audio/wav"), "{resources}");
    let templates = mcp.request("resources/templates/list", json!({})).await;
    assert_eq!(templates["result"]["resourceTemplates"][0]["uriTemplate"], "mimvox://file/{path}");
    let read = mcp.request("resources/read", json!({ "uri": uri })).await;
    let blob = BASE64.decode(read["result"]["contents"][0]["blob"].as_str().unwrap()).unwrap();
    assert_eq!(blob, wav);
    let outside = format!("mimvox://file/{}", api.data.join("secret.txt").display().to_string().replace('/', "%2F"));
    std::fs::write(api.data.join("secret.txt"), "no").unwrap();
    let read = mcp.request("resources/read", json!({ "uri": outside })).await;
    assert!(read["error"].is_object(), "{read}");

    let local = home.path().join("my take.wav");
    std::fs::write(&local, &wav).unwrap();
    let up = mcp.ok("upload_file", json!({ "local_path": local })).await;
    let remote = up["path"].as_str().unwrap().to_string();
    assert_eq!(up["size_bytes"], wav.len());
    assert!(remote.ends_with("-my_take.wav"), "{remote}");
    let res = mcp.err("upload_file", json!({ "local_path": home.path().join("missing.wav") })).await;
    assert_eq!(res["structuredContent"]["error"]["code"], "not_found");

    let down = home.path().join("down");
    std::fs::create_dir_all(&down).unwrap();
    let got = mcp.ok("download_file", json!({ "path": remote, "dest": down })).await;
    let saved = PathBuf::from(got["path"].as_str().unwrap());
    assert_eq!(saved.parent().unwrap(), down);
    assert_eq!(std::fs::read(&saved).unwrap(), wav);
    let res = mcp.err("download_file", json!({ "path": remote, "dest": down })).await;
    assert_eq!(res["structuredContent"]["error"]["code"], "exists");
    mcp.ok("download_file", json!({ "path": remote, "dest": saved, "overwrite": true })).await;
    let res = mcp.err("download_file", json!({ "path": api.data.join("secret.txt"), "dest": down })).await;
    assert_eq!(res["structuredContent"]["error"]["http_status"], 403);
}

#[tokio::test]
async fn books_and_jobs_with_progress() {
    let api = api(None).await;
    let home = tempfile::tempdir().unwrap();
    let mut mcp = Mcp::start(&api.url, home.path(), &[], &[]).await;
    let doc = api.data.join("book.txt");
    let chapters: String = (1..=4).map(|i| format!("Chapter {i}\nText of chapter {i}, a few words.\n")).collect();
    std::fs::write(&doc, format!("A Test Book\n{chapters}")).unwrap();

    let imported = mcp.ok("import_text", json!({ "path": doc })).await;
    assert_eq!(imported["chapters"].as_array().unwrap().len(), 4);
    assert_eq!(imported["chapters"][1], json!({ "title": "Chapter 2", "words": 7, "index": 1 }));
    let full = mcp.ok("import_text", json!({ "path": doc, "include_text": true })).await;
    assert!(full["chapters"][0]["text"].as_str().unwrap().starts_with("Text of chapter 1"));

    let started = mcp.ok("start_book", json!({ "path": doc, "format": "mp3", "m4b": true })).await;
    let (book, job) = (started["id"].as_str().unwrap().to_string(), started["job"].as_str().unwrap().to_string());
    let status = mcp.ok("book_status", json!({ "id": book })).await;
    assert_eq!(status["settings"]["title"], "A Test Book");
    assert_eq!(status["settings"]["output_format"], "mp3");
    assert_eq!(status["settings"]["m4b"], true);
    assert_eq!(status["settings"]["voice_id"], "fake-voice");
    assert_eq!(status["settings"]["source"], doc.display().to_string());

    let req = json!({ "name": "wait_job", "arguments": { "job": job }, "_meta": { "progressToken": "p-book" } });
    let res = mcp.request("tools/call", req).await;
    let ended = &res["result"]["structuredContent"];
    assert_eq!(ended["state"], "done", "{res}");
    assert_eq!(ended["event"], "book.done");
    assert_eq!(ended["result"]["chapters"], 4);
    let progress = mcp.progress("p-book");
    assert!(!progress.is_empty(), "no progress notifications");
    let values: Vec<f64> = progress.iter().map(|n| n["params"]["progress"].as_f64().unwrap()).collect();
    assert!(values.windows(2).all(|w| w[0] < w[1]), "{values:?}");
    assert_eq!(progress[0]["params"]["total"], 100.0);
    assert!(progress[0]["params"]["message"].as_str().unwrap().starts_with("book.progress"));
    assert_eq!(mcp.ok("job_status", json!({ "job": job })).await["state"], "done");
    assert_eq!(mcp.ok("book_status", json!({})).await["books"][0]["status"], "done");

    let chapters: Vec<Value> = (1..=6).map(|i| json!({ "title": format!("Part {i}"), "text": "Some text." })).collect();
    let started = mcp.ok("start_book", json!({ "title": "Short", "chapters": chapters })).await;
    let book = started["id"].as_str().unwrap().to_string();
    mcp.ok("pause_book", json!({ "id": book })).await;
    let paused = mcp.ok("wait_job", json!({ "job": started["job"] })).await;
    assert_eq!(paused["state"], "paused", "{paused}");
    let resumed = mcp.ok("resume_book", json!({ "id": book })).await;
    assert_ne!(resumed["job"], started["job"]);
    let done = mcp.ok("wait_job", json!({ "job": resumed["job"] })).await;
    assert_eq!(done["state"], "done");

    let req = json!({ "name": "speak", "arguments": { "text": "One. Two. Three." }, "_meta": { "progressToken": 7 } });
    let res = mcp.request("tools/call", req).await;
    assert_eq!(res["result"]["structuredContent"]["state"], "done", "{res}");
    let progress: Vec<Value> =
        std::mem::take(&mut mcp.notes).into_iter().filter(|n| n["params"]["progressToken"] == 7).collect();
    assert_eq!(progress.len(), 3, "{progress:?}");

    let res = mcp.call("wait_job", json!({ "job": "job-unknown", "timeout_s": 1 })).await;
    assert_eq!(res["structuredContent"]["timed_out"], true, "{res}");
    assert_eq!(mcp.ok("job_status", json!({ "job": "job-unknown" })).await["state"], "unknown");
    let res = mcp.err("start_book", json!({ "title": "Nothing" })).await;
    assert_eq!(res["structuredContent"]["error"]["code"], "bad_request");
}

#[tokio::test]
async fn cancelling_speak_cancels_its_job() {
    let api = api(None).await;
    let home = tempfile::tempdir().unwrap();
    let mut mcp = Mcp::start(&api.url, home.path(), &[], &[]).await;
    let long = "One. Two. Three. Four. Five. Six. Seven. Eight.";
    let id = 99;
    let call = json!({ "name": "speak", "arguments": { "text": long } });
    mcp.send(json!({ "jsonrpc": "2.0", "id": id, "method": "tools/call", "params": call })).await;
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let cancel = json!({ "requestId": id, "reason": "test" });
    mcp.send(json!({ "jsonrpc": "2.0", "method": "notifications/cancelled", "params": cancel })).await;
    let res = mcp.call("wait_job", json!({ "job": "job1", "timeout_s": 20 })).await;
    assert_eq!(res["isError"], true, "{res}");
    assert_eq!(res["structuredContent"]["error"]["code"], "cancelled", "{res}");
}

#[tokio::test]
async fn api_down_is_a_clear_tool_error() {
    let home = tempfile::tempdir().unwrap();
    let url = format!("http://127.0.0.1:{}", free_port());
    let mut mcp = Mcp::start(&url, home.path(), &[], &[]).await;
    let list = mcp.request("tools/list", json!({})).await;
    assert_eq!(list["result"]["tools"].as_array().unwrap().len(), TOOLS.len());
    for (tool, args) in
        [("list_voices", json!({})), ("speak", json!({ "text": "Hi." })), ("wait_job", json!({ "job": "job1" }))]
    {
        let res = mcp.err(tool, args).await;
        assert_eq!(res["structuredContent"]["error"]["code"], "unreachable", "{tool}: {res}");
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Mimvox Settings") && text.contains("mimvox serve"), "{text}");
    }
}

#[tokio::test]
async fn token_rules() {
    let api = api(Some("s3cret-token")).await;
    let home = tempfile::tempdir().unwrap();
    let mut mcp = Mcp::start(&api.url, home.path(), &[], &[]).await;
    let res = mcp.err("list_voices", json!({})).await;
    let error = &res["structuredContent"]["error"];
    assert_eq!(error["code"], "unauthorized");
    assert_eq!(error["http_status"], 401);
    assert!(error["message"].as_str().unwrap().contains("MIMVOX_API_TOKEN"), "{error}");

    let mut mcp = Mcp::start(&api.url, home.path(), &[], &[("MIMVOX_API_TOKEN", "s3cret-token")]).await;
    assert_eq!(mcp.ok("list_voices", json!({})).await["voices"][0]["id"], "fake-voice");

    let file = home.path().join("token.txt");
    std::fs::write(&file, "s3cret-token\n").unwrap();
    let mut mcp = Mcp::start(&api.url, home.path(), &["--token-file", file.to_str().unwrap()], &[]).await;
    assert_eq!(mcp.ok("list_voices", json!({})).await["voices"][0]["id"], "fake-voice");

    // The server's own token file, found through MIMVOX_DATA as the server finds it.
    std::fs::create_dir_all(home.path().join("api")).unwrap();
    std::fs::write(home.path().join("api").join("token"), "s3cret-token\n").unwrap();
    let mut mcp = Mcp::start(&api.url, home.path(), &[], &[]).await;
    assert_eq!(mcp.ok("list_voices", json!({})).await["voices"][0]["id"], "fake-voice");
    // Not sent anywhere but loopback.
    let lan = api.url.replace("127.0.0.1", "localhost.invalid");
    let mut mcp = Mcp::start(&lan, home.path(), &[], &[]).await;
    assert_eq!(mcp.err("list_voices", json!({})).await["structuredContent"]["error"]["code"], "unreachable");
}

/// Reads one JSON-RPC message from a streamable HTTP response (JSON or Server-Sent Events).
async fn http_message(res: reqwest::Response) -> Value {
    let is_sse = res.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream");
    let mut stream = res;
    let mut buf = String::new();
    loop {
        if !is_sse {
            buf.push_str(&stream.text().await.unwrap());
            return serde_json::from_str(&buf).unwrap();
        }
        while let Some(end) = buf.find("\n\n") {
            let block: String = buf.drain(..end + 2).collect();
            let data: Vec<&str> = block.lines().filter_map(|l| l.strip_prefix("data:")).map(str::trim_start).collect();
            if let Ok(msg) = serde_json::from_str::<Value>(&data.join("\n"))
                && msg.get("id").is_some()
            {
                return msg;
            }
        }
        let chunk = tokio::time::timeout(WAIT, stream.chunk()).await.unwrap().unwrap().expect("stream ended");
        buf.push_str(&String::from_utf8_lossy(&chunk));
    }
}

#[tokio::test]
async fn streamable_http() {
    let api = api(None).await;
    let home = tempfile::tempdir().unwrap();
    let mut child = command(&api.url, home.path(), &["--http", "127.0.0.1:0"], &[]).spawn().unwrap();
    let mut stderr = BufReader::new(child.stderr.take().unwrap()).lines();
    let endpoint = tokio::time::timeout(WAIT, async {
        while let Some(line) = stderr.next_line().await.unwrap() {
            eprintln!("[mcp] {line}");
            if let Some(url) = line.split("MCP over HTTP at ").nth(1) {
                return url.trim().to_string();
            }
        }
        panic!("mimvox-mcp ended before it listened");
    })
    .await
    .unwrap();
    assert!(endpoint.starts_with("http://127.0.0.1:") && endpoint.ends_with("/mcp"), "{endpoint}");

    let client = reqwest::Client::new();
    let post = |body: Value, session: Option<&str>| {
        let mut req = client
            .post(&endpoint)
            .header("accept", "application/json, text/event-stream")
            .header("content-type", "application/json")
            .body(body.to_string());
        if let Some(s) = session {
            req = req.header("mcp-session-id", s).header("mcp-protocol-version", "2025-06-18");
        }
        req.send()
    };
    let init = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "mimvox-test", "version": "0" } } });
    let res = post(init, None).await.unwrap();
    assert_eq!(res.status(), 200);
    let session = res.headers().get("mcp-session-id").map(|v| v.to_str().unwrap().to_string());
    let msg = http_message(res).await;
    assert_eq!(msg["result"]["serverInfo"]["name"], "mimvox-mcp", "{msg}");
    let s = session.as_deref();
    let res = post(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }), s).await.unwrap();
    assert!(res.status().is_success(), "{}", res.status());

    let msg = http_message(post(json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }), s).await.unwrap()).await;
    assert_eq!(msg["result"]["tools"].as_array().unwrap().len(), TOOLS.len(), "{msg}");
    let call = json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": { "name": "list_voices", "arguments": {} } });
    let msg = http_message(post(call, s).await.unwrap()).await;
    assert_eq!(msg["result"]["structuredContent"]["voices"][0]["id"], "fake-voice", "{msg}");

    let (ok, _, err) = {
        let out = command(&api.url, home.path(), &["--http", "0.0.0.0:0"], &[]).output().await.unwrap();
        (out.status.success(), out.stdout, String::from_utf8_lossy(&out.stderr).to_string())
    };
    assert!(!ok && err.contains("--http-allow-remote"), "{err}");
}
