//! Runs mimvox-server against crates/desktop/examples/fake_engine.py.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

const WAIT: Duration = Duration::from_secs(20);

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

fn fake_engine_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../desktop/examples/fake_engine_src")
}

struct Setup {
    dir: tempfile::TempDir,
}

impl Setup {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("data")).unwrap();
        let python = python().display().to_string().replace('\\', "/");
        std::fs::write(dir.path().join("runtime.toml"), format!("[python]\nengine = \"{python}\"\n")).unwrap();
        Self { dir }
    }

    fn data(&self) -> PathBuf {
        self.dir.path().join("data")
    }

    fn command(&self, args: &[&str], env: &[(&str, &str)]) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_mimvox-server"));
        cmd.arg("--data-dir")
            .arg(self.data())
            .args(args)
            .env("MIMVOX_RUNTIME", self.dir.path().join("runtime.toml"))
            .env("MIMVOX_ENGINE_SRC", fake_engine_src())
            .env_remove("MIMVOX_API_TOKEN")
            .env_remove("MIMVOX_DATA")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd
    }

    async fn start(&self, args: &[&str], env: &[(&str, &str)]) -> Server {
        let mut args = args.to_vec();
        if !args.contains(&"--bind") {
            args.extend(["--bind", "127.0.0.1:0"]);
        }
        self.spawn(self.command(&args, env)).await
    }

    async fn spawn(&self, mut cmd: Command) -> Server {
        let mut child = cmd.spawn().unwrap();
        let (tx, mut lines) = mpsc::unbounded_channel();
        let mut stderr = BufReader::new(child.stderr.take().unwrap()).lines();
        tokio::spawn(async move {
            while let Ok(Some(line)) = stderr.next_line().await {
                eprintln!("[server] {line}");
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let base = tokio::time::timeout(WAIT, async {
            while let Some(line) = lines.recv().await {
                if let Some(rest) = line.split("listening on ").nth(1) {
                    return rest.split(' ').next().unwrap().to_string();
                }
            }
            panic!("the server ended before it listened");
        })
        .await
        .expect("the server did not start listening");
        Server { child, base, lines }
    }

    /// Runs the binary to its end: (exit success, stdout, stderr).
    async fn run(&self, args: &[&str], env: &[(&str, &str)]) -> (bool, String, String) {
        let out = tokio::time::timeout(WAIT, self.command(args, env).output()).await.unwrap().unwrap();
        (out.status.success(), String::from_utf8_lossy(&out.stdout).into(), String::from_utf8_lossy(&out.stderr).into())
    }
}

struct Server {
    child: Child,
    base: String,
    lines: mpsc::UnboundedReceiver<String>,
}

impl Server {
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    async fn rpc(&self, method: &str, params: Value) -> (StatusCode, Value) {
        let res = reqwest::Client::new()
            .post(self.url("/rpc"))
            .json(&json!({ "method": method, "params": params }))
            .send()
            .await
            .unwrap();
        (res.status(), res.json().await.unwrap())
    }

    async fn wait_line(&mut self, needle: &str) {
        tokio::time::timeout(WAIT, async {
            while let Some(line) = self.lines.recv().await {
                if line.contains(needle) {
                    return;
                }
            }
            panic!("the server ended before printing {needle:?}");
        })
        .await
        .unwrap_or_else(|_| panic!("no {needle:?} from the server"));
    }
}

/// Reads `GET /events` as JSON events.
struct Events {
    res: reqwest::Response,
    buf: String,
}

impl Events {
    async fn open(server: &Server) -> Self {
        let res = reqwest::get(server.url("/events")).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert!(res.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
        Self { res, buf: String::new() }
    }

    async fn next(&mut self) -> Value {
        loop {
            if let Some(end) = self.buf.find("\n\n") {
                let block: String = self.buf.drain(..end + 2).collect();
                let data: Vec<&str> =
                    block.lines().filter_map(|l| l.strip_prefix("data:")).map(str::trim_start).collect();
                if !data.is_empty() {
                    return serde_json::from_str(&data.join("\n")).unwrap();
                }
                continue;
            }
            let chunk = tokio::time::timeout(WAIT, self.res.chunk())
                .await
                .expect("no event in time")
                .unwrap()
                .expect("stream ended");
            self.buf.push_str(&String::from_utf8_lossy(&chunk));
        }
    }

    async fn until(&mut self, pred: impl Fn(&Value) -> bool) -> Vec<Value> {
        let mut seen = Vec::new();
        loop {
            let ev = self.next().await;
            let hit = pred(&ev);
            seen.push(ev);
            if hit {
                return seen;
            }
        }
    }
}

#[tokio::test]
async fn health_and_rpc() {
    let setup = Setup::new();
    let server = setup.start(&[], &[]).await;

    let health: Value = reqwest::get(server.url("/health")).await.unwrap().json().await.unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["engine"]["version"], "0.1.0-fake");

    let (status, body) = server.rpc("voices.list", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["voices"][0]["id"], "fake-voice");

    let (status, body) = server.rpc("no.such.method", json!({})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "not_found");
    assert!(body["message"].as_str().unwrap().contains("no.such.method"));

    let (status, body) = server.rpc("tts.generate", json!({ "voice_id": "nobody", "text": "Hi." })).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "not_found");

    let res = reqwest::Client::new()
        .post(server.url("/rpc"))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(res.json::<Value>().await.unwrap()["code"], "bad_request");
}

#[tokio::test]
async fn events_reach_every_subscriber() {
    let setup = Setup::new();
    let server = setup.start(&[], &[]).await;
    let (mut a, mut b) = (Events::open(&server).await, Events::open(&server).await);

    let (status, started) = server.rpc("tts.generate", json!({ "voice_id": "fake-voice", "text": "One. Two." })).await;
    assert_eq!(status, StatusCode::OK);
    let job = started["job"].as_str().unwrap().to_string();
    for events in [&mut a, &mut b] {
        let seen = events.until(|e| e["event"] == "tts.done" && e["job"] == job.as_str()).await;
        let chunks = seen.iter().filter(|e| e["event"] == "tts.chunk" && e["job"] == job.as_str()).count();
        assert_eq!(chunks, 2);
        assert!(seen.last().unwrap()["data"]["path"].as_str().unwrap().ends_with(".wav"));
    }
}

#[tokio::test]
async fn files_up_and_down() {
    let setup = Setup::new();
    let server = setup.start(&["--max-upload-mb", "1"], &[]).await;
    let client = reqwest::Client::new();
    let data = setup.data();

    let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let res = client.post(server.url("/files?name=../../my%20take.wav")).body(payload.clone()).send().await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let up: Value = res.json().await.unwrap();
    let path = PathBuf::from(up["path"].as_str().unwrap());
    assert_eq!(up["size_bytes"], 200_000);
    let uploads = std::fs::canonicalize(data.join("api").join("uploads")).unwrap();
    assert_eq!(std::fs::canonicalize(path.parent().unwrap()).unwrap(), uploads);
    assert!(path.file_name().unwrap().to_str().unwrap().ends_with("-my_take.wav"));

    let get = |p: &Path| client.get(server.url("/files")).query(&[("path", p.to_str().unwrap())]).send();
    let res = get(&path).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "audio/wav");
    assert_eq!(res.bytes().await.unwrap().as_ref(), payload.as_slice());

    // Refused while the body is still being sent: the client may see the connection close before the 413.
    // A client of its own: the server closes those connections, which a pooled client would reuse.
    let oversize = reqwest::Client::new();
    let declared = oversize.post(server.url("/files?name=big.bin")).body(vec![0u8; 1_500_000]).send().await;
    let streamed = futures_util::stream::iter((0..30).map(|_| Ok::<_, std::io::Error>(vec![0u8; 50_000])));
    let chunked =
        oversize.post(server.url("/files?name=big.bin")).body(reqwest::Body::wrap_stream(streamed)).send().await;
    for res in [declared, chunked].into_iter().flatten() {
        assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
    assert_eq!(std::fs::read_dir(data.join("api").join("uploads")).unwrap().count(), 1);

    std::fs::write(data.join("secret.txt"), "no").unwrap();
    std::fs::create_dir_all(data.join("outputs")).unwrap();
    let outside = [
        data.join("secret.txt"),
        data.join("outputs").join("..").join("secret.txt"),
        data.join("api").join("uploads").join("..").join("..").join("secret.txt"),
        setup.dir.path().join("runtime.toml"),
        PathBuf::from("//attacker/share/x.wav"),
        data.join("outputs").join("..").join("..").join("..").join("..").join("etc").join("passwd"),
    ];
    for p in &outside {
        assert_eq!(get(p).await.unwrap().status(), StatusCode::FORBIDDEN, "{}", p.display());
    }
    let res = client.get(server.url("/files")).query(&[("path", "outputs/x.wav")]).send().await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(get(&data.join("outputs").join("missing.wav")).await.unwrap().status(), StatusCode::NOT_FOUND);
    assert_eq!(get(&data.join("outputs")).await.unwrap().status(), StatusCode::BAD_REQUEST);

    let res =
        client.post(server.url("/files")).query(&[("name", "書".repeat(300) + ".mp3")]).body("x").send().await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let up: Value = res.json().await.unwrap();
    let name = Path::new(up["path"].as_str().unwrap()).file_name().unwrap().to_str().unwrap().to_string();
    assert!(name.len() <= 240 && name.ends_with(".mp3"), "{name}");

    let rpc = reqwest::Client::new()
        .post(server.url("/rpc"))
        .header("content-type", "application/json")
        .body(format!(r#"{{"method":"text.analyze","params":{{"text":"{}"}}}}"#, "a".repeat(17 << 20)))
        .send()
        .await;
    if let Ok(res) = rpc {
        assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(res.json::<Value>().await.unwrap()["code"], "too_large");
    }

    #[cfg(unix)]
    {
        let link = data.join("outputs").join("escape.txt");
        std::os::unix::fs::symlink(data.join("secret.txt"), &link).unwrap();
        assert_eq!(get(&link).await.unwrap().status(), StatusCode::FORBIDDEN);
    }
}

#[tokio::test]
async fn book_segment_timing_files() {
    let setup = Setup::new();
    let server = setup.start(&[], &[]).await;
    let client = reqwest::Client::new();
    let data = setup.data();
    let book = data.join("books").join("my-book");
    let segments = book.join("segments");
    let timing = r#"{"segments":[]}"#;
    for (path, body) in [
        (segments.join("001.json"), timing),
        (book.join("book.json"), "{}"),
        (book.join("state.json"), "{}"),
        (book.join("work").join("001.segments.json"), "{}"),
        (book.join("stats").join("001.json"), "{}"),
        (data.join("secret.txt"), "no"),
    ] {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();
    }
    let get = |p: &Path| client.get(server.url("/files")).query(&[("path", p.to_str().unwrap())]).send();

    let res = get(&segments.join("001.json")).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "application/json");
    assert_eq!(res.text().await.unwrap(), timing);
    assert_eq!(get(&segments.join("002.json")).await.unwrap().status(), StatusCode::NOT_FOUND);

    let refused = [
        book.join("book.json"),
        book.join("state.json"),
        book.join("work").join("001.segments.json"),
        book.join("stats").join("001.json"),
        segments.join("..").join("book.json"),
        segments.join("..").join("work").join("001.segments.json"),
        segments.join("..").join("..").join("..").join("secret.txt"),
        segments.join("..").join("..").join("..").join("..").join("runtime.toml"),
    ];
    for p in &refused {
        assert_eq!(get(p).await.unwrap().status(), StatusCode::FORBIDDEN, "{}", p.display());
    }

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(book.join("book.json"), segments.join("003.json")).unwrap();
        assert_eq!(get(&segments.join("003.json")).await.unwrap().status(), StatusCode::FORBIDDEN);
        std::fs::create_dir_all(data.join("elsewhere")).unwrap();
        std::fs::write(data.join("elsewhere").join("001.json"), "no").unwrap();
        let evil = data.join("books").join("evil");
        std::fs::create_dir_all(&evil).unwrap();
        std::os::unix::fs::symlink(data.join("elsewhere"), evil.join("segments")).unwrap();
        assert_eq!(get(&evil.join("segments").join("001.json")).await.unwrap().status(), StatusCode::FORBIDDEN);
    }
}

#[tokio::test]
async fn auth() {
    let setup = Setup::new();
    let server = setup.start(&["--require-token"], &[("MIMVOX_API_TOKEN", "s3cret-token")]).await;
    let client = reqwest::Client::new();
    let health = |auth: Option<&str>| {
        let mut req = client.get(server.url("/health"));
        if let Some(a) = auth {
            req = req.header("authorization", a);
        }
        req.send()
    };
    let res = health(None).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(res.headers()["www-authenticate"], "Bearer");
    assert_eq!(res.json::<Value>().await.unwrap()["code"], "unauthorized");
    assert_eq!(health(Some("Bearer wrong")).await.unwrap().status(), StatusCode::UNAUTHORIZED);
    assert_eq!(health(Some("s3cret-token")).await.unwrap().status(), StatusCode::UNAUTHORIZED);
    assert_eq!(health(Some("Bearer s3cret-token")).await.unwrap().status(), StatusCode::OK);
}

#[tokio::test]
async fn token_file_and_cli() {
    let setup = Setup::new();
    let (ok, token, _) = setup.run(&["token", "new"], &[]).await;
    assert!(ok);
    let token = token.trim().to_string();
    assert_eq!(token.len(), 43);
    let file = setup.data().join("api").join("token");
    assert_eq!(std::fs::read_to_string(&file).unwrap().trim(), token);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
    }
    let (ok, shown, _) = setup.run(&["--show-token"], &[]).await;
    assert!(ok);
    assert_eq!(shown.trim(), token);
    let (_, shown, _) = setup.run(&["token", "show"], &[("MIMVOX_API_TOKEN", "from-env")]).await;
    assert_eq!(shown.trim(), "from-env");

    let server = setup.start(&["--require-token"], &[]).await;
    let res = reqwest::Client::new().get(server.url("/health")).bearer_auth(&token).send().await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn loopback_without_token_refuses_web_pages() {
    let setup = Setup::new();
    let server = setup.start(&[], &[]).await;
    let client = reqwest::Client::new();
    let res = client.get(server.url("/health")).header("origin", "https://evil.example").send().await.unwrap();
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
    let res = client.get(server.url("/health")).header("host", "rebound.evil.example:8765").send().await.unwrap();
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
    let res = client.get(server.url("/health")).header("origin", "http://localhost:5173").send().await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    for (site, status) in [("cross-site", 403), ("same-site", 403), ("same-origin", 200), ("none", 200)] {
        let res = client.get(server.url("/health")).header("sec-fetch-site", site).send().await.unwrap();
        assert_eq!(res.status().as_u16(), status, "{site}");
    }
    let res = client
        .get(server.url("/files"))
        .query(&[("path", "//attacker/share/x.wav")])
        .header("sec-fetch-site", "cross-site")
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn non_loopback_needs_a_token() {
    let setup = Setup::new();
    let (ok, _, err) = setup.run(&["--bind", "0.0.0.0:0"], &[]).await;
    assert!(!ok);
    assert!(err.contains("refusing to listen on 0.0.0.0:0 without a token"), "{err}");

    let server = setup.start(&["--bind", "0.0.0.0:0"], &[("MIMVOX_API_TOKEN", "lan-token")]).await;
    let port = server.base.rsplit(':').next().unwrap().to_string();
    let url = format!("http://127.0.0.1:{port}/health");
    let client = reqwest::Client::new();
    assert_eq!(client.get(&url).send().await.unwrap().status(), StatusCode::UNAUTHORIZED);
    assert_eq!(client.get(&url).bearer_auth("lan-token").send().await.unwrap().status(), StatusCode::OK);
}

#[tokio::test]
async fn openai_speech() {
    let setup = Setup::new();
    let server = setup.start(&[], &[]).await;
    let client = reqwest::Client::new();
    let speech = |body: Value| client.post(server.url("/v1/audio/speech")).json(&body).send();

    let res = speech(
        json!({ "model": "tts-1", "input": "Hello there. Bye now.", "voice": "fake voice", "response_format": "wav" }),
    )
    .await
    .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "audio/wav");
    let wav = res.bytes().await.unwrap();
    assert_eq!(&wav[..4], b"RIFF");

    let res = speech(json!({ "model": "tts-1", "input": "Hi.", "voice": "fake-voice", "response_format": "wav" }))
        .await
        .unwrap();
    let wav = res.bytes().await.unwrap();
    let rate = u32::from_le_bytes(wav[24..28].try_into().unwrap()) as usize;
    let frames = u32::from_le_bytes(wav[40..44].try_into().unwrap()) as usize / 2;
    assert_eq!(rate, 44_100);
    let res = speech(json!({ "model": "tts-1", "input": "Hi.", "voice": "fake-voice", "response_format": "pcm" }))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "audio/pcm");
    let pcm = res.bytes().await.unwrap();
    assert_eq!(pcm.len(), 2 * (frames * 24_000).div_ceil(rate));
    assert_ne!(&pcm[..4], b"RIFF");

    let res = speech(json!({ "model": "tts-1", "input": "Hi.", "voice": "nobody" })).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let err: Value = res.json().await.unwrap();
    assert_eq!(err["error"]["type"], "invalid_request_error");
    assert_eq!(err["error"]["code"], "not_found");

    let res = speech(json!({ "model": "tts-1", "input": "Hi.", "voice": "fake-voice", "response_format": "midi" }))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    let models: Value = client.get(server.url("/v1/models")).send().await.unwrap().json().await.unwrap();
    assert_eq!(models["data"][0]["id"], "mimvox");
    let voices: Value = client.get(server.url("/v1/audio/voices")).send().await.unwrap().json().await.unwrap();
    assert_eq!(voices["voices"][0]["name"], "Fake voice");
}

#[tokio::test]
async fn speech_short_of_memory_is_503() {
    let setup = Setup::new();
    let server = setup.start(&[], &[("MIMVOX_FAKE_TTS_ERROR", "no_memory")]).await;
    let body = json!({ "model": "tts-1", "input": "Hi.", "voice": "fake-voice", "response_format": "wav" });
    let res = reqwest::Client::new().post(server.url("/v1/audio/speech")).json(&body).send().await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    let err: Value = res.json().await.unwrap();
    assert_eq!(err["error"]["code"], "no_memory");
    assert_eq!(err["error"]["type"], "server_error");
    assert_eq!(err["error"]["message"], "fake no_memory");
}

#[tokio::test]
async fn aac_needs_ffmpeg() {
    let setup = Setup::new();
    let empty = tempfile::tempdir().unwrap();
    let path = empty.path().display().to_string();
    let server = setup.start(&[], &[("PATH", &path), ("MIMVOX_FFMPEG", "")]).await;
    let body = json!({ "model": "tts-1", "input": "Hi.", "voice": "fake-voice", "response_format": "aac" });
    let res = reqwest::Client::new().post(server.url("/v1/audio/speech")).json(&body).send().await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    let err: Value = res.json().await.unwrap();
    assert_eq!(err["error"]["code"], "not_ready");
    assert!(err["error"]["message"].as_str().unwrap().contains("ffmpeg"));
}

#[tokio::test]
async fn engine_restarts_after_it_exits() {
    let setup = Setup::new();
    let mut server = setup.start(&[], &[]).await;
    server.wait_line("ready").await;
    let mut events = Events::open(&server).await;
    let (status, _) = server.rpc("shutdown", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    let seen = events.until(|e| e["event"] == "ready").await;
    assert!(seen.iter().any(|e| e["event"] == "engine.exited" && e["data"]["code"] == 0));
    let ready = &seen.last().unwrap()["data"];
    assert_eq!(ready["version"], "0.1.0-fake");
    assert!(ready.get("languages").is_none(), "ready data is passed on as the engine sent it: {ready}");
    server.wait_line("ready").await;
    let (status, _) = server.rpc("voices.list", json!({})).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn one_engine_per_data_folder() {
    let setup = Setup::new();
    let _first = setup.start(&[], &[]).await;
    let (ok, _, err) = setup.run(&["--bind", "127.0.0.1:0"], &[]).await;
    assert!(!ok);
    assert!(err.contains("already running"), "{err}");
}

#[cfg(unix)]
#[tokio::test]
async fn sigterm_shuts_the_engine_down() {
    let setup = Setup::new();
    let mut server = setup.start(&[], &[]).await;
    server.wait_line("ready").await;
    let pid = server.child.id().unwrap().to_string();
    assert!(std::process::Command::new("kill").args(["-TERM", &pid]).status().unwrap().success());
    let status = tokio::time::timeout(WAIT, server.child.wait()).await.unwrap().unwrap();
    assert!(status.success());
    let log = std::fs::read_to_string(setup.data().join("logs").join("engine.log")).unwrap();
    assert!(log.contains("shutdown requested"), "{log}");
}

#[cfg(windows)]
#[tokio::test]
async fn ctrl_break_shuts_the_engine_down() {
    use windows_sys::Win32::System::Console::{AllocConsole, CTRL_BREAK_EVENT, GenerateConsoleCtrlEvent};
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    // The event reaches only processes on the sender's console, which the server inherits; this fails harmlessly
    // when the test already has one.
    // SAFETY: no arguments; it only gives this process a console.
    unsafe { AllocConsole() };
    let setup = Setup::new();
    let mut cmd = setup.command(&["--bind", "127.0.0.1:0"], &[]);
    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
    let mut server = setup.spawn(cmd).await;
    server.wait_line("ready").await;
    let pid = server.child.id().unwrap();
    // SAFETY: a plain Win32 call; the group is the server's own, so nothing else gets the event.
    let sent = unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid) };
    assert_ne!(sent, 0, "GenerateConsoleCtrlEvent: {}", std::io::Error::last_os_error());
    let status = tokio::time::timeout(WAIT, server.child.wait()).await.unwrap().unwrap();
    assert_eq!(status.code(), Some(0));
    let log = std::fs::read_to_string(setup.data().join("logs").join("engine.log")).unwrap();
    assert!(log.contains("shutdown requested"), "{log}");
    let lock = std::fs::File::open(setup.data().join("engine.lock")).unwrap();
    assert!(lock.try_lock().is_ok(), "engine.lock is still held");
}

#[tokio::test]
async fn speech_turn_outlives_a_client_that_leaves() {
    let setup = Setup::new();
    let mut server = setup.start(&[], &[]).await;
    server.wait_line("ready").await;
    let mut events = Events::open(&server).await;
    let long = "One. Two. Three. Four. Five. Six. Seven. Eight.";
    let gone = reqwest::Client::builder().timeout(Duration::from_millis(700)).build().unwrap();
    let res = gone
        .post(server.url("/v1/audio/speech"))
        .json(&json!({ "model": "tts-1", "input": long, "voice": "fake-voice", "response_format": "wav" }))
        .send()
        .await;
    assert!(res.is_err(), "the first request should time out on the client side");

    let res = reqwest::Client::new()
        .post(server.url("/v1/audio/speech"))
        .json(&json!({ "model": "tts-1", "input": "Hi.", "voice": "fake-voice", "response_format": "wav" }))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK, "{}", res.text().await.unwrap());

    let seen = events.until(|e| e["event"] == "tts.done").await;
    let first = seen.iter().find(|e| e["event"] == "tts.chunk").unwrap()["job"].clone();
    assert!(seen.iter().any(|e| e["event"] == "tts.error" && e["job"] == first && e["data"]["code"] == "cancelled"));
    assert_ne!(seen.last().unwrap()["job"], first);
    let ended = seen.iter().position(|e| e["event"] == "tts.error" && e["job"] == first).unwrap();
    let next = seen.iter().position(|e| e["event"] == "tts.chunk" && e["job"] != first).unwrap();
    assert!(ended < next, "the next job started before the abandoned one ended");

    let res = reqwest::Client::new()
        .post(server.url("/v1/audio/speech"))
        .json(&json!({ "model": "tts-1", "input": "a".repeat(4097), "voice": "fake-voice" }))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}
