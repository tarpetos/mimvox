//! Follows `GET /events` for the whole run: every job's latest progress and its end, so a wait can start after the job.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::{Notify, broadcast, watch};

use crate::api::Api;

const MAX_JOBS: usize = 1000;
const RETRY_MAX: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
pub struct Event {
    pub event: String,
    pub job: Option<String>,
    pub data: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Running,
    Done,
    Paused,
    Error,
}

#[derive(Clone, Debug, Serialize)]
pub struct JobState {
    pub job: String,
    pub state: State,
    /// The last event of the job.
    pub event: String,
    /// The data of its latest progress event.
    pub progress: Option<Value>,
    /// The data of its end event.
    pub result: Option<Value>,
    #[serde(skip)]
    seq: u64,
}

impl JobState {
    pub fn ended(&self) -> bool {
        self.state != State::Running
    }

    /// A job the engine took with it when it exited.
    pub fn lost(job: &str, event: &str) -> Self {
        let result = json!({ "code": "internal", "message": "the engine exited" });
        Self { job: job.into(), state: State::Error, event: event.into(), progress: None, result: Some(result), seq: 0 }
    }
}

/// How an event ends its job, if it does.
pub fn end_state(event: &str) -> Option<State> {
    if event == "redub.transcript" || event.ends_with(".done") {
        Some(State::Done)
    } else if event.ends_with(".paused") {
        Some(State::Paused)
    } else if event.ends_with(".error") {
        Some(State::Error)
    } else {
        None
    }
}

/// Progress as (value, total) and a short message, for MCP progress notifications.
pub fn progress_of(ev: &Event, total_hint: Option<f64>) -> Option<(f64, Option<f64>, String)> {
    let d = &ev.data;
    let mut message = ev.event.clone();
    if let Some(stage) = d["stage"].as_str() {
        message.push_str(&format!(" {stage}"));
    }
    if let (Some(e), Some(n)) = (d["epoch"].as_u64(), d["total_epochs"].as_u64()) {
        message.push_str(&format!(", epoch {e}/{n}"));
    }
    if let (Some(c), Some(n)) = (d["chapter"].as_u64(), d["chapters"].as_u64()) {
        message.push_str(&format!(", chapter {}/{n}", c + 1));
    }
    if let Some(pct) = d["pct"].as_f64().or_else(|| d["percent"].as_f64()) {
        return Some((pct, Some(100.0), message));
    }
    let index = d["index"].as_f64()?;
    let total = d["segments"].as_f64().or(total_hint);
    if let Some(n) = total {
        message.push_str(&format!(" {}/{n}", index + 1.0));
    }
    Some((index + 1.0, total, message))
}

struct Jobs {
    map: HashMap<String, JobState>,
    seq: u64,
}

/// The stream's state after the latest attempt to open it.
#[derive(Clone)]
struct Conn {
    result: Result<(), String>,
    attempts: u64,
}

struct Inner {
    api: Arc<Api>,
    jobs: Mutex<Jobs>,
    live: broadcast::Sender<Event>,
    conn: watch::Sender<Conn>,
    wake: Notify,
}

#[derive(Clone)]
pub struct Tracker {
    inner: Arc<Inner>,
}

impl Tracker {
    /// Connects in the background and reconnects whenever the stream ends.
    pub fn start(api: Arc<Api>) -> Self {
        let (live, _) = broadcast::channel(4096);
        let (conn, _) = watch::channel(Conn { result: Err("not connected yet".into()), attempts: 0 });
        let inner = Arc::new(Inner {
            api,
            jobs: Mutex::new(Jobs { map: HashMap::new(), seq: 0 }),
            live,
            conn,
            wake: Notify::new(),
        });
        tokio::spawn(follow(inner.clone()));
        Self { inner }
    }

    /// Waits until the stream is open, so a job started next cannot end unseen. While it is not, this tries once more.
    pub async fn ready(&self, timeout: Duration) -> Result<(), String> {
        let mut rx = self.inner.conn.subscribe();
        let before = {
            let conn = rx.borrow_and_update();
            if conn.result.is_ok() {
                return Ok(());
            }
            conn.attempts
        };
        self.inner.wake.notify_one();
        let conn = match tokio::time::timeout(timeout, rx.wait_for(|c| c.result.is_ok() || c.attempts > before)).await {
            Ok(Ok(conn)) => conn.clone(),
            _ => self.inner.conn.borrow().clone(),
        };
        conn.result
    }

    pub fn is_connected(&self) -> bool {
        self.inner.conn.borrow().result.is_ok()
    }

    /// The live events from now on, and the job's state so far (in that order, so nothing falls between).
    pub fn watch(&self, job: &str) -> (broadcast::Receiver<Event>, Option<JobState>) {
        let rx = self.inner.live.subscribe();
        (rx, self.job(job))
    }

    pub fn job(&self, job: &str) -> Option<JobState> {
        self.inner.jobs.lock().unwrap().map.get(job).cloned()
    }
}

async fn follow(inner: Arc<Inner>) {
    let mut delay = Duration::from_secs(1);
    loop {
        let reason = match inner.api.events().await {
            Ok(res) => {
                set_conn(&inner, Ok(()));
                delay = Duration::from_secs(1);
                let mut stream = res.bytes_stream();
                // Bytes until a block is whole: a chunk can end inside a UTF-8 character.
                let mut buf = Vec::new();
                loop {
                    match stream.next().await {
                        Some(Ok(chunk)) => {
                            buf.extend_from_slice(&chunk);
                            while let Some(end) = buf.windows(2).position(|w| w == b"\n\n") {
                                let block: Vec<u8> = buf.drain(..end + 2).collect();
                                handle_block(&inner, &String::from_utf8_lossy(&block));
                            }
                        }
                        Some(Err(e)) => break e.to_string(),
                        None => break "the event stream ended".to_string(),
                    }
                }
            }
            Err(e) => e.to_string(),
        };
        set_conn(&inner, Err(reason));
        tokio::select! {
            _ = tokio::time::sleep(delay) => {}
            _ = inner.wake.notified() => {}
        }
        delay = (delay * 2).min(RETRY_MAX);
    }
}

fn set_conn(inner: &Inner, result: Result<(), String>) {
    inner.conn.send_modify(|c| {
        c.result = result;
        c.attempts += 1;
    });
}

fn handle_block(inner: &Inner, block: &str) {
    let data: Vec<&str> =
        block.lines().filter_map(|l| l.strip_prefix("data:")).map(|l| l.strip_prefix(' ').unwrap_or(l)).collect();
    if data.is_empty() {
        return;
    }
    let Ok(v) = serde_json::from_str::<Value>(&data.join("\n")) else { return };
    let ev = Event {
        event: v["event"].as_str().unwrap_or_default().to_string(),
        job: v["job"].as_str().map(str::to_string),
        data: v["data"].clone(),
    };
    record(inner, &ev);
    let _ = inner.live.send(ev);
}

fn record(inner: &Inner, ev: &Event) {
    let mut jobs = inner.jobs.lock().unwrap();
    // Job ids start again with a new engine, and the jobs of the old one are gone with it.
    match ev.event.as_str() {
        "ready" => jobs.map.clear(),
        "engine.exited" => {
            for st in jobs.map.values_mut().filter(|s| !s.ended()) {
                *st = JobState { seq: st.seq, ..JobState::lost(&st.job, &ev.event) };
            }
        }
        _ => {}
    }
    let Some(job) = &ev.job else { return };
    jobs.seq += 1;
    let seq = jobs.seq;
    let st = jobs.map.entry(job.clone()).or_insert_with(|| JobState {
        job: job.clone(),
        state: State::Running,
        event: String::new(),
        progress: None,
        result: None,
        seq,
    });
    st.seq = seq;
    st.event = ev.event.clone();
    match end_state(&ev.event) {
        Some(state) => {
            st.state = state;
            st.result = Some(ev.data.clone());
        }
        None => st.progress = Some(ev.data.clone()),
    }
    if jobs.map.len() > MAX_JOBS {
        let mut ended: Vec<(u64, String)> =
            jobs.map.values().filter(|s| s.ended()).map(|s| (s.seq, s.job.clone())).collect();
        ended.sort_unstable();
        for (_, id) in ended.into_iter().take(MAX_JOBS / 2) {
            jobs.map.remove(&id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(event: &str, job: Option<&str>, data: Value) -> Event {
        Event { event: event.into(), job: job.map(str::to_string), data }
    }

    #[test]
    fn ends() {
        for (e, s) in [
            ("tts.done", Some(State::Done)),
            ("redub.transcript", Some(State::Done)),
            ("voices.media.done", Some(State::Done)),
            ("train.paused", Some(State::Paused)),
            ("book.error", Some(State::Error)),
            ("job.error", Some(State::Error)),
            ("tts.chunk", None),
            ("book.chapter", None),
        ] {
            assert_eq!(end_state(e), s, "{e}");
        }
    }

    #[test]
    fn progress() {
        let p = progress_of(
            &ev("train.progress", Some("job-1"), json!({"stage":"train","pct":40.0,"epoch":3,"total_epochs":9})),
            None,
        );
        assert_eq!(p, Some((40.0, Some(100.0), "train.progress train, epoch 3/9".into())));
        let p = progress_of(&ev("tts.chunk", Some("job-1"), json!({"index":1})), Some(3.0));
        assert_eq!(p, Some((2.0, Some(3.0), "tts.chunk 2/3".into())));
        assert_eq!(progress_of(&ev("voice.warm", None, json!({"ok":true})), None), None);
    }
}
