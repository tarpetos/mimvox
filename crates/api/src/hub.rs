use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use mimvox_engine_client::protocol::EngineError;
use mimvox_engine_client::{EngineClient, EngineMsg, EngineSink};
use serde_json::{Value, json};
use tokio::sync::{broadcast, mpsc, oneshot};

const EVENT_CAPACITY: usize = 4096;
/// The engine reads stdin lines of up to 64 MiB and treats a longer one as the end of its input.
pub const MAX_REQUEST_LINE: usize = 32 << 20;

/// One engine event, with the JSON line `/events` sends for it.
#[derive(Debug)]
pub struct EngineEvent {
    pub event: String,
    pub job: Option<String>,
    pub data: Value,
    pub json: String,
}

#[derive(Clone, Debug)]
pub struct EngineExit {
    pub client: u64,
    pub code: Option<i32>,
    pub stderr_tail: String,
}

pub type Reply = Result<Value, EngineError>;

#[derive(Default)]
struct Pending {
    slots: HashMap<(u64, u64), oneshot::Sender<Reply>>,
    exited: HashSet<u64>,
}

struct Inner {
    engine: Mutex<Option<Arc<EngineClient>>>,
    pending: Mutex<Pending>,
    events: broadcast::Sender<Arc<EngineEvent>>,
    exits: mpsc::UnboundedSender<EngineExit>,
}

/// Async front of an [`EngineClient`]: requests answered by id, every event broadcast, exits reported. Survives engine
/// restarts: [`Hub::attach`] the next client and callers carry on. The client may be shared with others that send
/// their own requests ([`Hub::tee_sink`]): answers to those are left to them.
#[derive(Clone)]
pub struct Hub(Arc<Inner>);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn exited_error() -> EngineError {
    EngineError::new("internal", "the engine exited before it answered")
}

impl Hub {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<EngineExit>) {
        let (events, _) = broadcast::channel(EVENT_CAPACITY);
        let (exits, exits_rx) = mpsc::unbounded_channel();
        let inner = Inner { engine: Mutex::new(None), pending: Mutex::default(), events, exits };
        (Self(Arc::new(inner)), exits_rx)
    }

    /// The sink to spawn a client with; it holds the hub weakly, so a dropped hub ends nothing else.
    pub fn sink(&self) -> EngineSink {
        self.tee_sink(|_, _| {})
    }

    /// A sink that shows every message to the hub, then hands it to `then`: one engine for the hub and another user.
    pub fn tee_sink(&self, then: impl Fn(u64, EngineMsg) + Send + Sync + 'static) -> EngineSink {
        let weak = Arc::downgrade(&self.0);
        Arc::new(move |client, msg| {
            if let Some(inner) = weak.upgrade() {
                Hub(inner).observe(client, &msg);
            }
            then(client, msg);
        })
    }

    /// A client that already exited is dropped (releasing its data dir lock); its exit was reported already.
    pub fn attach(&self, client: EngineClient) {
        self.attach_shared(Arc::new(client));
    }

    /// [`Hub::attach`] for a client its owner keeps too; the hub lets go of it when it exits or on [`Hub::take_engine`].
    pub fn attach_shared(&self, client: Arc<EngineClient>) {
        let dead = {
            let p = lock(&self.0.pending);
            if p.exited.contains(&client.id()) { Some(client) } else { lock(&self.0.engine).replace(client) }
        };
        drop(dead);
    }

    pub fn take_engine(&self) -> Option<Arc<EngineClient>> {
        lock(&self.0.engine).take()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<EngineEvent>> {
        self.0.events.subscribe()
    }

    pub async fn call(&self, method: &str, params: Value) -> Reply {
        let size = method.len() + serde_json::to_string(&params).map_or(0, |s| s.len());
        if size > MAX_REQUEST_LINE {
            return Err(EngineError::new("too_large", format!("requests are limited to {MAX_REQUEST_LINE} bytes")));
        }
        let client =
            lock(&self.0.engine).clone().ok_or_else(|| EngineError::new("not_ready", "the engine is not running"))?;
        let key = (client.id(), client.reserve_id());
        let rx = {
            let mut p = lock(&self.0.pending);
            if p.exited.contains(&key.0) {
                return Err(exited_error());
            }
            let (tx, rx) = oneshot::channel();
            p.slots.insert(key, tx);
            rx
        };
        let _abandon = Abandon { hub: self, key };
        client.request_with_id(key.1, method, params);
        drop(client);
        rx.await.unwrap_or_else(|_| Err(exited_error()))
    }

    /// Builds nothing while no one listens.
    fn emit(&self, event: &str, job: Option<&String>, data: &Value) {
        if self.0.events.receiver_count() == 0 {
            return;
        }
        let json = json!({ "event": event, "job": job, "data": data }).to_string();
        let ev = EngineEvent { event: event.to_string(), job: job.cloned(), data: data.clone(), json };
        let _ = self.0.events.send(Arc::new(ev));
    }

    fn observe(&self, client: u64, msg: &EngineMsg) {
        match msg {
            EngineMsg::Ready(hello) => self.emit("ready", None, &hello.raw),
            EngineMsg::Event(ev) => self.emit(&ev.event, ev.job.as_ref(), &ev.data),
            EngineMsg::Response { id, result } => {
                let waiting = lock(&self.0.pending).slots.remove(&(client, *id));
                if let Some(tx) = waiting {
                    let _ = tx.send(result.clone());
                }
            }
            EngineMsg::Exited { code, stderr_tail } => {
                {
                    let mut p = lock(&self.0.pending);
                    p.exited.insert(client);
                    let keys: Vec<_> = p.slots.keys().filter(|k| k.0 == client).copied().collect();
                    for key in keys {
                        if let Some(tx) = p.slots.remove(&key) {
                            let _ = tx.send(Err(exited_error()));
                        }
                    }
                }
                let gone = {
                    let mut engine = lock(&self.0.engine);
                    if engine.as_ref().is_some_and(|e| e.id() == client) { engine.take() } else { None }
                };
                drop(gone);
                self.emit("engine.exited", None, &json!({ "code": code }));
                let _ = self.0.exits.send(EngineExit { client, code: *code, stderr_tail: stderr_tail.clone() });
            }
        }
    }
}

/// Forgets the slot of a call whose future was dropped, so its late answer is discarded.
struct Abandon<'a> {
    hub: &'a Hub,
    key: (u64, u64),
}

impl Drop for Abandon<'_> {
    fn drop(&mut self) {
        lock(&self.hub.0.pending).slots.remove(&self.key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn oversized_requests_never_reach_the_engine() {
        let (hub, _exits) = Hub::new();
        let err = hub.call("tts.generate", json!({ "text": "a".repeat(MAX_REQUEST_LINE) })).await.unwrap_err();
        assert_eq!(err.code, "too_large");
        let err = hub.call("hello", json!({})).await.unwrap_err();
        assert_eq!(err.code, "not_ready");
    }
}
