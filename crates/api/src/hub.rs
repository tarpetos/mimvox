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

enum Slot {
    Waiting(oneshot::Sender<Reply>),
    Arrived(Reply),
    Abandoned,
}

#[derive(Default)]
struct Pending {
    slots: HashMap<(u64, u64), Slot>,
    exited: HashSet<u64>,
}

struct Inner {
    engine: Mutex<Option<Arc<EngineClient>>>,
    pending: Mutex<Pending>,
    events: broadcast::Sender<Arc<EngineEvent>>,
    exits: mpsc::UnboundedSender<EngineExit>,
}

/// Async front of an [`EngineClient`]: requests answered by id, every event broadcast, exits reported. Survives engine
/// restarts: [`Hub::attach`] the next client and callers carry on.
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
        let weak = Arc::downgrade(&self.0);
        Arc::new(move |client, msg| {
            if let Some(inner) = weak.upgrade() {
                Hub(inner).on_msg(client, msg);
            }
        })
    }

    /// A client that already exited is dropped (releasing its data dir lock); its exit was reported already.
    pub fn attach(&self, client: EngineClient) {
        let dead = {
            let p = lock(&self.0.pending);
            if p.exited.contains(&client.id()) {
                Some(client)
            } else {
                *lock(&self.0.engine) = Some(Arc::new(client));
                None
            }
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
        let key = (client.id(), client.request(method, params));
        drop(client);
        let rx = {
            let mut p = lock(&self.0.pending);
            match p.slots.remove(&key) {
                Some(Slot::Arrived(reply)) => return reply,
                _ if p.exited.contains(&key.0) => return Err(exited_error()),
                _ => {
                    let (tx, rx) = oneshot::channel();
                    p.slots.insert(key, Slot::Waiting(tx));
                    rx
                }
            }
        };
        let _abandon = Abandon { hub: self, key };
        rx.await.unwrap_or_else(|_| Err(exited_error()))
    }

    fn emit(&self, event: &str, job: Option<String>, data: Value) {
        let json = json!({ "event": event, "job": job, "data": data }).to_string();
        let _ = self.0.events.send(Arc::new(EngineEvent { event: event.to_string(), job, data, json }));
    }

    fn on_msg(&self, client: u64, msg: EngineMsg) {
        match msg {
            EngineMsg::Ready(hello) => self.emit("ready", None, hello.raw),
            EngineMsg::Event(ev) => self.emit(&ev.event, ev.job, ev.data),
            EngineMsg::Response { id, result } => {
                let mut p = lock(&self.0.pending);
                match p.slots.remove(&(client, id)) {
                    Some(Slot::Waiting(tx)) => {
                        let _ = tx.send(result);
                    }
                    Some(Slot::Abandoned) => {}
                    Some(Slot::Arrived(_)) | None => {
                        p.slots.insert((client, id), Slot::Arrived(result));
                    }
                }
            }
            EngineMsg::Exited { code, stderr_tail } => {
                {
                    let mut p = lock(&self.0.pending);
                    p.exited.insert(client);
                    let keys: Vec<_> = p
                        .slots
                        .iter()
                        .filter(|(k, slot)| k.0 == client && !matches!(slot, Slot::Arrived(_)))
                        .map(|(k, _)| *k)
                        .collect();
                    for key in keys {
                        if let Some(Slot::Waiting(tx)) = p.slots.remove(&key) {
                            let _ = tx.send(Err(exited_error()));
                        }
                    }
                }
                let gone = {
                    let mut engine = lock(&self.0.engine);
                    if engine.as_ref().is_some_and(|e| e.id() == client) { engine.take() } else { None }
                };
                drop(gone);
                self.emit("engine.exited", None, json!({ "code": code }));
                let _ = self.0.exits.send(EngineExit { client, code, stderr_tail });
            }
        }
    }
}

/// Marks the slot of a call whose future was dropped, so its late answer is discarded.
struct Abandon<'a> {
    hub: &'a Hub,
    key: (u64, u64),
}

impl Drop for Abandon<'_> {
    fn drop(&mut self) {
        let mut p = lock(&self.hub.0.pending);
        if let Some(Slot::Waiting(_)) = p.slots.get(&self.key) {
            p.slots.insert(self.key, Slot::Abandoned);
        }
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
