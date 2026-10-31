use shoji::{
    traits::backend::chat_message::ToolCallState as OutputToolCallState,
    types::basic::{ToolCall, Value, parse_lenient_json},
};

use crate::openai::stream_state::ToolCallChunk;

#[derive(Debug, Default)]
pub struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallState {
    pub fn merge(
        &mut self,
        chunk: ToolCallChunk,
    ) {
        if let Some(id) = chunk.id {
            self.id.push_str(&id);
        }
        if let Some(name) = chunk.name {
            self.name.push_str(&name);
        }
        if let Some(arguments) = chunk.arguments {
            self.arguments.push_str(&arguments);
        }
    }

    pub fn build(
        &self,
        finalize: bool,
    ) -> OutputToolCallState {
        if finalize && let Some(parsed) = parse_arguments(self) {
            return OutputToolCallState::Finished(ToolCall {
                identifier: Some(self.id.clone()),
                name: self.name.clone(),
                arguments: Value {
                    json: parsed.to_string(),
                },
            });
        }
        OutputToolCallState::Candidate(format!(
            "{{\"name\":{},\"arguments\":{}",
            serde_json::to_string(&self.name).expect("tool name serializes"),
            self.arguments,
        ))
    }
}

/// Arguments a provider streams for a tool that declares no parameters stay empty,
/// which is the same request as an empty object. Text that needs the bounded repair
/// pass models' output is also accepted here: a candidate that reaches the server
/// becomes a `malformed_tool_call` and the call never runs.
fn parse_arguments(state: &ToolCallState) -> Option<serde_json::Value> {
    if state.arguments.trim().is_empty() {
        return Some(serde_json::Value::Object(serde_json::Map::new()));
    }
    parse_lenient_json(&state.arguments)
}

#[cfg(test)]
mod tests {
    use shoji::{
        traits::backend::chat_message::ToolCallState as OutputToolCallState,
        types::session::chat::ChatReplyFinishReason,
    };

    use super::ToolCallState;
    use crate::openai::stream_state::{StreamChunk, StreamState, ToolCallChunk};

    fn finished_arguments(arguments: Option<String>) -> String {
        let mut state = StreamState::new();
        let _ = state.process_chunk(StreamChunk {
            tool_calls: vec![ToolCallChunk {
                index: 0,
                id: Some("call_1".to_string()),
                name: Some("now".to_string()),
                arguments,
            }],
            ..StreamChunk::default()
        });
        let output = state
            .process_chunk(StreamChunk {
                finish_reason: Some(ChatReplyFinishReason::ToolCalls),
                ..StreamChunk::default()
            })
            .expect("the finishing chunk produces output");
        match output.tool_calls.into_iter().next() {
            Some(OutputToolCallState::Finished(call)) => call.arguments.json,
            other => panic!("expected a finished tool call, got {other:?}"),
        }
    }

    #[test]
    fn empty_arguments_finalize_as_an_empty_object() {
        assert_eq!(finished_arguments(Some(String::new())), "{}");
    }

    #[test]
    fn absent_arguments_finalize_as_an_empty_object() {
        assert_eq!(finished_arguments(None), "{}");
    }

    #[test]
    fn repaired_arguments_finalize_instead_of_falling_back_to_a_candidate() {
        assert_eq!(
            finished_arguments(Some(r#"{ "content": "called "The Square Mile" home" }"#.to_string())),
            r#"{"content":"called \"The Square Mile\" home"}"#
        );
    }
}
