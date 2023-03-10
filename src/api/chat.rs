use std::iter::once;

use actix_identity::Identity;
use actix_web::{
    web::{Data, Json},
    HttpResponse,
};
use async_openai::{
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs,
        ChatCompletionResponseMessage, CreateChatCompletionRequestArgs,
        CreateCompletionRequestArgs, Role,
    },
    Client,
};
use serde::{Deserialize, Serialize};

use crate::{
    api::error::ApiError,
    db::{users::get_user, Pool},
};

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub reply: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptStyle {
    Chat,
    Readme,
    Html,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    Text,
    Code,
}

#[derive(Deserialize, Serialize, Default, Clone)]
pub struct Code {
    pub html: Option<String>,
    pub css: Option<String>,
    pub js: Option<String>,
}
#[derive(Deserialize)]
pub struct ExampleRequest {
    pub prompt: String,
    pub code: Option<Code>,
    pub context: Option<Vec<ChatCompletionResponseMessage>>,
}

#[derive(Serialize, Default)]
pub struct ExampleResponse {
    pub code: Code,
    pub context: Option<Vec<ChatCompletionRequestMessage>>,
}

#[derive(Deserialize)]
pub struct EditRequest {
    pub instruction: String,
    pub input: Option<String>,
}

static SYSTEM: &str = r#"You are a system that provides working front-end code.\
The code must not depend on 3rd party libraries. \
The code must not use style attributes on html tags. \
You reply with code separated in markdown code blocks for HTML, CSS and JS.
You must use codeblocks with language specifiers, like ```js \
and must not include CSS or JavaScript in the html block. \
The user will ask for web components or any part or a web site."#;

pub async fn chat(
    pool: Data<Pool>,
    user_id: Identity,
    openai_client: Data<Option<Client>>,
    chat_request: Json<ChatRequest>,
) -> Result<HttpResponse, ApiError> {
    let mut conn = pool.get()?;
    let user = get_user(&mut conn, user_id.id().unwrap())?;
    if !user.is_admin {
        return Ok(HttpResponse::Unauthorized().finish());
    }
    if let Some(client) = &**openai_client {
        let request = CreateCompletionRequestArgs::default()
            .model("text-davinci-003")
            .prompt(&chat_request.prompt)
            .max_tokens(2048_u16)
            .build()?;

        let mut response = client.completions().create(request).await?;
        let reply = response.choices.pop().map(|r| r.text).unwrap_or_default();
        return Ok(HttpResponse::Ok().json(ChatResponse { reply }));
    };
    Ok(HttpResponse::NotImplemented().finish())
}

pub async fn explain(
    openai_client: Data<Option<Client>>,
    chat_request: Json<ChatRequest>,
) -> Result<HttpResponse, ApiError> {
    if let Some(client) = &**openai_client {
        let request = CreateCompletionRequestArgs::default()
            .model("text-davinci-003")
            .prompt(&chat_request.prompt)
            .max_tokens(2048_u16)
            .temperature(0.0)
            .build()?;

        let mut response = client.completions().create(request).await?;
        let reply = response.choices.pop().map(|r| r.text).unwrap_or_default();
        return Ok(HttpResponse::Ok().json(ChatResponse { reply }));
    };
    Ok(HttpResponse::NotImplemented().finish())
}

fn code_to_prompt(Code { html, css, js }: Code) -> String {
    let mut prompt = vec![];
    prompt.push("```html\n");
    if let Some(html) = html.as_ref() {
        prompt.push(html);
    };
    prompt.push("\n```");
    prompt.push("```css\n");
    if let Some(css) = css.as_ref() {
        prompt.push(css);
    };
    prompt.push("\n```");
    prompt.push("```js\n");
    if let Some(js) = js.as_ref() {
        prompt.push(js);
    };
    prompt.push("\n```");

    prompt.join("\n")
}

pub async fn generate_example(
    user_id: Identity,
    openai_client: Data<Option<Client>>,
    chat_request: Json<ExampleRequest>,
) -> Result<HttpResponse, ApiError> {
    if let Some(client) = &**openai_client {
        let ExampleRequest {
            context,
            prompt,
            code,
        } = chat_request.into_inner();

        let name = user_id.id().unwrap();
        println!("{prompt}\n---");
        let mut messages = match (context, code) {
            (None, None) => {
                let prompt = ChatCompletionRequestMessageArgs::default()
                    .role(Role::User)
                    .content(format!("Give me {}", prompt))
                    .name(&name)
                    .build()?;
                vec![
                    ChatCompletionRequestMessageArgs::default()
                        .role(Role::System)
                        .content(SYSTEM)
                        .name(&name)
                        .build()?,
                    prompt,
                ]
            }
            (None, Some(code)) => {
                let prompt = ChatCompletionRequestMessageArgs::default()
                    .role(Role::User)
                    .content(format!(
                        "Given the following code {}. Can you {prompt}",
                        code_to_prompt(code)
                    ))
                    .name(&name)
                    .build()?;
                vec![
                    ChatCompletionRequestMessageArgs::default()
                        .role(Role::System)
                        .content(SYSTEM)
                        .name(&name)
                        .build()?,
                    prompt,
                ]
            }
            (Some(messages), None) => {
                let prompt = ChatCompletionRequestMessageArgs::default()
                    .role(Role::User)
                    .content(format!("Can you {prompt}?",))
                    .name(&name)
                    .build()?;
                messages
                    .into_iter()
                    .map(|ChatCompletionResponseMessage { role, content }| {
                        ChatCompletionRequestMessage {
                            role,
                            content,
                            name: Some(name.clone()),
                        }
                    })
                    .chain(once(prompt))
                    .collect()
            }
            (Some(messages), Some(code)) => {
                let prompt = ChatCompletionRequestMessageArgs::default()
                    .role(Role::User)
                    .content(format!(
                        "I've modified the code to be {}. Can you {prompt}",
                        code_to_prompt(code)
                    ))
                    .name(&name)
                    .build()?;
                messages
                    .into_iter()
                    .map(|ChatCompletionResponseMessage { role, content }| {
                        ChatCompletionRequestMessage {
                            role,
                            content,
                            name: Some(name.clone()),
                        }
                    })
                    .chain(once(prompt))
                    .collect()
            }
        };
        let request = CreateChatCompletionRequestArgs::default()
            .model("gpt-3.5-turbo")
            .messages(messages.clone())
            .temperature(0.0)
            .build()?;

        let res = ExampleResponse::default();
        let mut response = client.chat().create(request).await?;
        let reply = if let Some(m) = response.choices.pop().map(|r| r.message) {
            m
        } else {
            return Ok(HttpResponse::Ok().json(res));
        };

        messages.push(ChatCompletionRequestMessage {
            role: reply.role,
            content: reply.content.clone(),
            name: Some(name),
        });
        let content = reply.content;

        println!("{content}");
        let reply_split = content.split("```");

        let mut response = ExampleResponse {
            context: Some(messages),
            ..Default::default()
        };
        for substring in reply_split {
            if let Some(x) = substring.strip_prefix("css") {
                response.code.css = Some(x.trim().to_string());
            } else if let Some(x) = substring.strip_prefix("html") {
                response.code.html = Some(x.trim().to_string());
            } else if let Some(x) = substring.strip_prefix("js") {
                response.code.js = Some(x.trim().to_string());
            }
        }

        return Ok(HttpResponse::Ok().json(response));
    };
    Ok(HttpResponse::NotImplemented().finish())
}
