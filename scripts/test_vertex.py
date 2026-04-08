from anthropic import AnthropicVertex
c = AnthropicVertex(project_id="dafny-sketcher", region="us-east5")
msg = c.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=10,
    messages=[{"role": "user", "content": "hi"}]
)
print(msg.content[0].text)
