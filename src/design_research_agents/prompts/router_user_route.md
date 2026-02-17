Choose route candidates for this request.
Return one JSON object with this shape:
{"tool_names":["<identifier>", "..."],"reason":"short rationale"}
`tool_names` must be a non-empty ordered list. The first entry is executed.
Legacy `selection` is accepted but do not use it unless required.
Do not include markdown or extra text.

$routes_block

User request:
$user_prompt
