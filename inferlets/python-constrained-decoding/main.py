"""Grammar-constrained JSON generation — Python inferlet example.

Demonstrates two paths to typed JSON output:

1. **Schema string + dict** — :meth:`Context.generate_json` constrains
   generation to a JSON Schema string and returns a parsed ``dict``.
2. **Pydantic model** — :meth:`Context.generate_pydantic` derives the
   schema from a pydantic v2 ``BaseModel`` and returns a typed instance.

Both paths use ``Schema.json_schema(...)`` under the hood; the difference
is whether the user owns the schema as a string or as a Python class.

The inferlet returns the parsed object directly — ``session.send`` and the
``main`` return value both auto-serialize structured values, so no manual
``json.dumps`` is needed.
"""

import json

from inferlet import Model, Context, Sampler, runtime, session
from pydantic import BaseModel, Field


class Person(BaseModel):
    """Pydantic model — schema is derived from this class."""
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    email: str
    skills: list[str] = Field(min_length=1)


# Equivalent JSON Schema as a string (for the dict path).
PERSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "name":   {"type": "string", "minLength": 1},
        "age":    {"type": "integer", "minimum": 0, "maximum": 150},
        "email":  {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["name", "age", "email", "skills"],
})

SYSTEM_PROMPT = (
    "You are a helpful assistant that generates structured data. "
    "Output ONLY a raw JSON object — no markdown, no explanation."
)
DEFAULT_PROMPT = (
    "Generate a profile for a fictional software engineer named Alice."
)


async def main(input: dict):
    prompt = input.get("prompt", DEFAULT_PROMPT)
    max_tokens = int(input.get("max_tokens", 512))
    mode = input.get("mode", "dict")  # "dict" or "pydantic"

    model = Model.load(runtime.models()[0])
    ctx = Context(model)
    ctx.system(SYSTEM_PROMPT).user(prompt)

    if mode == "pydantic":
        # Typed path: schema is derived from `Person`; result is a typed instance.
        person = await ctx.generate_pydantic(
            Person,
            sampler=Sampler.argmax(),
            max_tokens=max_tokens,
        )
        # Use the typed object — that's the point of generate_pydantic:
        session.send(f"Hello {person.name}, age {person.age}.\n")
        session.send(f"Skills: {', '.join(person.skills)}\n")
        session.send("[done]")
        # Return the model directly — the bakery wrapper serializes it via
        # `model_dump_json()`.
        return person

    # Dict path: schema as a string; result is a parsed dict.
    person = await ctx.generate_json(
        Sampler.argmax(),
        schema=PERSON_SCHEMA,
        max_tokens=max_tokens,
    )
    session.send(f"Hello {person['name']}, age {person['age']}.\n")
    session.send(f"Skills: {', '.join(person['skills'])}\n")
    session.send("[done]")
    # Return the dict directly — the bakery wrapper auto-stringifies.
    return person
