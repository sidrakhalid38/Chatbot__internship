from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(
    title="NexusChat Practice API",
    description="Learning FastAPI before building the real RAG API.",
    version="0.1.0",
)


# -----------------------------
# Request and response models
# -----------------------------
class EchoRequest(BaseModel):
    message: str
    repeat: Optional[int] = 1


class EchoResponse(BaseModel):
    original: str
    echoed: str
    times: int


class MathRequest(BaseModel):
    a: float
    b: float
    operation: str


class MathResponse(BaseModel):
    result: float
    expression: str


# -----------------------------
# GET /health
# -----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "NexusChat Practice API",
    }


# -----------------------------
# POST /echo
# -----------------------------
@app.post("/echo", response_model=EchoResponse)
def echo(request: EchoRequest):
    if request.repeat < 1 or request.repeat > 10:
        raise HTTPException(
            status_code=400,
            detail="repeat must be between 1 and 10",
        )

    echoed_text = " ".join([request.message] * request.repeat)

    return EchoResponse(
        original=request.message,
        echoed=echoed_text,
        times=request.repeat,
    )


# -----------------------------
# POST /calculate
# -----------------------------
@app.post("/calculate", response_model=MathResponse)
def calculate(request: MathRequest):
    operation = request.operation.lower().strip()

    if operation == "add":
        result = request.a + request.b
        expression = f"{request.a} + {request.b} = {result}"

    elif operation == "subtract":
        result = request.a - request.b
        expression = f"{request.a} - {request.b} = {result}"

    elif operation == "multiply":
        result = request.a * request.b
        expression = f"{request.a} * {request.b} = {result}"

    elif operation == "divide":
        if request.b == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot divide by zero",
            )

        result = request.a / request.b
        expression = f"{request.a} / {request.b} = {result}"

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown operation: {operation}. "
                "Use add, subtract, multiply, or divide."
            ),
        )

    return MathResponse(
        result=result,
        expression=expression,
    )


if __name__ == "__main__":
    uvicorn.run(
        "practice_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )