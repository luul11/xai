from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import requests
from requests.exceptions import RequestException
import os
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
XAI_API_BASE = "https://api.x.ai/v1"


class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., description="消息列表")
    model: str = Field(..., description="模型ID")
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


async def stream_generator(response, stream):
    try:
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                if decoded_chunk.startswith("data: "):
                    yield f"data: {decoded_chunk[6:]}\n\n"
                else:
                    yield f"data: {json.dumps({'error': 'Invalid chunk format'})}\n\n"
    except RequestException as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        if stream:
            response.close()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    logger.info(f"收到的请求头: {dict(request.headers)}")  # 保留日志用于验证
    # 从自定义头获取 xAI 密钥
    api_key = request.headers.get("X-XAI-API-Key", "")  # 新增：自定义头
    if not api_key:
        logger.error("缺少X-XAI-API-Key头")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少X-XAI-API-Key头")

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SillyTavern-Proxy/1.0",
            "Accept": "text/event-stream" if req.stream else "application/json"
        }
        payload = req.dict(exclude_unset=True)
        filtered_payload = {k: v for k, v in payload.items() if k in ["messages", "model", "max_tokens", "temperature", "top_p", "stream"]}
        logger.info(f"转发到xAI的负载: {filtered_payload}")

        response = requests.post(
            f"{XAI_API_BASE}/chat/completions",
            headers=headers,
            json=filtered_payload,
            stream=req.stream,
            timeout=200
        )
        response.raise_for_status()

        if req.stream:
            return StreamingResponse(
                stream_generator(response, req.stream),
                media_type="text/event-stream"
            )
        else:
            try:
                return response.json()
            except json.JSONDecodeError:
                logger.error(f"无效的JSON响应: {response.text}")
                raise HTTPException(status_code=502, detail="上游服务器返回无效响应")

    except RequestException as e:
        error_detail = ""
        if e.response is not None:
            try:
                error_detail = e.response.json().get("error", e.response.text)
            except json.JSONDecodeError:
                error_detail = e.response.text[:500]
            status_code = e.response.status_code
        else:
            error_detail = str(e)
            status_code = 504
        logger.error(f"请求失败: {error_detail}")
        raise HTTPException(
            status_code=status_code,
            detail=f"xAI API错误: {error_detail}"
        )


# 保持原来的模型列表端点，新增三个grok-4系列模型到备用数据
@app.get("/v1/models")
async def get_models(request: Request):
    api_key = request.headers.get("X-XAI-API-Key", "")  # 新增：自定义头

    if not api_key:
        raise HTTPException(status_code=401, detail="缺少API密钥")

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{XAI_API_BASE}/models", headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        logger.warning(f"获取模型失败: {str(e)}，返回备用数据")
        return {
            "object": "list",
            "data": [
                {"id": "grok-3-beta", "object": "model", "created": 1744681729, "owned_by": "xAI"},
                {"id": "grok-3-mini-beta", "object": "model", "created": 1744681729, "owned_by": "xAI"},
                {"id": "grok-3-fast-beta", "object": "model", "created": 1744681729, "owned_by": "xAI"},
                {"id": "grok-3-mini-fast-beta", "object": "model", "created": 1744681729, "owned_by": "xAI"},
                # 新增的三个grok-4系列模型
                {"id": "grok-4-0709", "object": "model", "created": 1744681729, "owned_by": "xAI"},
                {"id": "grok-4-fast-non-reasoning", "object": "model", "created": 1744681729, "owned_by": "xAI"},
                {"id": "grok-4-fast-reasoning", "object": "model", "created": 1744681729, "owned_by": "xAI"}
            ]
        }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
