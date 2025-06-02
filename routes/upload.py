from fastapi import APIRouter, File, UploadFile, HTTPException
from rag.indexer import index_file_to_opensearch

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    ✅ 파일 업로드 API
    - 업로드된 파일을 OpenSearch에 인덱싱
    - 지원 포맷: .txt, .pdf
    - 성공 시 메시지 반환
    """
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    try:
        content = await file.read()
        indexed = index_file_to_opensearch(file.filename, content)
        return {"status": "success", "chunks_indexed": indexed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 업로드 중 오류 발생: {str(e)}")