from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os

from config import UPLOAD_FOLDER
from rag.vectorestore import allowed_file
from rag.vectorestore import add_file_to_index
from rag.pipeline import rag_search

import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception as e:
    print("pysqlite3 fallback unavailable:", e)


app= FastAPI()

templates =Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    print("home")
    return templates.TemplateResponse(request, 'index.html')




@app.get("/manage", response_class=HTMLResponse)
def manage(request:Request):
    return templates.TemplateResponse(request, 'manage.html')

@app.post("/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(..., alias="files[]")):
    print("Uploading files", flush=True)

    uploaded = []

    for file in files:

        if file and allowed_file(file.filename):

            filename = os.path.basename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Read uploaded file
            content = await file.read()

            # Save file
            with open(filepath, "wb") as f:
                f.write(content)

            # Add to vector index
            add_file_to_index(
                filename,
                content.decode("utf-8")
            )

            uploaded.append(filename)

    return {
        "uploaded": uploaded,
        "message": f"Uploaded {len(uploaded)} files successfully."
    }



@app.get("/list_files")
async def list_files(request:Request):
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.txt')]
    return {'files': files}



@app.post("/delete_file")
async def delete_file(request:Request):
    data= await request.json()
    filename = data.get("filename")
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return JSONResponse({"error","filename not found"}, status_code=404)
    os.remove(path)

    return {"deleted":filename, "message": "file deleted and index updated"}




@app.post("/search")
async def search_files(request:Request):
    data = await request.json()
    query= data.get("query","")
    k= int(data.get("top_k", 3))
    history = data.get("history", [])
    resp, code = rag_search(query, k, history)

    return JSONResponse(resp, status_code=code)









if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

