import fastapi
import fastapi.middleware
import fastapi.middleware.cors
from fastapi.responses import JSONResponse
import fastapi.encoders
import uvicorn
import json
import util
import base64
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

app = fastapi.FastAPI()
app.add_middleware(
	fastapi.middleware.cors.CORSMiddleware,
	allow_origins=["*"],  # Allows all origins
	allow_credentials=True,
	allow_methods=["*"],  # Allows all methods
	allow_headers=["*"],  # Allows all headers
)

class ImageRequest(BaseModel):
    image: str
    
@app.post("/predict")
async def Predict(request : ImageRequest):
	img = request.image
	returnData = util.Predict(img)
	print(returnData)
	response = {
		"prediction": (str)(returnData["prediction"]),
		"probabilities": returnData["probabilities"]
	}
	print(response)
	return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(app, port=5000, host="127.0.0.1")
