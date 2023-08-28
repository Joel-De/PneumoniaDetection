from __future__ import annotations
import argparse
import uvicorn
import torch

from model import PneumoniaDetectionModel
from fastapi import FastAPI, File, UploadFile
from data.dataset import PneumoniaDetectionDataset
import logging
import io
from PIL import Image
from typing import Annotated
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Annotated
import hashlib
from fastapi import Depends, FastAPI, HTTPException, status, Cookie
import database_interface as DBInterface
from database_interface import Doctor, Diagnosis, Patient
import asyncio
import psycopg
import datetime
from fastapi import Response
from pydantic import BaseModel
from enum import Enum
from typing import Optional
from fastapi.responses import JSONResponse
import uuid
import redis
from database_interface import DatabaseError

import json

app = FastAPI()

r = redis.Redis(host='localhost', port=6379, db=0)


class Session(BaseModel):
    user_info: Doctor
    time_created: datetime.datetime







class APIBodys:

    class Patient(BaseModel):
        first_name: str
        last_name: str
        sex: str
        age: int
        health_card_number: str

    class Doctor(BaseModel):
        username: str
        password: str  # this is a hashed password NOT plaintext
        location: str
        first_name: str
        last_name: str



def createUserSession(sessionUUID: uuid.UUID, session: Session) -> bool:
    sessionUUID = str(sessionUUID)
    if r.exists(sessionUUID):
        return False
    r.json().set(sessionUUID, '$', session.model_dump_json())
    r.expire(sessionUUID, 3600)
    return True


def getUserSession(sessionUUID: uuid.UUID) -> Session | None:

    if (userInfo := r.json().get(str(sessionUUID))) is None:
        return None

    return Session.model_validate(json.loads(userInfo))


def getSession(token: str):
    if (userInfo := getUserSession(token)) is None:
        return buildResponse(
            403,
            APIResponse(
                status=Responses.ERROR,
                message="Invalid session, or session has expired",
            ),
        )
    return userInfo


class Responses(str, Enum):
    SUCCESS: str = "success"
    ERROR: str = "error"


class APIResponse(BaseModel):
    status: Responses
    data: Optional[dict | BaseModel | list] = {}
    message: str


def buildResponse(statusCode: int, response: APIResponse):
    return JSONResponse(
        status_code=statusCode, content=response.model_dump(mode="python")
    )


@app.get("/")
def read_root():
    return {"Title": "Pneumonia Detection API"}


@app.post("/run_diagnosis")
# async def read_item(file: UploadFile, cookie : Annotated[str | None, Cookie()]):
async def runDiagnosis(file: UploadFile, patientUUID: uuid.UUID, cookie: str):
    contents = await file.read()

    if isinstance(session := getSession(cookie), JSONResponse):
        return session

    image = Image.open(io.BytesIO(contents))
    img = PneumoniaDetectionDataset.basicPreprocess(modelData["imgSize"])(image)
    img = img.type(torch.FloatTensor).unsqueeze(0).to(device)
    # result = model(img)
    result = torch.randn(2)
    outputClass = PneumoniaDetectionDataset.getClassMap()[result.argmax().item()]

    try:
        await DBInterface.addDiagnosis(
            Diagnosis(
                patient_uuid=patientUUID,
                doctor_uuid=session.user_info.doctor_uuid,
                diagnosis=result.argmax().item(),
                creation_date=datetime.datetime.now(),
            )
        )
    except DatabaseError as e:
        return buildResponse(statusCode=400, response=APIResponse(status=Responses.ERROR, message=str(e)))


    printMessage = (
        f"The patient has {outputClass}"
        if outputClass == "Pneumonia"
        else f"The patient is healthy!"
    )

    return buildResponse(statusCode=200, response=APIResponse(status=Responses.SUCCESS, message=printMessage))


@app.post("/login")
async def login(formData: Annotated[OAuth2PasswordRequestForm, Depends()]):
    if (userInfo := await DBInterface.getDoctor(formData.username)) is None:
        raise HTTPException(status_code=400, detail="Incorrect Username")

    hashed_password = hashlib.sha256(formData.password.encode("utf-8")).hexdigest()
    if not hashed_password == userInfo.password:
        raise HTTPException(status_code=400, detail="Incorrect password")

    sessionUUID = uuid.uuid4()

    response = buildResponse(
        200,
        APIResponse(
            status=Responses.SUCCESS,
            data={"cookie": str(sessionUUID)},  # delete this after no need !
            message="Successfully logged in",
        ),
    )

    createUserSession(sessionUUID, Session(user_info=userInfo, time_created=datetime.datetime.now()))
    response.set_cookie("sessionID", str(sessionUUID))

    return response


@app.post("/create_user")
async def createUser(
        userInfo:APIBodys.Doctor
):
    if len(userInfo.password) <= 6:  # primitive password strength check
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Please enter a password with more than 6 characters",
            ),
        )

    try:
        await DBInterface.addDoctor(
            Doctor(
                username=userInfo.username,
                password=hashlib.sha256(userInfo.password.encode("utf-8")).hexdigest(),
                first_name=userInfo.first_name,
                last_name=userInfo.last_name,
                patient_ids=[],
                location=userInfo.location,
                creation_date=datetime.datetime.now(),
            )
        )
        return APIResponse(
            status=Responses.SUCCESS, message="Successfully created user account!"
        )
    except psycopg.errors.UniqueViolation as e:
        logging.error(f"Duplicate violation when attempting to add user {userInfo.username}!")
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Username is already taken! Please choose another ",
            ),
        )


@app.post("/get_patients")
async def getPatients(
        cookie: str
):
    if isinstance(session := getSession(cookie), JSONResponse):
        return session

    try:
        patientList = await DBInterface.getAllPatients(session.user_info.username)
        return APIResponse(
            status=Responses.SUCCESS, data=patientList, message="Successfully fetched users active patients"
        )
    except psycopg.errors.UniqueViolation as e:
        logging.error(f"Error when fetching pateint list for user {session.user_info.username}!")
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Error when fetching patients!",
            ),
        )


@app.post("/add_patient")
async def addPatient( patientData:APIBodys.Patient, cookie: str
):
    if isinstance(session := getSession(cookie), JSONResponse):
        return session

    try:
        addedPatient = await DBInterface.addPatient(Patient(
                    first_name=patientData.first_name,
                    last_name=patientData.last_name,
                    doctor_uuid=session.user_info.doctor_uuid,
                    sex=patientData.sex,
                    age=patientData.age,
                    health_card_number=patientData.health_card_number,
                    creation_date=datetime.datetime.now(),
                ))

        await DBInterface.addPatientToDoctor(addedPatient.patient_uuid, session.user_info.doctor_uuid)
        return APIResponse(
            status=Responses.SUCCESS, message="Successfully added patient"
        )
    except psycopg.errors.UniqueViolation as e:
        logging.error(f"Error when adding patient that already exists: {patientData.first_name} {patientData.last_name}")
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Error when adding patient, already exists",
            ),
        )


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", help="Device to use for training")
    p.add_argument(
        "--load_model",
        type=str,
        default="static/prod.pth",
        help="Location of where the model you want to load is stored",
    )
    arguments = p.parse_args()
    return arguments


def loadModel():
    args = parseArgs()

    modelData = torch.load(args.load_model, map_location=torch.device("cpu"))
    model = PneumoniaDetectionModel()
    model.load_state_dict(modelData["model"])
    logging.info(f"Loaded {args.load_model}")
    model.to(args.device)
    model.eval()
    return model, modelData, args.device


model, modelData, device = loadModel()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # for windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # asyncio.run(DBInterface.initDB())

    uvicorn.run(app, host="0.0.0.0", port=8000)
