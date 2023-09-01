from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import io
import json
import logging
import sys
import uuid
from enum import Enum
from typing import Annotated, Optional


import psycopg
import redis
import torch
import uvicorn
from PIL import Image
from fastapi import Depends, FastAPI, Cookie, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

import constants
import database_interface as DBInterface
from common.dataset import PneumoniaDetectionDataset
from database_interface import DatabaseError
from database_interface import Doctor, Diagnosis, Patient
from model import PneumoniaDetectionModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8003",
        "http://localhost:3000",
        "http://localhost:8003",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

r = redis.Redis(host=constants.REDIS_URL, port=constants.REDIS_PORT, db=0)


class Session(BaseModel):
    user_info: Doctor
    time_created: datetime.datetime


class APIBodys:
    class Patient(BaseModel):
        first_name: str = Field(
            Query(
                ...,
            )
        )
        last_name: str = Field(
            Query(
                ...,
            )
        )
        sex: str = Field(
            Query(
                ...,
            )
        )
        age: int = Field(
            Query(
                ...,
            )
        )
        health_card_number: str = Field(
            Query(
                ...,
            )
        )

    class Doctor(BaseModel):
        username: str
        password: str
        location: str
        first_name: str
        last_name: str


def createUserSession(sessionUUID: uuid.UUID, session: Session) -> bool:
    sessionUUID = str(sessionUUID)
    if r.exists(sessionUUID):
        return False
    r.json().set(sessionUUID, "$", session.model_dump_json())
    r.expire(sessionUUID, 3600)
    return True


def getUserSession(sessionUUID: uuid.UUID) -> Session | None:
    if (userInfo := r.json().get(str(sessionUUID))) is None:
        return None

    return Session.model_validate(json.loads(userInfo))


def removeUserSession(sessionUUID: uuid.UUID) -> Session | bool:
    sessionUUID = str(sessionUUID)
    if not r.exists(sessionUUID):
        return False

    return r.delete(sessionUUID)


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
async def runDiagnosis(
    file: UploadFile,
    patient_uuid: uuid.UUID,
    sessionID: Annotated[str | None, Cookie()],
):
    contents = await file.read()

    if isinstance(session := getSession(sessionID), JSONResponse):
        return session

    image = Image.open(io.BytesIO(contents))
    img = PneumoniaDetectionDataset.basicPreprocess(modelData["imgSize"])(image)
    img = img.type(torch.FloatTensor).unsqueeze(0).to(device)
    result = model(img)
    # result = torch.randn(2)
    outputClass = PneumoniaDetectionDataset.getClassMap()[result.argmax().item()]

    try:
        await DBInterface.addDiagnosis(
            Diagnosis(
                patient_uuid=patient_uuid,
                doctor_uuid=session.user_info.doctor_uuid,
                diagnosis=result.argmax().item(),
                creation_date=datetime.datetime.now(),
            )
        )
    except DatabaseError as e:
        return buildResponse(
            statusCode=400, response=APIResponse(status=Responses.ERROR, message=str(e))
        )

    printMessage = (
        f"The patient has {outputClass}"
        if outputClass == "Pneumonia"
        else f"The patient is healthy!"
    )

    return buildResponse(
        statusCode=200,
        response=APIResponse(
            status=Responses.SUCCESS,
            data={"result": result.argmax().item()},
            message=printMessage,
        ),
    )


@app.post("/login")
async def login(formData: Annotated[OAuth2PasswordRequestForm, Depends()]):
    if (userInfo := await DBInterface.getDoctor(formData.username)) is None:
        return buildResponse(
            statusCode=403,
            response=APIResponse(
                status=Responses.SUCCESS, message="Incorrect Username"
            ),
        )

    hashed_password = hashlib.sha256(formData.password.encode("utf-8")).hexdigest()
    if not hashed_password == userInfo.password:
        return buildResponse(
            statusCode=403,
            response=APIResponse(
                status=Responses.SUCCESS, message="Incorrect Password"
            ),
        )

    sessionUUID = uuid.uuid4()

    response = buildResponse(
        200,
        APIResponse(
            status=Responses.SUCCESS,
            data={"cookie": str(sessionUUID)},  # delete this after no need !
            message="Successfully logged in",
        ),
    )

    createUserSession(
        sessionUUID, Session(user_info=userInfo, time_created=datetime.datetime.now())
    )
    response.set_cookie("sessionID", str(sessionUUID))

    return response


@app.post("/logout")
async def logout(sessionID: Annotated[str | None, Cookie()]):
    if isinstance(session := getSession(sessionID), JSONResponse):
        return session

    removeUserSession(sessionID)

    return buildResponse(
        200,
        APIResponse(
            status=Responses.SUCCESS,
            message="Successfully logged user out!",
        ),
    )


@app.post("/create_user")
async def createUser(userInfo: APIBodys.Doctor):
    if len(userInfo.password) <= 6:  # primitive password strength check
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Password less than 6 characters",
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
        logging.error(
            f"Duplicate violation when attempting to add user {userInfo.username}!"
        )
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Username is already taken! Please choose another ",
            ),
        )


@app.get("/get_patients")
async def getPatients(sessionID: Annotated[str | None, Cookie()]):
    if isinstance(session := getSession(sessionID), JSONResponse):
        return session

    logging.info(
        f"Fetching all patients for doctor {session.user_info.first_name} {session.user_info.last_name}"
    )
    try:
        patientList = await DBInterface.getAllPatients(session.user_info.doctor_uuid)
        return APIResponse(
            status=Responses.SUCCESS,
            data=patientList,
            message="Successfully fetched users active patients",
        )
    except psycopg.errors.UniqueViolation as e:
        logging.error(
            f"Error when fetching pateint list for user {session.user_info.username}!"
        )
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Error when fetching patients!",
            ),
        )


@app.get("/patient_profile_picture")
async def getPatientProfilePicture(
    health_card_number: str, sessionID: Annotated[str | None, Cookie()]
):
    if isinstance(session := getSession(sessionID), JSONResponse):
        return session

    try:
        patient = await DBInterface.getPatientProfilePicture(health_card_number)
        return Response(content=patient, media_type="image/png")

    except psycopg.errors.UniqueViolation as e:
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Error when fetching patient picture",
            ),
        )


@app.post("/get_all_diagnosis")
async def getAllDiagnosis(
    patient_uuid: uuid.UUID, sessionID: Annotated[str | None, Cookie()]
):
    if isinstance(session := getSession(sessionID), JSONResponse):
        return session

    try:
        diagnosisList = await DBInterface.getAllDiagnosis(
            session.user_info.doctor_uuid, patient_uuid
        )
        return APIResponse(
            status=Responses.SUCCESS,
            data=diagnosisList,
            message=f"Successfully fetched diagnosis for patient {patient_uuid}",
        )
    except psycopg.errors.UniqueViolation as e:
        logging.error(f"Error when fetching diagnosis list for patient {patient_uuid}!")
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Error when fetching diagnosis!",
            ),
        )


@app.post("/add_patient")
async def addPatient(
    file: UploadFile,
    sessionID: Annotated[str | None, Cookie()],
    patientData: APIBodys.Patient = Depends(),
):
    # async def addPatient(patientData: APIBodys.Patient, cookie : str):
    if isinstance(session := getSession(sessionID), JSONResponse):
        return session

    try:
        addedPatient = await DBInterface.addPatient(
            Patient(
                first_name=patientData.first_name,
                last_name=patientData.last_name,
                doctor_uuid=session.user_info.doctor_uuid,
                sex=patientData.sex,
                age=patientData.age,
                health_card_number=patientData.health_card_number,
                creation_date=datetime.datetime.now(),
                patient_image=await file.read(),
            )
        )

        await DBInterface.addPatientToDoctor(
            addedPatient.patient_uuid, session.user_info.doctor_uuid
        )
        print(addedPatient)
        return buildResponse(
            statusCode=200,
            response=APIResponse(
                status=Responses.SUCCESS,
                data={"patient_uuid": str(addedPatient.patient_uuid)},
                message="Successfully added patient",
            ),
        )

    except psycopg.errors.UniqueViolation as e:
        logging.error(
            f"Error when adding patient that already exists: {patientData.first_name} {patientData.last_name}"
        )
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message="Error when adding patient, already exists",
            ),
        )


@app.post("/remove_patient")
async def removePatient(
    patient_uuid: uuid.UUID, sessionID: Annotated[str | None, Cookie()]
):
    if isinstance(session := getSession(sessionID), JSONResponse):
        return session

    try:
        await DBInterface.removePatientFromDoctor(
            patient_uuid, session.user_info.doctor_uuid
        )
        return buildResponse(
            statusCode=200,
            response=APIResponse(
                status=Responses.SUCCESS,
                data={"patient_uuid": str(patient_uuid)},
                message="Successfully removed patient",
            ),
        )

    except DatabaseError as e:
        logging.error(f"Error when removing patient: {str(e)}")
        return buildResponse(
            400,
            APIResponse(
                status=Responses.ERROR,
                message=f"Error when removing patient {str(e)}",
            ),
        )


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", help="Device to use for training")
    p.add_argument(
        "--load_model",
        type=str,
        default="./prod.pth",
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    model, modelData, device = loadModel()
    # for windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(DBInterface.initDB())

    uvicorn.run(app, host="0.0.0.0", port=8080)
